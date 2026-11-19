"""Generic affine operation fusion; restored learned heads and all arithmetic unchanged."""
import math
import types
import torch
from debug.no_scratch_fill_graph_v1 import NoScratchFillGraph
from generic_affine_execution_v1 import GenericAffineOps
from recurrent_sampler_native import dimensions
from cuda_divisor_generator_v1 import FIELDS


@torch.inference_mode()
def predict(self, fields, scene, step, horizons, external, substeps=1):
    counts = tuple(scene['counts']); shape = tuple(reversed(counts))
    if (horizons not in (1, 2, 3, 4) or substeps not in (1, 8) or not 0 < scene['cells_per_metre'] <= 1024
            or scene.get('sgs', False) or counts not in self.geometry_cache):
        raise ValueError('Prepared registered reference-regime geometry required')
    if set(fields) != set(FIELDS) or len(external) != horizons*substeps:
        raise ValueError('Complete current state and prescribed input block required')
    for name, expected_shape in [('pressure', shape), *zip(FIELDS[1:], dimensions(counts))]:
        value = fields[name]
        if (value.device != self.device or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                or value.numel() != math.prod(expected_shape) or not value.is_contiguous()):
            raise ValueError('Exact native dtype and contiguous CUDA state required')
    for block in external:
        if len(block) != 3 or any(v.device != self.device or v.dtype != torch.float64
                                or tuple(v.shape) != size for v, size in zip(block, dimensions(counts)[:3])):
            raise ValueError('Complete CUDA double prescribed velocities required')
    self.configure_transport(1 / scene['cells_per_metre'], 1 / (60 * substeps))
    current = [fields[name].reshape(size) for name, size in zip(FIELDS[1:], dimensions(counts))]
    previous = fields['pressure'].reshape(1, 1, *shape)
    dt, h, rho = 1/(60*substeps), 1/scene['cells_per_metre'], 1000.
    results = []
    for lead in range(horizons*substeps):
        advected = self.transport(current, counts, dt/h)
        entering = self.affine_ops.three(advected[:3], external[lead], counts)
        viscosity = [self.linear(entering, size, 'viscosity', c) for c, size in enumerate(dimensions(counts)[:3])]
        provisional = self.affine_ops.three(entering, viscosity, counts, (.001/rho*dt/h**2), wall=True)
        flux = [self.linear(provisional, shape, 'flux', 0, channel) for channel in range(3)]
        rhs = self.affine_ops.ordered_three(flux, h*h, dt)
        scale = (h*h)*(1/(rho*h))
        history_delta = [self.linear([previous[0, 0]], size, 'momentum', c) for c, size in enumerate(dimensions(counts)[:3])]
        history = self.affine_ops.three(None, history_delta, counts, scale, multiply_only=True, as_double=True, wall=True)
        action = self.linear(history, shape, 'flux', 0, sign=-1)
        request = self.affine_ops.binary('difference_scale', rhs, action, rho/h)
        correction = self.pressure(request.reshape(1, 1, *shape), counts)
        pressure = self.affine_ops.binary('sum_gauge', previous, correction)
        gradient = [self.linear([pressure[0, 0]], size, 'momentum', c) for c, size in enumerate(dimensions(counts)[:3])]
        velocity = self.affine_ops.three(provisional, gradient, counts, dt*(1/(rho*h)), wall=True)
        if (lead+1) % substeps == 0:
            results.append(dict(pressure=pressure.contiguous().flatten(),
                vx=velocity[0].flatten(), vy=velocity[1].flatten(), vz=velocity[2].flatten(), tracer=advected[3].flatten()))
        current, previous = [*velocity, advected[3]], pressure
    return results


class AffineGeneratorGraph(NoScratchFillGraph):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.affine_ops = GenericAffineOps(build)
        self.model.affine_ops = self.affine_ops
        self.model.predict = types.MethodType(predict, self.model)
        self.asset_identity.update(generic_pointwise_fusion=self.affine_ops.kernels.identity,
            learned_coefficients_unchanged=True, learned_gather_and_transport_order_unchanged=True,
            CPU_forward_unported=True)

    def close(self):
        try:
            super().close()
        finally:
            self.affine_ops.close()
