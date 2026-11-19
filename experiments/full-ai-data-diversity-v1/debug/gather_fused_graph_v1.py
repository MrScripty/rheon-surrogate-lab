"""Complete generator with generic learned gather/readout composition."""
import math
import types
import torch
from debug.cube64_layout_transform_candidate_v1 import Cube64LayoutTransformCandidate
from learned_gather_affine_v1 import LearnedGatherAffine
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
        provisional = self.gather_ops.three(self, entering, entering, counts, 'viscosity', (.001/rho*dt/h**2))
        rhs = self.gather_ops.ordered_rhs(self, provisional, shape, 'flux', 0, h*h, dt)
        scale = (h*h)*(1/(rho*h))
        history = self.gather_ops.three(self, [previous[0,0]], None, counts, 'momentum', scale, multiply_only=True, as_double=True)
        request = self.gather_ops.difference(self, history, rhs, shape, 'flux', 0, -1., rho/h)
        correction = self.pressure(request.reshape(1, 1, *shape), counts)
        pressure = self.affine_ops.binary('sum_gauge', previous, correction)
        velocity = self.gather_ops.three(self, [pressure[0,0]], provisional, counts, 'momentum', dt*(1/(rho*h)))
        if (lead+1) % substeps == 0:
            results.append(dict(pressure=pressure.contiguous().flatten(),
                vx=velocity[0].flatten(), vy=velocity[1].flatten(), vz=velocity[2].flatten(), tracer=advected[3].flatten()))
        current, previous = [*velocity, advected[3]], pressure
    return results


class GatherFusedGraph(Cube64LayoutTransformCandidate):
    def __init__(self,pair,shared_pair,build,control_build):
        super().__init__(pair,shared_pair,build,control_build)
        self.gather_ops=LearnedGatherAffine(build)
        self.model.gather_ops=self.gather_ops
        self.model.predict=types.MethodType(predict,self.model)
        self.asset_identity.update(generic_gather_readout_fusion=self.gather_ops.kernels.identity,
            restored_sparse_metadata_and_weights_unchanged=True, floating_sum_cast_and_wall_order_preserved=True,
            native_numerical_oracle_in_candidate=False)

    def close(self):
        try:super().close()
        finally:self.gather_ops.close()
