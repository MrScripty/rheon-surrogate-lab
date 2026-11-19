"""Graph-backed complete inference for the changed absolute learned pressure map."""
import types
import torch
from debug.geometry_specialized_graph_v1 import GeometrySpecializedGraph, transport as specialized_transport
from fused_cuda_generator_v3 import FusedCudaGeneratorV3
from complete_native_entry_v1 import CompleteNativeEntry
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


@torch.inference_mode()
def predict(core, fields, scene, step, horizons, external, substeps=1):
    counts = tuple(scene['counts']); shape = tuple(reversed(counts))
    if substeps not in (1, 8) or horizons not in (1, 2, 3, 4) or len(external) != horizons*substeps:
        raise ValueError('Complete original physical-output/internal-stage contract required')
    if set(fields) != set(FIELDS) or any(v.device.type != 'cuda' for v in fields.values()):
        raise ValueError('Complete current CUDA fields required')
    h, dt, rho = 1./scene['cells_per_metre'], 1./(60.*substeps), 1000.
    core.configure_transport(h, dt)
    current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], dimensions(counts))]
    outputs = []
    for index in range(horizons*substeps):
        advected = core.transport(current, counts, dt/h)
        entering = core.affine_ops.three(advected[:3], external[index], counts)
        provisional = core.gather_ops.three(core, entering, entering, counts, 'viscosity', (.001/rho*dt/h**2))
        rhs = core.gather_ops.ordered_rhs(core, provisional, shape, 'flux', 0, h*h, dt)
        pressure = core.pressure((rhs*(rho/h)).reshape(1, 1, *shape), counts)
        velocity = core.gather_ops.three(core, [pressure[0, 0]], provisional, counts, 'momentum', dt*(1/(rho*h)))
        if (index+1) % substeps == 0:
            outputs.append(dict(pressure=pressure.contiguous().flatten(), vx=velocity[0].flatten(), vy=velocity[1].flatten(),
                                vz=velocity[2].flatten(), tracer=advected[3].flatten()))
        current = [*velocity, advected[3]]
    return outputs


class AbsolutePressureGraph(GeometrySpecializedGraph):
    def __init__(self, *args):
        super().__init__(*args)
        self.model.predict = types.MethodType(predict, self.model)
        def supported_transport(core, fields, counts, alpha):
            if tuple(counts) in core.geometry_functions:
                return specialized_transport(core, fields, counts, alpha)
            return FusedCudaGeneratorV3.transport(core, fields, counts, alpha)
        self.model.transport = types.MethodType(supported_transport, self.model)
        self.asset_identity.update(absolute_pressure_composition=True, pressure_history_branch_removed=True,
                                   learned_coefficients_unchanged=True, learned_arithmetic_unchanged=False,
                                   original_N1_N8_advances_retained=True, native_numerical_oracle_in_candidate=False)


class AbsolutePressureNativeEntry(CompleteNativeEntry):
    generator_class = AbsolutePressureGraph

    def __init__(self, *args):
        super().__init__(*args)
        self.asset_identity.update(absolute_pressure_composition=True, learned_arithmetic_unchanged=False,
                                   learned_coefficients_unchanged=True, accuracy_and_latency_require_new_qualification=True)
