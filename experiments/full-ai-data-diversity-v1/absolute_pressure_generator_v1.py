"""Full learned one/eight-stage generator using restored absolute pressure map.

Every flow advance remains. This changes learned-map composition and floating
evaluation order; it is a new accuracy candidate, not an exact optimization.
All pressure response/CSR/sampler weights remain managed-restored and frozen.
"""
import torch
from macrostep_backbone_v1 import FrozenMacroBackbone
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


class AbsolutePressureGenerator(FrozenMacroBackbone):
    def __init__(self, pair, shared_pair, build, control_build):
        super().__init__(pair, shared_pair, build, control_build)
        self.asset_identity.update(absolute_pressure_composition=True, pressure_history_branch_removed=True,
                                   actual_internal_stages='OriginalN1/N8, counted without reduction',
                                   learned_parameters_changed=False, native_numerical_oracle_in_candidate=False,
                                   field_accuracy_unverified=True, cpu_model_forward=False)

    @torch.no_grad()
    def predict(self, fields, scene, start, horizons, substeps):
        if substeps not in (1, 8) or horizons not in (1, 2, 3, 4) or set(fields) != set(FIELDS):
            raise ValueError('Original complete physical-output and internal-stage contract required')
        if any(v.device.type != 'cuda' for v in fields.values()): raise ValueError('All learned inference requiresCUDA')
        counts = tuple(scene['counts']); shape = tuple(reversed(counts)); model = self.model
        model.prepare_geometry(counts)
        h, dt, rho = 1./scene['cells_per_metre'], 1./(60.*substeps), 1000.
        model.configure_transport(h, dt)
        external = self.controls.generate_from_packet(counts, self.packet(scene, start, horizons, substeps).to('cuda'))
        current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], dimensions(counts))]
        result = []
        for index in range(horizons*substeps):
            advected = model.transport(current, counts, dt/h)
            entering = model.affine_ops.three(advected[:3], external[index], counts)
            provisional = model.gather_ops.three(model, entering, entering, counts, 'viscosity', (.001/rho*dt/h**2))
            rhs = model.gather_ops.ordered_rhs(model, provisional, shape, 'flux', 0, h*h, dt)
            # Same frozen learned response, applied to the absolute request.
            # No native solve, analytical inverse or physical coefficient fit.
            pressure = model.pressure((rhs*(rho/h)).reshape(1, 1, *shape), counts)
            velocity = model.gather_ops.three(model, [pressure[0, 0]], provisional, counts, 'momentum', dt*(1/(rho*h)))
            if (index+1) % substeps == 0:
                result.append(dict(pressure=pressure.contiguous().flatten(), vx=velocity[0].flatten(), vy=velocity[1].flatten(),
                                   vz=velocity[2].flatten(), tracer=advected[3].flatten()))
            current = [*velocity, advected[3]]
        return result

    @torch.no_grad()
    def predict_native(self, fields, scene, start, horizons, substeps):
        if any(v.device.type != 'cpu' for v in fields.values()): raise ValueError('Native caller fields required')
        uploaded = {n: v.to('cuda').contiguous() for n, v in fields.items()}
        return [{n: v.detach().cpu().contiguous() for n, v in output.items()}
                for output in self.predict(uploaded, scene, start, horizons, substeps)]
