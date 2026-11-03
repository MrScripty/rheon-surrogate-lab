"""Reduced learned transitions with coalesced known forcing and final fine stage.

Only restored learned flow/pressure operators advance the physical fields.
Grouping changes known control quadrature and stage duration, never supplies
native future fields, RHS labels, inverses or assigned spatial coefficients.
"""
import torch
from macrostep_backbone_v1 import FrozenMacroBackbone
from cuda_native_transfer_v1 import FIELDS
from recurrent_sampler_native import dimensions


class GroupedMacroBackbone(FrozenMacroBackbone):
    @torch.no_grad()
    def feature_states_grouped(self, fields, scene, start, widths, horizons=1):
        widths = tuple(widths)
        if widths not in ((1,) * 8, (3, 4, 1), (2, 2, 3, 1)) or horizons not in (1, 2, 3, 4):
            raise ValueError('Registered finite grouped stage schedules required')
        if set(fields) != set(FIELDS) or any(v.device.type != 'cuda' for v in fields.values()):
            raise ValueError('Complete current CUDA fields required')
        counts = tuple(scene['counts']); shape = tuple(reversed(counts)); model = self.model
        model.prepare_geometry(counts)
        h, rho = 1. / scene['cells_per_metre'], 1000.
        packet = self.packet(scene, start, horizons, 8).to('cuda')
        fine_external = self.controls.generate_from_packet(counts, packet)
        current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        blocks = []
        for lead in range(horizons):
            features = [dict(pressure=previous.flatten(), **{n: v.flatten() for n, v in zip(FIELDS[1:], current)})]
            offset = lead * 8
            for width in widths:
                dt = width / (60. * 8.)
                model.configure_transport(h, dt)
                # These are integrated known force inputs, not model outputs.
                # Width1 is an identity control with unchanged input bytes.
                external = fine_external[offset]
                for j in range(1, width):
                    external = [a + b for a, b in zip(external, fine_external[offset + j])]
                offset += width
                advected = model.transport(current, counts, dt / h)
                entering = model.affine_ops.three(advected[:3], external, counts)
                provisional = model.gather_ops.three(model, entering, entering, counts, 'viscosity', (.001 / rho * dt / h ** 2))
                rhs = model.gather_ops.ordered_rhs(model, provisional, shape, 'flux', 0, h * h, dt)
                scale = (h * h) * (1 / (rho * h))
                history = model.gather_ops.three(model, [previous[0, 0]], None, counts, 'momentum', scale, multiply_only=True, as_double=True)
                request = model.gather_ops.difference(model, history, rhs, shape, 'flux', 0, -1., rho / h)
                correction = model.pressure(request.reshape(1, 1, *shape), counts)
                pressure = model.affine_ops.binary('sum_gauge', previous, correction)
                velocity = model.gather_ops.three(model, [pressure[0, 0]], provisional, counts, 'momentum', dt * (1 / (rho * h)))
                emitted = dict(pressure=pressure.contiguous().flatten(), vx=velocity[0].flatten(), vy=velocity[1].flatten(),
                               vz=velocity[2].flatten(), tracer=advected[3].flatten())
                features.append(emitted)
                current, previous = [*velocity, advected[3]], pressure
            blocks.append(features)
        return blocks
