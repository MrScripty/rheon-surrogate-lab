"""Retain ordered known fine forces without changing grouped learned dynamics."""
import torch
from coupled_macrostep_v2 import CoupledMacroBackbone
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


class ForceAwareCoupledBackbone(CoupledMacroBackbone):
    @torch.no_grad()
    def context(self, fields, scene, start):
        if set(fields) != set(FIELDS) or any(v.device.type != 'cuda' for v in fields.values()):
            raise ValueError('Complete current CUDA physical state required')
        counts = tuple(scene['counts']); shape = tuple(reversed(counts)); core = self.model
        core.prepare_geometry(counts)
        h, rho = 1. / scene['cells_per_metre'], 1000.
        external = self.controls.generate_from_packet(counts, self.packet(scene, start, 1, 8).to('cuda'))
        current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        features = [dict(pressure=previous.flatten(), **{n: v.flatten() for n, v in zip(FIELDS[1:], current)})]
        offset = 0
        for stage, width in enumerate((2, 2, 3, 1)):
            dt = width / (60. * 8.); core.configure_transport(h, dt)
            addition = external[offset]
            for j in range(1, width): addition = [a + b for a, b in zip(addition, external[offset+j])]
            offset += width
            advected = core.transport(current, counts, dt/h)
            entering = core.affine_ops.three(advected[:3], addition, counts)
            provisional = core.gather_ops.three(core, entering, entering, counts, 'viscosity', (.001/rho*dt/h**2))
            if stage == 3:
                return dict(counts=counts, shape=shape, h=h, rho=rho, dt=dt, previous=previous,
                            provisional=provisional, tracer=advected[3], features=features,
                            fine_forcing=external, forcing_group_widths=(2, 2, 3, 1))
            rhs = core.gather_ops.ordered_rhs(core, provisional, shape, 'flux', 0, h*h, dt)
            history = core.gather_ops.three(core, [previous[0, 0]], None, counts, 'momentum', (h*h)*(1/(rho*h)), multiply_only=True, as_double=True)
            request = core.gather_ops.difference(core, history, rhs, shape, 'flux', 0, -1., rho/h)
            correction = core.pressure(request.reshape(1, 1, *shape), counts)
            pressure = core.affine_ops.binary('sum_gauge', previous, correction)
            velocity = core.gather_ops.three(core, [pressure[0, 0]], provisional, counts, 'momentum', dt*(1/(rho*h)))
            features.append(dict(pressure=pressure.flatten(), **{n: v.flatten() for n, v in zip(FIELDS[1:], [*velocity, advected[3]])}))
            current, previous = [*velocity, advected[3]], pressure
        raise AssertionError('Final fine stage context must be returned')


@torch.no_grad()
def forcing_contrasts(context):
    """Four independent within-group temporal contrasts of known inputs."""
    external=context['fine_forcing'];sizes=dimensions(context['counts'])[:3]
    if len(external)!=8 or context['forcing_group_widths']!=(2,2,3,1):
        raise ValueError('Ordered eight-step forcing and registered grouping required')
    for step in external:
        if len(step)!=3 or any(v.device.type!='cuda' or v.dtype!=torch.float64 or tuple(v.shape)!=s for v,s in zip(step,sizes)):
            raise ValueError('Complete known CUDA F64 force carriers required')
    pairs=((0,1),(2,3),(4,6),(5,6))
    return [[external[a][c]-external[b][c] for c in range(3)] for a,b in pairs]
