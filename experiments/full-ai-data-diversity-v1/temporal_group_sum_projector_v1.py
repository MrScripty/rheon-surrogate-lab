"""Coarse phase using an exact ordered prescribed-force sum. No dense force history."""
import torch
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS

@torch.no_grad()
def phase_packet(self, fields, scene, width, packet):
    rate = 60 * scene['reference_substeps']
    if type(width) is not int or not 1 <= width <= 32 or rate not in (240, 480):
        raise ValueError('Explicit bounded local duration required')
    if set(fields) != set(FIELDS) or any(v.device.type != 'cuda' for v in fields.values()):
        raise ValueError('Complete current CUDA state required')
    counts = tuple(scene['counts'])
    shape, sizes = tuple(reversed(counts)), dimensions(counts)
    core = self.model
    core.prepare_geometry(counts)
    core._nested_enabled = True
    h, rho, dt = 1 / scene['cells_per_metre'], 1000., width / rate
    core.configure_transport(h, dt)
    addition = self.control_sum.sum_from_packet(counts, packet)
    current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], sizes)]
    advected = core.transport(current, counts, dt / h)
    entering = core.affine_ops.three(advected[:3], addition, counts)
    provisional = core.gather_ops.three(core, entering, entering, counts,
        'viscosity', .001 / rho * dt / h ** 2)
    previous = fields['pressure'].reshape(1, 1, *shape)
    rhs = core.gather_ops.ordered_rhs(core, provisional, shape, 'flux', 0, h * h, dt)
    history = core.gather_ops.three(core, [previous[0, 0]], None, counts,
        'momentum', (h * h) * (1 / (rho * h)), multiply_only=True, as_double=True)
    request = core.gather_ops.difference(core, history, rhs, shape, 'flux', 0, -1., rho / h)
    correction = core.pressure(request.reshape(1, 1, *shape), counts)
    pressure = core.affine_ops.binary('sum_gauge', previous, correction)
    # This uses the same fused arithmetic and wall handling before its final
    # native F32 cast, providing the unrounded anchor needed for gradients.
    velocity = core.gather_ops.three(core, [pressure[0, 0]], provisional, counts,
        'momentum', dt * (1 / (rho * h)), as_double=True)
    return dict(counts=counts, shape=shape, h=h, rho=rho, dt=dt,
        reference_dt_s=1 / rate, width=width, start_fine=None,
        current=fields, forcing=None, known_force_sum=addition, known_packet=packet,
        advected=advected, provisional=provisional, previous=previous,
        base_pressure=pressure, base_velocity=velocity, base_tracer=advected[3])

