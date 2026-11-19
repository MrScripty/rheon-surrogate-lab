"""Explicit-duration frozen GPU phase and differentiable correction response.

The phase contains only current state and the complete known force path. Native
future states and RHS labels belong exclusively to an offline target builder.
The zero correction retains the qualified pressure-history baseline bytes.
"""
import torch

from temporal_frozen_history_v1 import TemporalFrozenBackbone, FIELDS
from temporal_control_packets_v1 import control_packet
from recurrent_sampler_native import dimensions
from coupled_macrostep_v1 import RestoredLinear, _RestoredPressure, _AnchoredAdd, wall
from coupled_macrostep_v2 import _OrderedReadout


class TemporalGroupProjector(TemporalFrozenBackbone):
    def __init__(self, *args):
        super().__init__(*args)
        self.linear = RestoredLinear(self.model)
        self.asset_identity.update(explicit_duration_group_projector=True,
            future_fields_used_in_candidate=False, trainable_weights_added=False)

    @torch.no_grad()
    def phase(self, fields, scene, start_fine, width):
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
        packet = torch.from_numpy(control_packet(scene, start_fine, width)).to('cuda')
        forcing = self.controls.generate_from_packet(counts, packet)
        addition = forcing[0]
        for row in forcing[1:]:
            addition = [a + b for a, b in zip(addition, row)]
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
            reference_dt_s=1 / rate, width=width, start_fine=start_fine,
            current=fields, forcing=forcing, known_packet=packet,
            advected=advected, provisional=provisional, previous=previous,
            base_pressure=pressure, base_velocity=velocity, base_tracer=advected[3])

    def emit(self, phase, velocity_delta=None, tracer_delta=None, *, float_carriers=True):
        shape, counts = phase['shape'], phase['counts']
        h, rho, dt = phase['h'], phase['rho'], phase['dt']
        if velocity_delta is None:
            velocity_delta = [torch.zeros_like(v, dtype=torch.float64) for v in phase['provisional']]
        delta = [wall(v.double(), c) for c, v in enumerate(velocity_delta)]
        parts = [self.linear(delta, shape, 'flux', 0, c) for c in range(3)]
        rhs = _OrderedReadout.apply(*parts, self.model, h * h, dt)
        response = _RestoredPressure.apply((rhs * (rho / h)).reshape(1, 1, *shape),
                                          self.model, counts)
        pressure = _AnchoredAdd.apply(phase['base_pressure'], response)
        velocity = []
        for c, (base, dv, size) in enumerate(zip(phase['base_velocity'], delta, dimensions(counts)[:3])):
            gradient = self.linear([response[0, 0]], size, 'momentum', c) * (dt * (1 / (rho * h)))
            velocity.append(wall(_AnchoredAdd.apply(base, dv + gradient), c))
        tracer = phase['base_tracer'].double()
        if tracer_delta is not None:
            tracer = _AnchoredAdd.apply(tracer, tracer_delta.double())
        if float_carriers:
            velocity = [v.float() for v in velocity]
            tracer = tracer.float()
        return dict(pressure=pressure.contiguous().flatten(),
            **{n: v.contiguous().flatten() for n, v in zip(FIELDS[1:], [*velocity, tracer])})
