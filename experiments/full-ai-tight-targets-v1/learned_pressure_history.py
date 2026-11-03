"""Frozen learned pressure composition using only causal current pressure.

All spatial coefficients come from restored learned flux/momentum tensors.
This module contains no numerical pressure solve or analytical inverse.
"""
from functools import lru_cache
import torch
from recurrent_sampler_assets import csr
from recurrent_sampler_native import transport, spatial, forcing, dimensions


@lru_cache(maxsize=128)
def indices(source_shape, target_shape, offset):
    with torch.inference_mode(False):
        xyz = [(torch.arange(target_shape[2 - a]) + offset[a]).clamp(0, source_shape[2 - a] - 1)
               for a in range(3)]
    return xyz[2][:, None, None], xyz[1][None, :, None], xyz[0][None, None, :]


def linear(fields, target_shape, coefficients, head, selected_channel=None):
    rows, channels, offsets, weights = coefficients
    result = torch.zeros(target_shape, dtype=torch.float64)
    for index in range(int(rows[head]), int(rows[head + 1])):
        channel = int(channels[index])
        if selected_channel is not None and channel != selected_channel:
            continue
        field = fields[channel]
        sample = field[indices(tuple(field.shape), tuple(target_shape), tuple(offsets[index].tolist()))].double()
        result = result + weights[index] * sample
    return result


class LearnedPressureHistory:
    def __init__(self, base, mode='reuse', pressure_map=None):
        if mode not in ('direct', 'reuse'):
            raise ValueError('Declared learned pressure composition required')
        self.base = base
        self.mode = mode
        self.pressure_map = pressure_map
        self.negative_flux = csr(-base.flux.detach())
        self.asset_identity = dict(base.asset_identity,
            pressure_history_composition=dict(mode=mode, source='Current supplied pressure only',
                action='Negative learned flux composed with tied learned momentum; all weights inherited',
                new_trainable_parameters=0, numerical_solver_in_this_module=False,
                external_pressure_map_override=pressure_map is not None))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def rhs(self, velocity, counts, h, dt):
        shape = tuple(reversed(counts))
        total = torch.zeros(shape, dtype=torch.float64)
        for channel in range(3):
            value = linear(velocity, shape, self.base.coefficients['flux'], 0, channel)
            total = total + (h * h) * value
        return total / dt

    def action(self, pressure, counts, h, rho):
        fields = []
        scale = (h * h) * (1 / (rho * h))
        for component, shape in enumerate(dimensions(counts)[:3]):
            value = linear([pressure[0, 0]], shape, self.base.coefficients['momentum'], component) * scale
            wall = [slice(None)] * 3
            wall[2 - component] = [0, -1]
            value[tuple(wall)] = 0.
            fields.append(value)
        return linear(fields, tuple(reversed(counts)), self.negative_flux, 0)

    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons):
        if horizons not in (1, 2, 3, 4) or scene['cells_per_metre'] not in (8, 16) or scene.get('sgs', False):
            raise ValueError('Registered current-domain pressure composition required')
        counts = tuple(scene['counts'])
        current = self.base._guard(fields, counts)
        shape = tuple(reversed(counts))
        if 'g' + '_'.join(map(str, shape)) not in self.base.pressure_core.model.log_gains:
            raise ValueError('No learned pressure geometry for this shape')
        previous = torch.as_tensor(fields['pressure']).reshape(1, 1, *shape)
        dt, h, rho = 1 / 60, 1 / scene['cells_per_metre'], 1000.
        external = forcing(scene, step, horizons)
        results = []
        for lead in range(horizons):
            advected = transport(current, counts, self.base.coefficients['point'], self.base.mid_gate, self.base.mix, dt / h)
            entering = [(value.double() + addition).float() for value, addition in zip(advected[:3], external[lead])]
            provisional = spatial(0, entering, counts, self.base.coefficients['viscosity'], .001 / rho * dt / h**2, base=entering)
            rhs = self.rhs(provisional, counts, h, dt)
            residual = rhs - self.action(previous, counts, h, rho) if self.mode == 'reuse' else rhs
            q = (residual * (rho / h)).reshape(1, 1, *shape).contiguous()
            correction = self.pressure_map(q) if self.pressure_map is not None else self.base.pressure_core(q)
            pressure = previous + correction if self.mode == 'reuse' else correction
            pressure = pressure - pressure[:, :, :1, :1, :1]
            velocity = spatial(2, [pressure[0, 0]], counts, self.base.coefficients['momentum'], dt * (1 / (rho * h)), base=provisional)
            output = dict(pressure=pressure.contiguous().flatten(), vx=velocity[0].flatten(), vy=velocity[1].flatten(),
                          vz=velocity[2].flatten(), tracer=advected[3].flatten())
            results.append(output)
            current, previous = [*velocity, advected[3]], pressure
        return results

    def close(self):
        self.base.close()
