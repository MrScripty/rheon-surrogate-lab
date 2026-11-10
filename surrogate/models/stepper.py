"""A learned fluid stepper whose velocity stays divergence free by construction.

One pass takes the current state and the forcing of the next `outputs_per_pass`
output intervals, and returns that many future states. Velocity changes are the
curl of a predicted edge potential that is zero on the box walls, so every
output is exactly as divergence free and sealed as the input, whatever the
weights are. Untrained, the model returns its input unchanged.

The network works in grid units so one set of weights serves any cell spacing:

    velocity   cells per output interval     u * dt / h
    pressure   the same scale as velocity    p * dt^2 / (rho * h^2)
    viscosity  diffusion number              (mu / rho) * dt / h^2
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..fields import (FluidState, check_staggering, curl, faces_as_cell_channels, remove_mean,
                      seal_potential)
from .unet3d import UNet3d

# Network inputs besides forcing: six face velocities, tracer, fluid mask, diffusion number.
_STATE_CHANNELS = 9
_FACE_CHANNELS = 6


@dataclass(frozen=True)
class StepperConfig:
    outputs_per_pass: int = 1
    widths: tuple = (32, 64, 128)
    blocks_per_level: int = 2
    # Typical magnitudes in grid units, measured from the training data. They only
    # keep network inputs and outputs near one; they are not part of the physics.
    velocity_scale: float = 1.0
    velocity_change_scale: float = 1.0
    tracer_scale: float = 1.0
    tracer_change_scale: float = 1.0
    pressure_scale: float = 1.0
    force_scale: float = 1.0

    def __post_init__(self):
        if self.outputs_per_pass < 1:
            raise ValueError('outputs_per_pass must be at least 1.')
        object.__setattr__(self, 'widths', tuple(self.widths))

    def to_json(self):
        return {**asdict(self), 'widths': list(self.widths)}


def _per_field(values, like):
    """Broadcast one value per batch entry over a field tensor with the same batch dimension."""
    return values.to(like.dtype).reshape(-1, *([1] * (like.dim() - 1)))


class DivergenceFreeStepper(nn.Module):
    def __init__(self, config=StepperConfig()):
        super().__init__()
        self.config = config
        k = config.outputs_per_pass
        self.trunk = UNet3d(_STATE_CHANNELS + _FACE_CHANNELS * k, config.widths, config.blocks_per_level)
        features = config.widths[0]
        # Each head lands directly on its own staggered location; see fields.py for the shapes.
        self.potential_x = nn.Conv3d(features, k, (2, 2, 1), padding=(1, 1, 0))
        self.potential_y = nn.Conv3d(features, k, (2, 1, 2), padding=(1, 0, 1))
        self.potential_z = nn.Conv3d(features, k, (1, 2, 2), padding=(0, 1, 1))
        self.tracer_change = nn.Conv3d(features, k, 1)
        self.pressure = nn.Conv3d(features, k, 1)
        for head in (self.potential_x, self.potential_y, self.potential_z, self.tracer_change, self.pressure):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def outputs_per_pass(self):
        return self.config.outputs_per_pass

    def forward(self, state, forcing, scene):
        """Advance `state` by `outputs_per_pass` output intervals.

        state    FluidState with batch dimension: fields [B, ...].
        forcing  (fu, fv, fw): the velocity each interval's forcing adds, m/s, staggered
                 like the velocity, [B, outputs_per_pass, ...].
        scene    Scene with one value per batch entry.

        Returns a list of FluidState, one per output interval, in SI units.
        """
        check_staggering(state)
        config = self.config
        k = config.outputs_per_pass
        if any(component.shape[1] != k for component in forcing):
            raise ValueError(f'Forcing must cover {k} output intervals.')

        to_grid = scene.timestep / scene.spacing                       # m/s -> cells per interval
        pressure_to_grid = scene.timestep ** 2 / (scene.density * scene.spacing ** 2)
        diffusion = scene.viscosity / scene.density * scene.timestep / scene.spacing ** 2

        u, v, w = (component * _per_field(to_grid, component) for component in (state.u, state.v, state.w))
        velocity = faces_as_cell_channels(u, v, w) / config.velocity_scale
        tracer = state.tracer.unsqueeze(1) / config.tracer_scale
        fluid = torch.ones_like(tracer)
        diffusion_field = fluid * _per_field(torch.log10(diffusion.clamp_min(1e-12)) / 6.0, fluid)
        force = faces_as_cell_channels(*(component * _per_field(to_grid, component) for component in forcing))
        force = force.flatten(1, 2) / config.force_scale               # [B, 6k, Z, Y, X]

        features = self.trunk(torch.cat((velocity, tracer, fluid, diffusion_field, force), dim=1))

        ax, ay, az = seal_potential(self.potential_x(features), self.potential_y(features),
                                    self.potential_z(features))
        du, dv, dw = (change * config.velocity_change_scale for change in curl(ax, ay, az))
        # Total tracer is conserved in a sealed box, so the predicted change has zero mean.
        tracer_change = remove_mean(self.tracer_change(features)) * config.tracer_change_scale
        pressure = remove_mean(self.pressure(features)) * config.pressure_scale

        from_grid = 1.0 / to_grid
        outputs = []
        for index in range(k):
            outputs.append(FluidState(
                u=(u + du[:, index]) * _per_field(from_grid, u),
                v=(v + dv[:, index]) * _per_field(from_grid, v),
                w=(w + dw[:, index]) * _per_field(from_grid, w),
                tracer=state.tracer + tracer_change[:, index],
                pressure=pressure[:, index] / _per_field(pressure_to_grid, state.pressure)))
        return outputs
