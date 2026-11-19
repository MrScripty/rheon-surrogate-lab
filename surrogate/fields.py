"""Staggered (MAC) grid fields and the discrete operators every component shares.

A grid has Z, Y, X cells and tensors keep that axis order last:

    cell-centred scalar   [..., Z, Y, X]
    u, x-face velocity    [..., Z, Y, X+1]
    v, y-face velocity    [..., Z, Y+1, X]
    w, z-face velocity    [..., Z+1, Y, X]

A vector potential lives on cell edges:

    ax, x-edges           [..., Z+1, Y+1, X]
    ay, y-edges           [..., Z+1, Y, X+1]
    az, z-edges           [..., Z, Y+1, X+1]

Operators here work in grid units (differences between neighbours, no division by
the cell spacing). Callers own the conversion to physical units.
"""
from dataclasses import dataclass, fields, replace

import torch


@dataclass(frozen=True)
class FluidState:
    """One simulation state. Leading dimensions are free (batch, time, ...)."""
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    tracer: torch.Tensor
    pressure: torch.Tensor

    def map(self, function):
        return replace(self, **{f.name: function(getattr(self, f.name)) for f in fields(self)})

    def to(self, *args, **kwargs):
        return self.map(lambda tensor: tensor.to(*args, **kwargs))

    @property
    def grid(self):
        """Cell counts (Z, Y, X)."""
        return tuple(self.tracer.shape[-3:])


@dataclass(frozen=True)
class Scene:
    """Physical constants of one scene, SI units. Tensors are per batch entry."""
    spacing: torch.Tensor      # cell width, m
    timestep: torch.Tensor     # one output interval, s
    density: torch.Tensor      # kg/m^3
    viscosity: torch.Tensor    # dynamic viscosity, Pa s

    def to(self, *args, **kwargs):
        return replace(self, **{f.name: getattr(self, f.name).to(*args, **kwargs) for f in fields(self)})


def check_staggering(state):
    """Raise if the five fields do not describe one grid."""
    z, y, x = state.grid
    expected = {'u': (z, y, x + 1), 'v': (z, y + 1, x), 'w': (z + 1, y, x), 'pressure': (z, y, x)}
    for name, shape in expected.items():
        actual = tuple(getattr(state, name).shape[-3:])
        if actual != shape:
            raise ValueError(f'{name} has grid shape {actual}; a {z}x{y}x{x} cell grid needs {shape}')


def divergence(u, v, w):
    """Net outflow of each cell."""
    return (u[..., :, :, 1:] - u[..., :, :, :-1]
            + v[..., :, 1:, :] - v[..., :, :-1, :]
            + w[..., 1:, :, :] - w[..., :-1, :, :])


def curl(ax, ay, az):
    """Face velocities of an edge potential. Their divergence is exactly zero."""
    u = (az[..., :, 1:, :] - az[..., :, :-1, :]) - (ay[..., 1:, :, :] - ay[..., :-1, :, :])
    v = (ax[..., 1:, :, :] - ax[..., :-1, :, :]) - (az[..., :, :, 1:] - az[..., :, :, :-1])
    w = (ay[..., :, :, 1:] - ay[..., :, :, :-1]) - (ax[..., :, 1:, :] - ax[..., :, :-1, :])
    return u, v, w


def seal_potential(ax, ay, az):
    """Zero the potential on every edge lying in a wall of the box.

    The curl of the result has no flow through any wall, so adding it to a sealed
    velocity field keeps that field sealed and divergence free.
    """
    ax, ay, az = ax.clone(), ay.clone(), az.clone()
    for tensor, wall_axes in ((ax, (-3, -2)), (ay, (-3, -1)), (az, (-2, -1))):
        for axis in wall_axes:
            tensor.select(axis, 0).zero_()
            tensor.select(axis, -1).zero_()
    return ax, ay, az


def faces_as_cell_channels(u, v, w):
    """Stack each cell's six face values as channels: [..., 6, Z, Y, X]. Lossless."""
    return torch.stack((u[..., :, :, :-1], u[..., :, :, 1:],
                        v[..., :, :-1, :], v[..., :, 1:, :],
                        w[..., :-1, :, :], w[..., 1:, :, :]), dim=-4)


def remove_mean(cells):
    """Subtract each field's mean over its grid."""
    return cells - cells.mean(dim=(-3, -2, -1), keepdim=True)
