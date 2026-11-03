"""Learned spatial coarse mixing; no pressure operator or inverse solver.

Restrict cell averages to four bins per axis, learn one dense 64-to-64 map,
then interpolate coarse cell centres to the input grid. The dimensionless
length-squared factor is relative to a 16-cell longest side. All weights start
at zero without consuming the encoder/decoder's random initialization stream.
"""
import torch
from torch import nn


def restriction(size, device, dtype):
    """Exact overlaps of unit fine cells and four equal coarse intervals."""
    fine = torch.arange(size, device=device, dtype=dtype)
    edges = torch.arange(5, device=device, dtype=dtype) * (size / 4)
    overlap = (torch.minimum(edges[1:, None], fine[None, :] + 1)
               - torch.maximum(edges[:-1, None], fine[None, :])).clamp_min(0)
    return overlap / (size / 4)


def prolongation(size, device, dtype):
    """Linear coarse-centre interpolation, clamped at sealed domain edges."""
    location = ((torch.arange(size, device=device, dtype=dtype) + .5) * 4 / size - .5).clamp(0, 3)
    centres = torch.arange(4, device=device, dtype=dtype)
    return (1 - (location[:, None] - centres[None, :]).abs()).clamp_min(0)


class SpatialCoarseCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(64, 64, dtype=torch.float32))

    def forward(self, values):
        shape = values.shape[2:]
        coarse = values
        for axis, size in enumerate(shape, 2):
            coarse = torch.matmul(coarse.movedim(axis, -1),
                                  restriction(size, values.device, values.dtype).T).movedim(-1, axis)
        mixed = torch.nn.functional.linear(coarse.flatten(2), self.weight).reshape(-1, 1, 4, 4, 4)
        for axis, size in enumerate(shape, 2):
            mixed = torch.matmul(mixed.movedim(axis, -1),
                                 prolongation(size, values.device, values.dtype).T).movedim(-1, axis)
        mixed = mixed - mixed.mean(dim=(2, 3, 4), keepdim=True)
        return mixed * (max(shape) / 16) ** 2
