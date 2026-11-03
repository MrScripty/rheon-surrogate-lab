"""Causal CUDA feature columns for a prospective learned spatial readout.

No native operator, teacher field, coefficient solve or optimizer belongs here.
Column order is fixed: modulation, temporal difference, dz, dy, dx.
"""
import itertools
import torch
from recurrent_sampler_native import dimensions


OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


@torch.no_grad()
def columns(features, name, scene, model, conditioned):
    if len(features) != 4 or any(v.device.type != 'cuda' for f in features for v in f.values()):
        raise ValueError('Three frozen learned CUDA stages and current complete state required')
    counts = tuple(scene['counts'])
    kind = {'vx': 0, 'vy': 1, 'vz': 2, 'pressure': 3, 'tracer': 3}[name]
    shape = dimensions(counts)[kind]
    anchor = features[-1][name].double().reshape(shape)
    base = torch.empty((81, *shape), dtype=torch.float64, device='cuda')
    indices = [torch.arange(n, device='cuda') for n in shape]
    col = 0
    for state in features[:-1]:
        delta = state[name].double().reshape(shape) - anchor
        for dz, dy, dx in OFFSETS:
            z = (indices[0] + dz).clamp(0, shape[0] - 1)
            y = (indices[1] + dy).clamp(0, shape[1] - 1)
            x = (indices[2] + dx).clamp(0, shape[2] - 1)
            base[col] = delta[z[:, None, None], y[None, :, None], x[None, None, :]]
            col += 1
    if not conditioned:
        return base
    # Existing restored point-head projections place current velocity on each
    # target lattice. No future/native values or assigned physical stencil.
    current = [features[0][n].reshape(s) for n, s in zip(('vx', 'vy', 'vz'), dimensions(counts))]
    result = torch.empty((324, *shape), dtype=torch.float64, device='cuda')
    result[:81] = base
    for axis in range(3):
        velocity = model.linear(current, shape, 'point', kind * 3 + axis)
        modulation = velocity * (scene['cells_per_metre'] / 60.)
        result[(axis + 1) * 81:(axis + 2) * 81] = base * modulation
    return result
