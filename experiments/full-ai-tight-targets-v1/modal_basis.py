"""Full real cell-centred orthonormal DCT-II; no physical inverse coefficients."""
import math
import torch


def bases(shape, device='cpu'):
    result = []
    for size in shape:
        modes = torch.arange(size, dtype=torch.float64, device=device)[:, None]
        cells = torch.arange(size, dtype=torch.float64, device=device)[None, :] + .5
        axis = torch.cos(math.pi / size * modes * cells) * math.sqrt(2. / size)
        axis[0] /= math.sqrt(2.)
        result.append(axis)
    return tuple(result)


def transform(values, axes, inverse=False):
    for dimension, axis in zip(range(values.ndim - 3, values.ndim), axes):
        values = (values.movedim(dimension, -1) @ (axis if inverse else axis.T)).movedim(-1, dimension)
    return values
