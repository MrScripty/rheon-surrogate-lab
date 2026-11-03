"""Bounded low spatial-mode pressure feedback; no inference dependency."""
from functools import lru_cache
import math
import torch


@lru_cache(maxsize=64)
def basis(length, max_mode, device, dtype):
    cells = torch.arange(length, device=device, dtype=dtype)
    modes = torch.arange(min(length, max_mode + 1), device=device, dtype=dtype)
    result = torch.cos(math.pi / length * modes[:, None] * (cells[None, :] + .5))
    result = result * math.sqrt(2. / length)
    result[0] /= math.sqrt(2.)
    return result


def coefficients(value, max_mode):
    """Separable orthonormal DCT-II, retaining only a bounded low-mode block."""
    z, y, x = value.shape[-3:]
    axes = [basis(n, max_mode, value.device, value.dtype) for n in (z, y, x)]
    result = torch.einsum('kz,bzyx->bkyx', axes[0], value[:, 0])
    result = torch.einsum('ly,bkyx->bklx', axes[1], result)
    result = torch.einsum('mx,bklx->bklm', axes[2], result)
    # The constant pressure gauge is not a learnable pressure-solve error.
    return result.flatten(1)[:, 1:]


def loss(prediction, sample, specification):
    target = sample['reference'] - sample['base']
    target = target - target.mean(dim=(2, 3, 4), keepdim=True)
    error = coefficients(prediction - sample['reference'], specification['max_mode'])
    signal = coefficients(target, specification['max_mode']).square().sum(1)
    floor = specification['signal_floor_fraction'] ** 2 * target.flatten(1).square().sum(1)
    return error.square().sum(1) / torch.maximum(signal, floor).clamp_min(1e-30)
