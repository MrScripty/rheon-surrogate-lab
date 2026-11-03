"""Separable Neumann Galerkin corrections for uniform, sealed cubic cells.

The retained fine operator remains authoritative. No pressure labels enter a
correction. The DC residual is checked, retained, and never solved away.
"""
from functools import lru_cache
import math
import torch


@lru_cache(maxsize=64)
def geometry(shape, max_mode):
    axes = []
    eigenvalues = []
    for n in shape:
        count = n if max_mode is None else min(n, max_mode + 1)
        modes = torch.arange(count, dtype=torch.float64)
        cells = torch.arange(n, dtype=torch.float64)
        basis = torch.cos(math.pi / n * modes[:, None] * (cells[None] + .5))
        basis *= math.sqrt(2. / n)
        basis[0] /= math.sqrt(2.)
        axes.append(basis)
        eigenvalues.append(4 * torch.sin(math.pi * modes / (2 * n)).square())
    z, y, x = eigenvalues
    values = z[:, None, None] + y[None, :, None] + x[None, None, :]
    values[0, 0, 0] = 1.  # DC coefficient is explicitly zeroed below.
    return tuple(axes), values


def coefficients(value, axes):
    z, y, x = axes
    result = torch.einsum('kz,bzyx->bkyx', z, value[:, 0])
    result = torch.einsum('ly,bkyx->bklx', y, result)
    return torch.einsum('mx,bklx->bklm', x, result)


def reconstruct(value, axes):
    z, y, x = axes
    result = torch.einsum('mx,bklm->bklx', x, value)
    result = torch.einsum('ly,bklx->bkyx', y, result)
    return torch.einsum('kz,bkyx->bzyx', z, result)[:, None]


def correction(residual, h, rho, max_mode=3):
    if (residual.ndim != 5 or residual.shape[1] != 1
            or residual.dtype != torch.float64 or residual.device.type != 'cpu'
            or min(residual.shape[2:]) < 1 or not bool(torch.isfinite(residual).all())):
        raise ValueError('Finite CPU float64 N,1,Z,Y,X residual required')
    if max_mode is not None and (type(max_mode) is not int or max_mode < 0):
        raise ValueError('Nonnegative mode bound or explicit full solve required')
    if any(v.shape != (residual.shape[0],) or v.dtype != torch.float64
           or v.device.type != 'cpu' or not bool(torch.isfinite(v).all() and (v > 0).all())
           for v in (h, rho)):
        raise ValueError('Positive finite CPU float64 per-case spacing and mass density required')
    size = math.prod(residual.shape[2:])
    norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    constant = residual.flatten(1).mean(1).abs() * math.sqrt(size)
    if bool((constant > torch.maximum(norm * 1e-8, norm.new_full(norm.shape, 1e-12))).any()):
        raise ValueError('Incompatible sealed Neumann RHS; equation not modified')
    axes, eigenvalues = geometry(tuple(residual.shape[2:]), max_mode)
    q = residual * (rho / h).reshape(-1, 1, 1, 1, 1)
    modes = coefficients(q, axes) / eigenvalues
    modes[:, 0, 0, 0] = 0.
    result = reconstruct(modes, axes)
    return result - result[:, :, :1, :1, :1].clone()


def fine_action(pressure, h, rho):
    """Independent edge-flux stencil, used in numerical qualification only."""
    result = torch.zeros_like(pressure)
    for axis in (2, 3, 4):
        left = [slice(None)] * 5
        right = left.copy()
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        difference = pressure[tuple(left)] - pressure[tuple(right)]
        result[tuple(left)] += difference
        result[tuple(right)] -= difference
    return result * (h / rho).reshape(-1, 1, 1, 1, 1)
