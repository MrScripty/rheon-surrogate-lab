"""Alternate arithmetic representation of restored learned spectral weights.

No physical eigenvalues or analytical operator inverse are constructed here.
Every nonconstant divisor is the reciprocal of a restored learned gain.
"""
import torch
from torch import nn
from modal_basis import bases


class LearnedDivisorPressure(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.model = source.model
        self.identity = source.identity
        self.axes = {}
        self.divisors = {}

    def forward(self, q):
        if q.dtype != torch.float64 or q.device.type != 'cpu' or q.ndim != 5 or q.shape[:2] != (1, 1):
            raise ValueError('Complete CPU double learned pressure input required')
        if not bool(torch.isfinite(q).all()):
            raise ValueError('Finite learned pressure input required')
        shape = tuple(q.shape[2:])
        key = 'g'+'_'.join(map(str, shape))
        if shape not in self.axes:
            self.axes[shape] = bases(shape)
        if key not in self.divisors:
            gains = self.model.gains(key)
            self.divisors[key] = torch.where(gains != 0, gains.reciprocal(), torch.ones_like(gains))
        z, y, x = self.axes[shape]
        coefficients = torch.einsum('kz,bzyx->bkyx', z, q[:, 0])
        coefficients = torch.einsum('ly,bkyx->bklx', y, coefficients)
        coefficients = torch.einsum('mx,bklx->bklm', x, coefficients)
        coefficients = coefficients/self.divisors[key]
        coefficients[:, 0, 0, 0] = 0.
        result = torch.einsum('mx,bklm->bklx', x, coefficients)
        result = torch.einsum('ly,bklx->bkyx', y, result)
        result = torch.einsum('kz,bkyx->bzyx', z, result)[:, None]
        return result-result[:, :, :1, :1, :1]

    def close(self):
        self.axes = {}
        self.divisors = {}
