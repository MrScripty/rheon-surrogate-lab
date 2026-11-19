"""Learned diagonal pressure component; no physical inverse is constructed.

Finite geometry tables are an accuracy/capacity experiment, not evidence of
unseen-shape generalization or a complete temporal simulation generator.
"""
import torch
from torch import nn


class LearnedModalPressure(nn.Module):
    def __init__(self, geometries):
        super().__init__()
        self.log_gains = nn.ParameterDict({row['key']: nn.Parameter(torch.zeros(tuple(row['counts_zyx']), dtype=torch.float64))
                                           for row in geometries})

    def gains(self, key):
        values = self.log_gains[key].exp()
        # Gauge freedom is fixed. No physical eigenvalues or inverse scales
        # enter the learned parameters; their initialization is uniformly one.
        mask = torch.ones_like(values)
        mask[0, 0, 0] = 0.
        return values * mask

    def forward(self, q_coefficients, key):
        if q_coefficients.dtype != torch.float64 or q_coefficients.shape[-3:] != self.log_gains[key].shape:
            raise ValueError('Registered full double spectral geometry required')
        return q_coefficients * self.gains(key)
