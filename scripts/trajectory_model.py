"""Learned full-spectrum pressure response for one uniform sealed 3D box.

The orthonormal cosine basis encodes geometry, not inverse eigenvalues. Every
nonconstant modal response is fitted from Rheon pressure labels. No analytic
inverse, PCG, teacher pressure, or optimizer is used at inference.
"""
from __future__ import annotations

import torch
from torch import nn


class ModalPressure(nn.Module):
    def __init__(self, n: int, device: str = "cpu"):
        super().__init__()
        self.n = n
        modes = torch.arange(n, dtype=torch.float64, device=device)[:, None]
        cells = torch.arange(n, dtype=torch.float64, device=device)[None, :]
        basis = torch.cos(torch.pi * modes * (cells + 0.5) / n) * (2 / n) ** 0.5
        basis[0] /= 2**0.5
        self.register_buffer("basis", basis)
        # The constant mode is the nullspace, and has no learnable coefficient.
        self.gains = nn.Parameter(torch.ones(n**3 - 1, dtype=torch.float64, device=device))

    def transform(self, field: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matrix = self.basis if inverse else self.basis.T
        result = field.reshape(*field.shape[:-1], self.n, self.n, self.n)
        for axis in (-1, -2, -3):
            result = (result.movedim(axis, -1) @ matrix).movedim(-1, axis)
        return result.flatten(-3)

    def forward(self, rhs: torch.Tensor) -> torch.Tensor:
        coefficients = self.transform(rhs)
        weights = torch.cat((self.gains.new_zeros(1), self.gains))
        pressure = self.transform(coefficients * weights, inverse=True)
        return pressure - pressure[..., :1]

    @torch.no_grad()
    def fit(self, rhs: torch.Tensor, pressure: torch.Tensor) -> dict:
        """Closed-form supervised least-squares training of independent gains.

        All statistics are fitted only on the supplied training trajectories.
        float64 protects weakly excited modes and strict divergence acceptance.
        """
        x = self.transform(rhs)[:, 1:]
        y = self.transform(pressure)[:, 1:]
        energy = (x * x).sum(0)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or energy.min() <= 0:
            raise ValueError("nonfinite labels or unexcited pressure mode")
        gains = (x * y).sum(0) / energy
        if not torch.isfinite(gains).all() or gains.min() <= 0:
            raise ValueError("invalid fitted response")
        self.gains.copy_(gains)
        relative_error = ((self(rhs) - pressure).norm(dim=1) / pressure.norm(dim=1)).mean()
        return {"method": "supervised_modal_least_squares", "optimizer_steps": 0,
                "fitted_parameters": self.gains.numel(),
                "minimum_training_modal_energy": energy.min().item(),
                "training_mean_relative_pressure_l2": relative_error.item()}
