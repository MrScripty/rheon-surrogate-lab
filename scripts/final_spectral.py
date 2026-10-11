"""Analytic fixed-box spectral baseline; never fits or reads pressure targets."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from global_candidate import cosine_basis


def eigenvalues(counts, spacing, density):
    """Integrated finite-volume Neumann operator, X-fastest nonconstant modes."""
    nx, ny, nz = counts
    kz, ky, kx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    volume = float(np.prod(spacing))
    lam = sum(4 * volume / (density * h * h) * np.sin(np.pi * k / (2*n))**2
              for k, n, h in zip((kx, ky, kz), counts, spacing))
    values = lam.ravel()[1:]
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("positive nonconstant eigenvalues required")
    return values


def solve(rhs, basis, eigenvalues):
    raw = ((rhs @ basis) / eigenvalues) @ basis.T
    return raw - raw[:, :1]
