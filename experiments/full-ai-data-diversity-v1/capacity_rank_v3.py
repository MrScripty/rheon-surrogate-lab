"""Computed rank-aware capacity diagnostics; never solve or fit a head.

The legacy last QR diagonal is an optimistic comparison. A numerical-rank
projection is distinct from an unbounded-coefficient impossibility claim. A
finite registered coefficient norm gives an additional computed lower bound.
"""
import torch
from capacity_qr_v2 import compress as original_compress


@torch.no_grad()
def compress(matrix, target, denominator, reg):
    factor, stats = original_compress(matrix, target, denominator, reg)
    u, singular, _ = torch.linalg.svd(factor[:, :-1], full_matrices=False, driver='gesvd')
    threshold = singular.max()*max(matrix.shape)*torch.finfo(torch.float64).eps
    rank = int((singular > threshold).sum())
    rhs = factor[:, -1]
    residual = rhs-u[:, :rank]@(u[:, :rank].T@rhs)
    distance = residual.norm()
    discarded = singular[rank] if rank < singular.numel() else singular.new_zeros(())
    limit = reg['head_l2_norm_limit']
    bound = (distance-discarded*limit).clamp_min(0.)
    width = matrix.shape[1]
    controls = torch.stack((torch.zeros(width, device='cuda', dtype=torch.float64),
        torch.linspace(-.25, .25, width, device='cuda', dtype=torch.float64),
        torch.full((width,), .1, device='cuda', dtype=torch.float64)))
    if bool((controls.norm(dim=1) > limit).any()):
        raise ValueError('Registered capacity controls exceed head norm cap')
    measured = (factor[:, :-1]@controls.T-rhs[:, None]).norm(dim=0)
    if not bool((measured+reg['bounded_projection_control_tolerance'] >= bound).all()):
        raise ValueError('Computed bounded-head lower bound exceeds fixed-control losses')
    stats.update(effective_rank=rank, numerical_rank_threshold=float(threshold),
        effective_rank_projection_residual=float(distance),
        largest_discarded_singular_value=float(discarded), head_l2_norm_limit=limit,
        computed_bounded_head_residual_lower_bound=float(bound),
        bounded_head_control_vectors=3,
        rank_interpretation='Computed finite-precision diagnostics. Discarding small singular values does not prove an unbounded-coefficient model impossible. The additional lower bound is conditional on the stated joint head L2 norm cap; no head coefficients are solved.')
    return factor, stats
