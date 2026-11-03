"""Norm-preserving QR support for grids with fewer rows than feature columns."""
import torch
from screen_macrostep_spatial_features_v1 import compress as original_compress


@torch.no_grad()
def compress(matrix, target, denominator, reg):
    original_rows, width = matrix.shape
    padding = max(0, width+1-original_rows)
    if padding:
        matrix = torch.cat((matrix, matrix.new_zeros((padding, width))), dim=0)
        target = torch.cat((target, target.new_zeros(padding)))
    factor, stats = original_compress(matrix, target, denominator, reg)
    stats.update(full_grid_rows=original_rows, zero_padded_rows=padding,
                 underdetermined_optimistic_diagnostic=bool(padding),
                 underdetermined_scope='Zero rows preserve every full-grid residual norm; a zero projection diagnostic with more columns than physical rows does not establish fit, rank sufficiency or deployed accuracy.' if padding else None)
    return factor, stats
