"""Same streamed QR diagnostics with one direct SVD, avoiding Jacobi fallback.

No head coefficients are solved. The conditional bound still assumes the
registered finite joint head norm, and is a floating point diagnostic.
"""
import torch


@torch.no_grad()
def rank_statistics(factor, matrix_shape, reg):
    u, singular, _ = torch.linalg.svd(factor[:, :-1], full_matrices=False, driver='gesvd')
    threshold = singular.max()*max(matrix_shape)*torch.finfo(torch.float64).eps
    rank = int((singular > threshold).sum()); rhs = factor[:, -1]
    distance = (rhs-u[:, :rank]@(u[:, :rank].T@rhs)).norm()
    discarded = singular[rank] if rank < singular.numel() else singular.new_zeros(())
    limit = reg['head_l2_norm_limit']; bound = (distance-discarded*limit).clamp_min(0.)
    width = matrix_shape[1]
    controls = torch.stack((torch.zeros(width, device='cuda', dtype=torch.float64),
        torch.linspace(-.25,.25,width,device='cuda',dtype=torch.float64),
        torch.full((width,),.1,device='cuda',dtype=torch.float64)))
    if bool((controls.norm(dim=1)>limit).any()): raise ValueError('Capacity controls exceed head norm cap')
    measured = (factor[:, :-1]@controls.T-rhs[:, None]).norm(dim=0)
    if not bool((measured+reg['bounded_projection_control_tolerance']>=bound).all()):
        raise ValueError('Computed conditional bound exceeds fixed-control losses')
    return dict(computed_feature_rank=rank, effective_rank=rank,
        numerical_rank_threshold=float(threshold),effective_rank_projection_residual=float(distance),
        largest_discarded_singular_value=float(discarded),head_l2_norm_limit=limit,
        computed_bounded_head_residual_lower_bound=float(bound),bounded_head_control_vectors=3,
        SVD_driver='gesvd',SVD_calls=1,
        rank_interpretation='Computed finite-precision diagnostics. The lower bound assumes the registered finite joint head L2 cap. No unbounded-coefficient impossibility claim and no coefficients solved.')


@torch.no_grad()
def compress(matrix,target,denominator,reg):
    original_rows,width=matrix.shape; padding=max(0,width+1-original_rows)
    if padding:
        matrix=torch.cat((matrix,matrix.new_zeros((padding,width))),dim=0)
        target=torch.cat((target,target.new_zeros(padding)))
    n=matrix.shape[0]
    controls=torch.stack((torch.zeros(width,device='cuda',dtype=torch.float64),
        torch.linspace(-.25,.25,width,device='cuda',dtype=torch.float64),
        torch.full((width,),.1,device='cuda',dtype=torch.float64)))
    direct=torch.zeros(3,device='cuda',dtype=torch.float64)
    factor=torch.empty((0,width+1),device='cuda',dtype=torch.float64)
    for start in range(0,n,reg['maximum_design_rows_per_chunk']):
        end=min(n,start+reg['maximum_design_rows_per_chunk'])
        block=torch.cat((matrix[start:end],target[start:end,None]),dim=1)/denominator
        residuals=block[:,:-1]@controls.T-block[:,-1:];direct+=residuals.square().sum(dim=0)
        _,factor=torch.linalg.qr(torch.cat((factor,block),dim=0),mode='r')
    vectors=torch.cat((controls,-torch.ones((3,1),device='cuda',dtype=torch.float64)),dim=1)
    errors=(direct-(factor@vectors.T).square().sum(dim=0)).abs()
    allowed=reg['compression_relative_tolerance']*direct.abs()+reg['compression_absolute_tolerance']
    if factor.shape!=(width+1,width+1) or not bool(factor.isfinite().all()) or not bool((errors<=allowed).all()):
        raise ValueError('Full-grid residual norms changed during bounded QR')
    stats=dict(computed_orthogonal_residual=float(factor[-1,-1].abs()),
        zero_head_relative_error=float(direct[0].sqrt()),feature_columns=width,full_grid_rows=original_rows,
        control_vectors=3,maximum_absolute_compression_error=float(errors.max()),zero_padded_rows=padding,
        underdetermined_optimistic_diagnostic=bool(padding),
        underdetermined_scope='Zero padding preserves residual norms, but does not establish accuracy or a fitted model.' if padding else None)
    stats.update(rank_statistics(factor,(original_rows,width),reg))
    return factor,stats
