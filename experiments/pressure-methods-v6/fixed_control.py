"""Untrained fixed q/8 control with exactly the learned path's f32 conversion."""
import torch
class FixedJacobi(torch.nn.Module):
    def forward(self,values):
        if values.ndim!=5 or values.shape[1]!=1 or values.dtype!=torch.float32 or not bool(torch.isfinite(values).all()):raise ValueError('Finite single-channel f32 control input required')
        return values*.125
def work_kind(work,fixed):
    if not fixed:return work
    return dict(work,post_jacobi_calls=work['network_calls'],post_jacobi_seconds=work['network_seconds'],network_calls=0,network_seconds=0.,fixed_jacobi=True)
