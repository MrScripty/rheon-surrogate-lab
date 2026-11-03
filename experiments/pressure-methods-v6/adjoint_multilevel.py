"""Exact linear V-cycle transpose, including the final cell-zero gauge transpose.

For fixed h/rho, the ungauged symmetric V-cycle is C and the returned cycle is
G C (rho/h) r, where G removes cell zero. Its adjoint is (rho/h) C G^T g.
No model or numerical-stage gradient is discarded. First-order training only.
"""
import torch
from torch.autograd.function import once_differentiable
try:
    import training_fast_multilevel as fast
except ModuleNotFoundError as error:
    if error.name!='training_fast_multilevel':raise
    import fast_multilevel as fast
class FrozenCycle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,residual,h,rho):
        if h.requires_grad or rho.requires_grad:raise ValueError('This adjoint requires fixed physical scalars')
        ctx.ratio=float((rho/h)[0]);ctx.shape=tuple(residual.shape[2:])
        return fast.correction(residual,h,rho)
    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        adjoint=gradient.clone();adjoint[:,:,0,0,0]-=gradient.sum((2,3,4))
        # Do not gauge the result: the transpose of the output gauge has already
        # been applied to the incoming gradient, and C is the ungauged cycle.
        result=fast.geometry(ctx.shape).cycle(adjoint*ctx.ratio)
        return result,None,None
def correction(residual,h,rho):return FrozenCycle.apply(residual,h,rho)
