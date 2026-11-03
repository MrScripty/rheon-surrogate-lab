"""Reuse the qualified Galerkin hierarchy with a direct fine edge stencil."""
from functools import lru_cache
import torch
import multilevel as original

def fine_action(value):
    out=torch.zeros_like(value)
    for axis in [2,3,4]:
        left=[slice(None)]*5;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        difference=value[tuple(left)]-value[tuple(right)]
        out[tuple(left)]+=difference;out[tuple(right)]-=difference
    return out

class Level:
    def __init__(self,base,finest=False):
        self.base=base;self.finest=finest;self.child=Level(base.child)if base.child is not None else None
    def action(self,value):return fine_action(value)if self.finest else self.base.action(value)
    def cycle(self,rhs):
        if self.child is None:return self.base.cycle(rhs)
        # The first Jacobi step starts at zero. Avoid allocating and applying
        # the unchanged operator to that identically zero field at every level.
        out=self.base.omega*rhs/self.base.diagonal
        for _ in range(2):out=out+self.base.omega*(rhs-self.action(out))/self.base.diagonal
        out=out+self.base.prolong(self.child.cycle(self.base.restrict(rhs-self.action(out))))
        for _ in range(3):out=out+self.base.omega*(rhs-self.action(out))/self.base.diagonal
        return out
@lru_cache(maxsize=16)
def geometry(shape):return Level(original.geometry(shape),finest=True)
def correction(residual,h,rho):
    original.validate(residual)
    for value in [h,rho]:
        if value.shape!=(1,)or value.dtype!=torch.float64 or value.device.type!='cpu'or not bool(torch.isfinite(value).all()and(value>0).all()):raise ValueError('Positive finite CPU float64 physical scalars required')
    out=geometry(tuple(residual.shape[2:])).cycle(residual*(rho/h).reshape(1,1,1,1,1))
    return out-out[:,:,:1,:1,:1].clone()

