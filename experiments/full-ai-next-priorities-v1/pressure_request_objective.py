"""Offline full-modal teacher loss; this module is excluded from inference."""
import math
from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def teacher_geometry(counts,device):
    with torch.inference_mode(False):
        axes=[];eigenvalues=[]
        for n in counts:
            k=torch.arange(n,device=device,dtype=torch.float64)
            x=torch.arange(n,device=device,dtype=torch.float64)
            axis=torch.cos(math.pi*k[:,None]*(x[None,:]+.5)/n)*math.sqrt(2/n)
            axis[0]*=1/math.sqrt(2)
            axes.append(axis.float())
            eigenvalues.append(2*(1-torch.cos(math.pi*k/n)))
        values=eigenvalues[0][:,None,None]+eigenvalues[1][None,:,None]+eigenvalues[2][None,None,:]
        values[0,0,0]=1.
        gains=values.reciprocal();gains[0,0,0]=0.
    return tuple(axes),gains.float()


def teacher_pressure(q):
    """Differentiable training diagnostic only; never a candidate forward path."""
    axes,gains=teacher_geometry(tuple(q.shape[-3:]),str(q.device))
    z,y,x=axes
    coefficients=torch.einsum('kz,bhczyx->bhckyx',z,q)
    coefficients=torch.einsum('ly,bhckyx->bhcklx',y,coefficients)
    coefficients=torch.einsum('mx,bhcklx->bhcklm',x,coefficients)*gains
    pressure=torch.einsum('mx,bhcklm->bhcklx',x,coefficients)
    pressure=torch.einsum('ly,bhcklx->bhckyx',y,pressure)
    return torch.einsum('kz,bhckyx->bhczyx',z,pressure)


def loss(predicted,target):
    if predicted.shape!=target.shape or predicted.ndim!=6:
        raise ValueError('Matching batch,horizon,1,Z,Y,X request fields required')
    def relative_square(difference,reference):
        return difference.flatten(2).square().sum(2)/reference.flatten(2).square().sum(2).clamp_min(reference[0,0].numel()*1e-16)
    difference=predicted-target
    equation=relative_square(difference,target)
    pressure=relative_square(teacher_pressure(difference),teacher_pressure(target))
    combined=(equation+pressure)/2
    return combined.mean()+.1*combined.max(dim=1).values.mean()
