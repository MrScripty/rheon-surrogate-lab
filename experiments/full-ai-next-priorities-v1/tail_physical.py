"""Neural eight-mode pressure tail. No eigenvalue evaluation or inverse at inference."""
import math
from functools import lru_cache
import torch
import coarse


@lru_cache(maxsize=64)
def axes_for(shape,device):
    with torch.inference_mode(False):
        result=[]
        for n in shape:
            mode=torch.arange(min(8,n),dtype=torch.float64,device=device)
            cells=torch.arange(n,dtype=torch.float64,device=device)
            axis=torch.cos(math.pi/n*mode[:,None]*(cells[None,:]+.5))*math.sqrt(2/n)
            axis[0]/=math.sqrt(2)
            result.append(axis)
    return tuple(result)


def append(network,sample,pressure,owner):
    counts=tuple(pressure.shape[2:])
    axes=axes_for(counts,str(pressure.device))
    features=torch.tensor([math.log(n/16)for n in counts],dtype=torch.float32,device=pressure.device)
    positive=max(counts)**2/2*torch.sigmoid((network.tail_logits+network.tail_shape_head(features)).to(torch.float64))
    gains=torch.cat((positive.new_zeros(1),positive)).reshape(8,8,8)
    gains=gains[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
    residual=sample['native_input_residual']-owner.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
    coefficients=coarse.restrict(residual*(sample['rho']/sample['cell_size_m']).reshape(-1,1,1,1,1),axes)
    proposal=coarse.prolong(coefficients*gains,axes)
    proposal=proposal-proposal[:,:,:1,:1,:1].clone()
    remaining=residual-owner.full_operator(proposal,sample['cell_size_m'],sample['rho'])
    accept=(remaining.flatten(1).square().sum(1)<=residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
    return pressure+torch.where(accept,proposal,torch.zeros_like(proposal))
