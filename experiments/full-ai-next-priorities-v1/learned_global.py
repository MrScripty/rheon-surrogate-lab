"""Learned cosine-mode mixing: a neural representation, with no pressure inverse."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=64)
def bases(counts, device):
    with torch.inference_mode(False):
        result=[]
        for n in reversed(counts):
            mode=torch.arange(min(8,n),device=device,dtype=torch.float32)
            cells=torch.arange(n,device=device,dtype=torch.float32)+.5
            axis=torch.cos(math.pi/n*mode[:,None]*cells[None,:])*math.sqrt(2/n)
            axis[0]/=math.sqrt(2)
            result.append(axis)
    return tuple(result)


def restrict(value, axes):
    z,y,x=axes
    value=torch.einsum('mx,bczyx->bczym',x,value)
    value=torch.einsum('ly,bczym->bczlm',y,value)
    return torch.einsum('kz,bczlm->bcklm',z,value)


def prolong(value, axes):
    z,y,x=axes
    value=torch.einsum('mx,bcklm->bcklx',x,value)
    value=torch.einsum('ly,bcklx->bckyx',y,value)
    return torch.einsum('kz,bckyx->bczyx',z,value)


class GlobalMix(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width=width
        self.mixing=nn.Parameter(torch.randn(8,8,8,width,width)/math.sqrt(width))
        self.geometry_head=nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,width*width))
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)

    def forward(self,value,features,counts):
        nx,ny,nz=counts
        axes=bases(counts,str(value.device))
        coefficients=restrict(value[:,:,:nz,:ny,:nx],axes)
        kz,ky,kx=coefficients.shape[2:]
        modes=torch.stack(torch.meshgrid(
            torch.arange(kz,device=value.device)/nz,
            torch.arange(ky,device=value.device)/ny,
            torch.arange(kx,device=value.device)/nx,indexing='ij'),dim=-1)
        geometry=features[:,3:7,0,0,0].reshape(value.shape[0],1,1,1,4).expand(-1,kz,ky,kx,-1)
        descriptor=torch.cat((modes.unsqueeze(0).expand(value.shape[0],-1,-1,-1,-1),geometry),dim=-1)
        weights=self.mixing[:kz,:ky,:kx].unsqueeze(0)+self.geometry_head(descriptor).reshape(
            value.shape[0],kz,ky,kx,self.width,self.width)
        mixed=torch.einsum('bcklm,bklmcd->bdklm',coefficients,weights)
        return F.pad(prolong(mixed,axes),(0,1,0,1,0,1),mode='replicate')
