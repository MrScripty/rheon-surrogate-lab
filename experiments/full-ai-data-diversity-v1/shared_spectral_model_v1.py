"""Shared learned spectral response with four native-data-trained parameters.

Only generic modal phase encoding and a fixed DCT basis are prescribed.
No physical stencil, eigenvalue coefficients or analytical inverse is supplied.
Forward evaluation is CUDA-only. CPU construction/load is artifact work only.
"""
import math
import torch
from torch import nn


class SharedSpectralResponse(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.coefficients=nn.Parameter(torch.empty(4,dtype=torch.float64,device=device))
        with torch.no_grad():self.coefficients.uniform_(-.1,.1)

    def forward(self, features):
        if features.device.type!='cuda' or self.coefficients.device.type!='cuda':
            raise ValueError('Learned forward evaluation requiresCUDA; CPU fallback forbidden')
        if features.dtype!=torch.float64 or features.shape[-1]!=4:
            raise ValueError('Four fixed double Fourier features required')
        result=features[...,0]*self.coefficients[0]
        for channel in range(1,4):result=result+features[...,channel]*self.coefficients[channel]
        return result

    def response_grid(self, shape):
        device=self.coefficients.device
        if device.type!='cuda':raise ValueError('CUDA learned response construction required')
        z,y,x=torch.meshgrid(*(torch.arange(size,dtype=torch.float64,device=device)/size for size in shape),indexing='ij')
        return self(torch.stack((torch.ones_like(x),torch.cos(math.pi*x),torch.cos(math.pi*y),torch.cos(math.pi*z)),dim=-1))

    def decode(self, q, axes, response=None):
        if q.device.type!='cuda' or q.dtype!=torch.float64 or q.ndim!=5 or q.shape[:2]!=(1,1):
            raise ValueError('Complete CUDA double pressure input required')
        z,y,x=axes
        if any(axis.device.type!='cuda' or axis.dtype!=torch.float64 for axis in axes):
            raise ValueError('Fixed complete CUDA double basis required')
        if response is None:response=self.response_grid(tuple(q.shape[2:]))
        divisor=response.clone();divisor[0,0,0]=1.
        value=torch.einsum('kz,bzyx->bkyx',z,q[:,0])
        value=torch.einsum('ly,bkyx->bklx',y,value)
        value=torch.einsum('mx,bklx->bklm',x,value)/divisor
        value[:,0,0,0]=0.
        value=torch.einsum('mx,bklm->bklx',x,value)
        value=torch.einsum('ly,bklx->bkyx',y,value)
        value=torch.einsum('kz,bkyx->bzyx',z,value)[:,None]
        return value-value[:,:,:1,:1,:1]


def objective(model, batch):
    if any(batch[key].device.type!='cuda' for key in ('features','p','q','normalizer','example_weight')):
        raise ValueError('Every supervised objective tensor must be on CUDA')
    error=(model(batch['features'])*batch['p']-batch['q'])/batch['normalizer']
    return (batch['example_weight']*error.square()).sum()
