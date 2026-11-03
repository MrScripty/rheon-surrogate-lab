"""Compact CUDA residual subnet; CPU construction/typed restore only, no CPU forward."""
import math
import torch
from torch import nn
from torch.nn import functional as F


CHANNELS=42
HIDDEN=8
PARAMETERS=10876


class _ClampedPad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.sizes=value.shape[-3:]
        return F.pad(value,(1,1,1,1,1,1),mode='replicate')

    @staticmethod
    def backward(ctx, gradient):
        # Fixed-order face folding avoids atomic scatter in replicate-pad's
        # standard CUDA backward. It also handles a singleton spatial axis.
        value=gradient
        for axis,size in reversed(tuple(zip((2,3,4),ctx.sizes))):
            folded=value.narrow(axis,1,size).clone()
            folded.select(axis,0).add_(value.select(axis,0))
            folded.select(axis,size-1).add_(value.select(axis,size+1))
            value=folded
        return value


def wall(value,component):
    axis=value.ndim-1-component
    index=torch.arange(value.shape[axis],device=value.device)
    keep=(index>0)&(index<value.shape[axis]-1)
    shape=[1]*value.ndim;shape[axis]=-1
    return torch.where(keep.reshape(shape),value,torch.zeros_like(value))


class InternalGroupResidual(nn.Module):
    def __init__(self,device='cuda',seed=1729,normalization=None):
        super().__init__()
        selected=torch.device(device)
        devices=[torch.cuda.current_device()] if selected.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.conv1=nn.Conv3d(CHANNELS,HIDDEN,3,device=selected,dtype=torch.float32)
            self.conv2=nn.Conv3d(HIDDEN,HIDDEN,3,device=selected,dtype=torch.float32)
            self.velocity=nn.ModuleList(nn.ConvTranspose3d(HIDDEN,1,kernel_size=k,device=selected,dtype=torch.float32)
                                        for k in ((1,1,2),(1,2,1),(2,1,1)))
            self.tracer=nn.Conv3d(HIDDEN,1,1,device=selected,dtype=torch.float32)
            for layer in (*self.velocity,self.tracer):
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        normalization=normalization or dict(input_rms=[1.]*CHANNELS,output_rms=[1.]*4)
        if len(normalization['input_rms'])!=CHANNELS or len(normalization['output_rms'])!=4 or any(not math.isfinite(v) or v<=0. for v in [*normalization['input_rms'],*normalization['output_rms']]):
            raise ValueError('Positive frozen input/output RMS scales required')
        self.register_buffer('input_rms',torch.tensor(normalization['input_rms'],device=selected,dtype=torch.float32).reshape(1,CHANNELS,1,1,1))
        self.register_buffer('output_rms',torch.tensor(normalization['output_rms'],device=selected,dtype=torch.float32))
        if sum(p.numel() for p in self.parameters())!=PARAMETERS:raise ValueError('Frozen compact architecture parameter count changed')

    def forward(self,features):
        if features.device.type!='cuda' or features.dtype!=torch.float32 or features.ndim!=5 or features.shape[:2]!=(1,CHANNELS):
            raise ValueError('One current CUDA F32 cell-lattice feature field required')
        hidden=F.silu(self.conv1(_ClampedPad.apply(features/self.input_rms)))
        hidden=F.silu(self.conv2(_ClampedPad.apply(hidden)))
        velocity=[wall(layer(hidden)[0,0]*self.output_rms[c],c) for c,layer in enumerate(self.velocity)]
        return [*velocity,self.tracer(hidden)[0,0]*self.output_rms[3]]

