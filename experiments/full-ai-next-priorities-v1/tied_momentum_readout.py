"""Neural momentum coefficients shared with a pretrained neural flux layer.

This transforms learned convolution weights only. It loads no analytical
gradient stencil and invokes no physical solver or projection routine.
"""
import torch
from torch.nn import functional as F
from temporal_features_v2 import geometry


def index_readout(pressure,learned_flux_weight):
    if pressure.ndim!=5 or pressure.shape[1]!=1 or learned_flux_weight.shape!=(1,3,3,3,3):
        raise ValueError('Cell pressure and qualified learned3-to-1flux weights required')
    counts=tuple(reversed(pressure.shape[2:]))
    values=F.pad(pressure,(0,1,0,1,0,1),mode='replicate')
    kernel=-learned_flux_weight.to(device=pressure.device,dtype=pressure.dtype).flip(dims=(2,3,4)).transpose(0,1)
    result=F.conv3d(F.pad(values,(1,1,1,1,1,1),mode='replicate'),kernel)
    _,mask=geometry(counts,str(pressure.device))
    return result*mask[1:4]
