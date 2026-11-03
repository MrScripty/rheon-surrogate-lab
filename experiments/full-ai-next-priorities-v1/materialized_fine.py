"""Materialize a frozen linear learned spatial network; no physical inverse."""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F


def require_linear_spatial_network(network):
    roots=('encoders','bottleneck','global_pool','decoders','head')
    allowed=(nn.ModuleList,nn.Sequential,nn.Conv3d,nn.Identity,nn.AdaptiveAvgPool3d)
    checked=[]
    for name in roots:
        root=getattr(network,name)
        for path,module in root.named_modules():
            # ConvBlock is only a container around its qualified layers.
            if not isinstance(module,allowed) and not (
                hasattr(module,'layers') and isinstance(module.layers,nn.Sequential)
            ):
                raise ValueError('Unqualified spatial module: '+name+'.'+path+':'+type(module).__name__)
            if isinstance(module,nn.Conv3d) and module.bias is not None:
                raise ValueError('Spatial convolution must have no affine bias')
            checked.append(dict(name=name+'.'+path,type=type(module).__name__))
    if any(parameter.device.type!='cpu' or parameter.dtype!=torch.float32 for parameter in network.parameters()):
        raise ValueError('Frozen CPU float32 weights required')
    return checked


def materialize(network,counts):
    if torch.is_grad_enabled():
        raise ValueError('Learned operator materialization is inference only')
    require_linear_spatial_network(network)
    if len(counts)!=3 or min(counts)<1 or max(counts)>16:
        raise ValueError('Dense learned execution is limited to axes1..16')
    tick=time.perf_counter();cells=math.prod(counts)
    matrix=torch.empty((cells,cells),dtype=torch.float32)
    zero=torch.zeros((1,1,*counts),dtype=torch.float32)
    if torch.count_nonzero(network(zero)):
        raise ValueError('Frozen spatial network has nonzero affine offset')
    # Each column is the existing learned CNN's response to an input unit vector.
    # There are no RHS, pressure labels, differential operators or solves here.
    for start in range(0,cells,4):
        batch=min(4,cells-start)
        values=torch.zeros((batch,1,*counts),dtype=torch.float32)
        values.flatten(1)[torch.arange(batch),start+torch.arange(batch)]=1.
        result=network(values)
        matrix[:,start:start+batch]=result.flatten(1).T
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError('Nonfinite materialized learned coefficients')
    return matrix.contiguous(),dict(counts_zyx=list(counts),cells=cells,
        matrix_bytes=matrix.numel()*matrix.element_size(),seconds=time.perf_counter()-tick,
        method='Frozen bias-free linear CNN unit responses; no numerical pressure inverse',
        construction_forward_calls=1+(cells+3)//4)


class MaterializedFine(nn.Module):
    def __init__(self,source,matrix):
        super().__init__()
        self.source=source
        self.register_buffer('fine_matrix',matrix)

    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('source'),name)

    def forward(self,values):
        return F.linear(values.flatten(1),self.fine_matrix).reshape_as(values)
