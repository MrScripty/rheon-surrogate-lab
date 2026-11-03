"""Higher precision construction of the same frozen linear learned CNN map."""
import copy
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from materialized_fine import require_linear_spatial_network,MaterializedFine


class SpatialDouble(nn.Module):
    def __init__(self,source):
        super().__init__()
        require_linear_spatial_network(source)
        for name in ('encoders','bottleneck','global_pool','decoders','head'):
            setattr(self,name,copy.deepcopy(getattr(source,name)).double())

    def forward(self,values):
        skips=[]
        for encoder in self.encoders:
            values=encoder(values);skips.append(values)
            kernel=tuple(min(2,n)for n in values.shape[2:])
            values=F.avg_pool3d(values,kernel,stride=kernel,ceil_mode=True,count_include_pad=False)
        values=self.bottleneck(values)
        values=values+self.global_pool(values).expand_as(values)
        for decoder,skip in zip(self.decoders,reversed(skips)):
            values=F.interpolate(values,size=skip.shape[2:],mode='nearest')
            values=decoder(torch.cat((values,skip),dim=1))
        return self.head(values)


def materialize(network,counts):
    if torch.is_grad_enabled():raise ValueError('Inference-only learned materialization')
    if len(counts)!=3 or min(counts)<1 or max(counts)>16:
        raise ValueError('Axes1..16 required')
    tick=time.perf_counter();cells=math.prod(counts)
    spatial=SpatialDouble(network).eval()
    matrix=torch.empty((cells,cells),dtype=torch.float64)
    for start in range(0,cells,4):
        batch=min(4,cells-start)
        values=torch.zeros((batch,1,*counts),dtype=torch.float64)
        values.flatten(1)[torch.arange(batch),start+torch.arange(batch)]=1.
        matrix[:,start:start+batch]=spatial(values).flatten(1).T
    if not bool(torch.isfinite(matrix).all()):raise ValueError('Nonfinite learned map')
    return matrix,dict(counts_zyx=list(counts),cells=cells,matrix_bytes=matrix.numel()*8,
        seconds=time.perf_counter()-tick,construction_forward_calls=(cells+3)//4,
        method='Float64 unit responses of frozen float32 CNN coefficients; no physical inverse')


class PreciseFine(MaterializedFine):
    def forward(self,values):
        return F.linear(values.flatten(1).to(self.fine_matrix.dtype),self.fine_matrix).to(values.dtype).reshape_as(values)
