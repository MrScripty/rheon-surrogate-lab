"""Convex tensor readout for neural displacements; no backtrace or solver."""
from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def coordinates(shape,device):
    with torch.inference_mode(False):
        z,y,x=torch.meshgrid(*[torch.arange(n,device=device,dtype=torch.float64)for n in shape],indexing='ij')
        return torch.stack((x,y,z),0).reshape(1,1,3,-1)


def convex_field_readout(fields,displacements):
    if fields.ndim!=5 or fields.shape[:2]!=(1,4)or displacements.shape!=(1,4,3,*fields.shape[2:]):
        raise ValueError('Four colocated carrier fields and their learned displacements required')
    if not bool(torch.isfinite(fields).all())or not bool(torch.isfinite(displacements).all()):
        raise FloatingPointError('Finite fields and learned displacements required')
    if float(displacements.abs().max())>1:raise ValueError('Registered subcell readout support exceeded')
    nz,ny,nx=fields.shape[2:];source=fields.double().flatten(2)
    position=coordinates(tuple(fields.shape[2:]),str(fields.device))+displacements.double().reshape(1,4,3,-1)
    maximum=position.new_tensor((nx-1,ny-1,nz-1)).reshape(1,1,3,1)
    position=torch.minimum(position.clamp_min(0),maximum)
    low=position.floor().long();high=torch.minimum(low+1,maximum.long());fraction=position-low
    result=torch.zeros_like(source);minimum=torch.full_like(source,float('inf'));upper=-minimum
    for z in range(2):
        for y in range(2):
            for x in range(2):
                bits=(x,y,z);indices=[high[:,:,d]if bits[d]else low[:,:,d]for d in range(3)]
                index=indices[0]+nx*(indices[1]+ny*indices[2]);value=source.gather(2,index)
                weight=torch.ones_like(result)
                for d in range(3):weight=weight*(fraction[:,:,d]if bits[d]else 1-fraction[:,:,d])
                result=result+weight*value;minimum=torch.minimum(minimum,value);upper=torch.maximum(upper,value)
    return torch.minimum(torch.maximum(result,minimum),upper).reshape_as(fields).float()
