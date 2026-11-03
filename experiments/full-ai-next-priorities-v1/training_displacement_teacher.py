"""Numerical reference labels for training only; excluded from model inference."""
from functools import lru_cache
import torch
from torch.nn import functional as F


@lru_cache(maxsize=16)
def point_coordinates(counts,device):
    nx,ny,nz=counts;result=[]
    with torch.inference_mode(False):
        for kind in range(4):
            dimensions=[n+(1 if kind==d else 0)for d,n in enumerate(counts)]if kind<3 else list(counts)
            axes=[torch.arange(n,device=device,dtype=torch.float64)+(0. if kind==d else .5)for d,n in enumerate(dimensions)]
            z,y,x=torch.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
            result.append(torch.stack((x,y,z),0).reshape(3,-1))
    return tuple(result)


def scalar_sample(value,position,offset):
    nz,ny,nx=value.shape;dims=(nx,ny,nz);lo=[];hi=[];fraction=[]
    for d,n in enumerate(dims):
        q=(position[d]-offset[d]).clamp(0,n-1);a=q.floor().long()
        lo.append(a);hi.append((a+1).clamp_max(n-1));fraction.append(q-a)
    flat=value.reshape(-1);result=torch.zeros_like(position[0]);minimum=torch.full_like(result,float('inf'));maximum=-minimum
    for z in range(2):
        for y in range(2):
            for x in range(2):
                bits=(x,y,z);index=[hi[d]if bits[d]else lo[d]for d in range(3)]
                weight=torch.ones_like(result)
                for d in range(3):weight=weight*(fraction[d]if bits[d]else 1-fraction[d])
                v=flat[index[0]+nx*(index[1]+ny*index[2])]
                result=result+weight*v;minimum=torch.minimum(minimum,v);maximum=torch.maximum(maximum,v)
    return torch.minimum(torch.maximum(result,minimum),maximum)


@torch.no_grad()
def displacement_labels(physical_velocity,counts,dt_over_h):
    if physical_velocity.ndim!=5 or physical_velocity.shape[:2]!=(1,3):raise ValueError('One physical staggered velocity carrier required')
    nx,ny,nz=counts;v=physical_velocity.double()
    fields=(v[0,0,:nz,:ny,:nx+1],v[0,1,:nz,:ny+1,:nx],v[0,2,:nz+1,:ny,:nx])
    groups=point_coordinates(tuple(counts),str(v.device));position=torch.cat(groups,1)
    def sample(p):
        return torch.stack([scalar_sample(fields[d],p,[0. if a==d else .5 for a in range(3)])for d in range(3)])
    upper=v.new_tensor(counts).reshape(3,1);step=torch.as_tensor(dt_over_h,device=v.device,dtype=torch.float64).reshape(1,1)
    first=sample(position)
    middle=torch.minimum((position-.5*step*first).clamp_min(0),upper)
    departure=torch.minimum((position-step*sample(middle)).clamp_min(0),upper)
    shift=departure-position
    packed_shift=[];packed_first=[];start=0
    for kind,g in enumerate(groups):
        shape=tuple(reversed([n+(1 if kind==d else 0)for d,n in enumerate(counts)]if kind<3 else list(counts)))
        size=g.shape[1];pad=(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0])
        packed_shift.append(F.pad(shift[:,start:start+size].reshape(1,3,*shape),pad))
        packed_first.append(F.pad(first[:,start:start+size].reshape(1,3,*shape),pad));start+=size
    return torch.stack(packed_shift,1),torch.stack(packed_first,1)
