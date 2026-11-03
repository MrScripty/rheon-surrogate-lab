"""Tensor reference for a neural coordinate sampler, with arbitrary learned gates.

Fixed interpolation only: this module predicts no coordinates, invokes no
native backtrace and contains no integration-stage coefficient.
"""
from functools import lru_cache
import torch
from torch.nn import functional as F


@lru_cache(maxsize=32)
def coordinates(counts,kind):
    nx,ny,nz=counts;shape=(nz+(kind==2),ny+(kind==1),nx+(kind==0))
    z,y,x=torch.meshgrid(*(torch.arange(n,dtype=torch.float64)for n in shape),indexing='ij')
    return torch.stack((x,y,z))


def interpolate(field,query):
    dims=tuple(reversed(field.shape));lo=[];hi=[];fractions=[]
    for axis,n in enumerate(dims):
        p=query[axis].clamp(0,n-1);a=p.floor().long()
        lo.append(a);hi.append((a+1).clamp_max(n-1));fractions.append(p-a)
    flat=field.double().flatten();result=torch.zeros_like(query[0]);lower=torch.full_like(result,float('inf'));upper=-lower
    for bz in range(2):
        for by in range(2):
            for bx in range(2):
                bits=(bx,by,bz);index=[hi[d]if bits[d]else lo[d]for d in range(3)]
                value=flat[index[0]+dims[0]*(index[1]+dims[1]*index[2])];weight=torch.ones_like(result)
                for d in range(3):weight=weight*(fractions[d]if bits[d]else 1-fractions[d])
                result=result+weight*value;lower=torch.minimum(lower,value);upper=torch.maximum(upper,value)
    return torch.minimum(torch.maximum(result,lower),upper)


def common(fields,counts):
    nx,ny,nz=counts
    return torch.cat([F.pad(value[None,None],(0,nx+1-value.shape[2],0,ny+1-value.shape[1],0,nz+1-value.shape[0]),mode='replicate')for value in fields],dim=1)


def transport(fields,counts,weight,gate,mix,dt_over_h):
    padded=common(fields[:3],counts).double()
    projected=F.conv3d(F.pad(padded,(1,1,1,1,1,1),mode='replicate'),weight)
    outputs=[]
    for kind,field in enumerate(fields):
        base=coordinates(tuple(counts),kind);shape=field.shape
        point=projected[0,kind*3:kind*3+3,:shape[0],:shape[1],:shape[2]]
        offset=torch.tensor([0. if kind==a else .5 for a in range(3)],dtype=torch.float64).reshape(3,1,1,1)
        query=base+offset+(dt_over_h*point)*gate[kind,:,None,None,None]
        sampled=[]
        for component in range(3):
            phase=torch.tensor([0. if component==a else .5 for a in range(3)],dtype=torch.float64).reshape(3,1,1,1)
            sampled.append(interpolate(fields[component],query-phase))
        sampled=torch.stack(sampled);delta=[]
        for axis in range(3):
            value=torch.zeros_like(sampled[0])
            for component in range(3):value=value+mix[kind,axis,component]*sampled[component]
            delta.append((dt_over_h*value).clamp(-1,1))
        output=interpolate(field,base+torch.stack(delta)).float()
        if kind<3:
            selection=[slice(None)]*3;selection[2-kind]=[0,-1];output[tuple(selection)]=0.
        outputs.append(output)
    return outputs


def sparse_stage(kind,fields,counts,weight,scale,base=None):
    nx,ny,nz=counts
    if kind==2:
        packed=F.pad(fields[0][None,None],(0,1,0,1,0,1),mode='replicate')
    else:packed=common(fields,counts).double()
    result=F.conv3d(F.pad(packed,(1,1,1,1,1,1),mode='replicate'),weight)*scale
    if kind==1:return result[0,0,:nz,:ny,:nx]
    outputs=[]
    for component in range(3):
        shape=(nz+(component==2),ny+(component==1),nx+(component==0))
        value=(base[component].double()+result[0,component,:shape[0],:shape[1],:shape[2]]).float()
        selection=[slice(None)]*3;selection[2-component]=[0,-1];value[tuple(selection)]=0.
        outputs.append(value)
    return outputs
