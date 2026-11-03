"""Same learned graph with explicit physical staggered-coordinate arithmetic."""
import torch
from torch.nn import functional as F
from recurrent_sampler_reference import coordinates,common,interpolate,sparse_stage


def transport(fields,counts,weight,gate,mix,dt_over_h):
    padded=common(fields[:3],counts).double()
    upper=torch.tensor(counts,dtype=torch.float64).reshape(3,1,1,1)
    projected=F.conv3d(F.pad(padded,(1,1,1,1,1,1),mode='replicate'),weight)
    outputs=[]
    for kind,field in enumerate(fields):
        base=coordinates(tuple(counts),kind);shape=field.shape
        point=projected[0,kind*3:kind*3+3,:shape[0],:shape[1],:shape[2]]
        offset=torch.tensor([0. if kind==a else .5 for a in range(3)],dtype=torch.float64).reshape(3,1,1,1)
        query=base+offset+(dt_over_h*point)*gate[kind,:,None,None,None]
        query=torch.minimum(query.clamp_min(0),upper)
        sampled=[]
        for component in range(3):
            phase=torch.tensor([0. if component==a else .5 for a in range(3)],dtype=torch.float64).reshape(3,1,1,1)
            sampled.append(interpolate(fields[component],query-phase))
        sampled=torch.stack(sampled);delta=[]
        for axis in range(3):
            value=torch.zeros_like(sampled[0])
            for component in range(3):value=value+mix[kind,axis,component]*sampled[component]
            delta.append((dt_over_h*value).clamp(-1,1))
        physical=torch.minimum((base+offset+torch.stack(delta)).clamp_min(0),upper)
        output=interpolate(field,physical-offset).float()
        if kind<3:
            selection=[slice(None)]*3;selection[2-kind]=[0,-1];output[tuple(selection)]=0.
        outputs.append(output)
    return outputs

