"""Differentiable 48-parameter head for the registered recurrent sampler graph."""
import torch
from torch import nn


class LearnedSamplerHead(nn.Module):
    def __init__(self,initial_mix):
        super().__init__()
        self.mid_gate=nn.Parameter(torch.zeros((4,3),dtype=torch.float64))
        self.mix=nn.Parameter(torch.tensor(initial_mix,dtype=torch.float64))
        if self.mix.shape!=(4,3,3):raise ValueError('Inherited learned mixing required')

    def forward(self,point,patch,kind,alpha):
        middle=point*self.mid_gate[kind]
        phase=torch.full_like(point,.5)
        for axis in range(3):phase[:,axis]=torch.where(kind==axis,0.,.5)
        values=[]
        for component in range(3):
            source_phase=point.new_tensor([0. if a==component else .5 for a in range(3)])
            query=phase-source_phase+middle*alpha[:,None]+1.
            if bool((query.detach()<0).any())or bool((query.detach()>2).any()):
                raise ValueError('Training local patch envelope exceeded; no hidden sampling fallback')
            lo=query.floor().long();hi=(lo+1).clamp_max(2);fraction=query-lo
            field=patch[:,component].reshape(-1,27).double()
            result=torch.zeros_like(point[:,0]);lower=torch.full_like(result,float('inf'));upper=-lower
            for z in range(2):
                for y in range(2):
                    for x in range(2):
                        bits=(x,y,z);index=[hi[:,a]if bits[a]else lo[:,a]for a in range(3)]
                        v=field.gather(1,(index[0]+3*(index[1]+3*index[2]))[:,None])[:,0]
                        weight=torch.ones_like(result)
                        for axis in range(3):weight=weight*(fraction[:,axis]if bits[axis]else 1-fraction[:,axis])
                        result=result+weight*v;lower=torch.minimum(lower,v);upper=torch.maximum(upper,v)
            values.append(torch.minimum(torch.maximum(result,lower),upper))
        sampled=torch.stack(values,1);mix=self.mix[kind];end=[]
        for axis in range(3):
            value=torch.zeros_like(sampled[:,0])
            for component in range(3):value=value+mix[:,axis,component]*sampled[:,component]
            end.append(value)
        return middle,torch.stack(end,1)


def objective(model,group,scales):
    middle,end=model(group['point'],group['patch'],group['kind'],group['alpha'])
    scale=scales[group['kind']]
    return (((middle-group['middle'])/scale).square().mean()+
            ((end-group['end'])/scale).square().mean())
