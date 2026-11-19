"""Endpoint-conditioned decoding and retained-feature generation of full states.

Both models encode current velocity/tracer and prescribed forcing once. Previous
pressure is deliberately excluded after the pressure-history shortcut diagnosis.
GlobalMix contains learned modal weights, never an analytical pressure inverse.
"""
import torch
from torch import nn
from torch.nn import functional as F
from learned_global import GlobalMix
from temporal_features_v2 import geometry,packed_state,auxiliary,unpack_outputs


def zero_head(inputs,outputs):
    head=nn.Conv3d(inputs,outputs,1)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    return head


class BundleNet(nn.Module):
    def __init__(self,horizons,width=12,architecture='endpoint'):
        super().__init__()
        if horizons not in (2,3,4) or width!=12 or architecture not in ('endpoint','reuse'):
            raise ValueError('Registered endpoint/reuse K2/3/4 architecture required')
        self.horizons=horizons;self.width=width;self.architecture=architecture
        self.fine=nn.Sequential(nn.Conv3d(14,width,3,padding=1,padding_mode='replicate'),nn.SiLU(),
                                nn.Conv3d(width,width,1),nn.SiLU())
        self.forcing=nn.Conv3d(3*horizons,width,1)
        self.local=nn.ModuleList([nn.Conv3d(width,width,1)for _ in range(2)])
        self.global_blocks=nn.ModuleList([GlobalMix(width)for _ in range(2)])
        if architecture=='endpoint':
            self.endpoint_head=zero_head(width+3*horizons,5)
            # One small shared decoder consumes the *predicted* endpoint. It
            # produces every intermediate state jointly, including pressure.
            self.intermediate=nn.Sequential(nn.Conv3d(width+4+5+3*horizons,width,1),nn.SiLU())
            self.intermediate_head=zero_head(width,5*(horizons-1))
        else:
            self.first_head=zero_head(width+3,5)
            # Later updates operate on retained features and the preceding
            # predicted physical state; they do not rerun the spatial encoder.
            self.latent_update=nn.Sequential(nn.Conv3d(width+5+3,width,1),nn.SiLU(),nn.Conv3d(width,width,1))
            self.later_head=zero_head(width+3,5)

    def reconstruct(self,raw,current,counts):
        pressure=raw[:,:,0]
        pressure=pressure-pressure[:,:,:1,:1,:1]
        _,mask=geometry(counts,str(current.device))
        values=[pressure]
        for index in (1,2,3):values.append((current[:,None,index]+raw[:,:,index])*mask[index])
        values.append(current[:,None,4]+raw[:,:,4])
        return torch.stack(values,dim=2)

    def forward(self,state,features):
        if state.ndim!=5 or state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('Causal physical state and known forcing required')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]))
        fine=self.fine(torch.cat((state[:,1:],features[:,:10]),dim=1))
        forces=features[:,7:]
        hidden=fine+self.forcing(forces)
        for local,global_block in zip(self.local,self.global_blocks):
            hidden=F.silu(local(hidden)+global_block(hidden,features,counts))
        if self.architecture=='endpoint':
            endpoint=self.reconstruct(self.endpoint_head(torch.cat((hidden,forces),dim=1))[:,None],state,counts)
            middle=self.intermediate(torch.cat((hidden,state[:,1:],endpoint[:,0],forces),dim=1))
            raw=self.intermediate_head(middle).reshape(state.shape[0],self.horizons-1,5,*state.shape[2:])
            return torch.cat((self.reconstruct(raw,state,counts),endpoint),dim=1)
        first=self.reconstruct(self.first_head(torch.cat((hidden,forces[:,:3]),dim=1))[:,None],state,counts)
        outputs=[first];previous=first[:,0]
        for horizon in range(1,self.horizons):
            known=forces[:,3*horizon:3*(horizon+1)]
            hidden=hidden+self.latent_update(torch.cat((hidden,previous,known),dim=1))
            raw=self.later_head(torch.cat((hidden,known),dim=1))[:,None]
            # Increments refer to the original block state, while the retained
            # latent state and previous prediction carry horizon information.
            generated=self.reconstruct(raw,state,counts)
            outputs.append(generated);previous=generated[:,0]
        return torch.cat(outputs,dim=1)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        current=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(current,features),scene['counts'],scales)
