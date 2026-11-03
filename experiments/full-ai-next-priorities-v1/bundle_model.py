"""Shared spatial encoding, joint future state heads and a learned global pressure branch."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from temporal_features import geometry,packed_state,auxiliary,unpack_outputs

@lru_cache(maxsize=64)
def pressure_basis(counts,device):
    with torch.inference_mode(False):
        bases=[]
        for n in reversed(counts):
            positions=torch.arange(n,device=device,dtype=torch.float32)+.5
            modes=torch.arange(4,device=device,dtype=torch.float32)
            axis=torch.cos(math.pi/n*modes[:,None]*positions[None,:])
            bases.append(F.pad(axis,(0,1),mode='replicate'))
    return tuple(bases)

class BundleNet(nn.Module):
    def __init__(self,horizons,width=8):
        super().__init__()
        if horizons not in (1,2,3,4):raise ValueError('Registered temporal horizon required')
        self.horizons=horizons;self.width=width
        def conv(inputs,outputs):return nn.Conv3d(inputs,outputs,3,padding=1,padding_mode='replicate')
        self.fine=nn.Sequential(conv(12+3*horizons,width),nn.SiLU(),conv(width,width),nn.SiLU())
        self.coarse=nn.Sequential(conv(width,2*width),nn.SiLU(),conv(2*width,2*width),nn.SiLU())
        self.decode=nn.Sequential(conv(3*width,width),nn.SiLU())
        self.head=nn.Conv3d(width,5*horizons,1)
        self.modal=nn.Sequential(nn.Linear(2*width*27,64),nn.SiLU(),nn.Linear(64,63*horizons))
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.modal[-1].weight);nn.init.zeros_(self.modal[-1].bias)

    def forward(self,state,features):
        if state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('Causal state/forcing channel contract differs')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]))
        fine=self.fine(torch.cat((state,features),dim=1))
        low=self.coarse(F.avg_pool3d(fine,2,ceil_mode=True))
        decoded=self.decode(torch.cat((fine,F.interpolate(low,size=fine.shape[2:],mode='nearest')),dim=1))
        delta=self.head(decoded).reshape(state.shape[0],self.horizons,5,*state.shape[2:])
        coefficients=self.modal(F.adaptive_avg_pool3d(low,(3,3,3)).flatten(1)).reshape(state.shape[0],self.horizons,63)
        coefficients=torch.cat((coefficients.new_zeros((*coefficients.shape[:2],1)),coefficients),dim=2).reshape(-1,4,4,4)
        z,y,x=pressure_basis(counts,str(state.device))
        global_pressure=torch.einsum('mx,bklm->bklx',x,coefficients)
        global_pressure=torch.einsum('ly,bklx->bkyx',y,global_pressure)
        global_pressure=torch.einsum('kz,bkyx->bzyx',z,global_pressure).reshape(state.shape[0],self.horizons,*state.shape[2:])
        # The coefficients are learned; no inverse eigenvalues or numerical solve occur.
        pressure=state[:,None,0]+delta[:,:,0]+global_pressure
        pressure=pressure-pressure[:,:,:1,:1,:1]
        values=[pressure]
        _,mask=geometry(counts,str(state.device))
        for field in (1,2,3):values.append((state[:,None,field]+delta[:,:,field])*mask[field])
        values.append(state[:,None,4]+delta[:,:,4])
        return torch.stack(values,dim=2)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        predicted=self(state,features)
        return unpack_outputs(predicted,scene['counts'],scales)
