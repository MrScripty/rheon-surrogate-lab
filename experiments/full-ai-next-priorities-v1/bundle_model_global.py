"""Shared spatial encoder and two learned global blocks for every state field."""
import torch
from torch import nn
from torch.nn import functional as F
from learned_global import GlobalMix
from temporal_features_v2 import geometry,packed_state,auxiliary,unpack_outputs


class BundleNet(nn.Module):
    def __init__(self,horizons,width=12):
        super().__init__()
        if horizons not in(1,2,3,4)or width!=12:
            raise ValueError('Registered shared-global architecture required')
        self.horizons=horizons
        self.width=width
        self.fine=nn.Sequential(nn.Conv3d(15,width,3,padding=1,padding_mode='replicate'),nn.SiLU(),
                                nn.Conv3d(width,width,1),nn.SiLU())
        self.forcing=nn.Conv3d(3*horizons,width,1)
        self.local=nn.ModuleList([nn.Conv3d(width,width,1)for _ in range(2)])
        self.global_blocks=nn.ModuleList([GlobalMix(width)for _ in range(2)])
        self.decode=nn.Sequential(nn.Conv3d(2*width,width,3,padding=1,padding_mode='replicate'),nn.SiLU())
        self.head=nn.Conv3d(width+3*horizons,5*horizons,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,state,features):
        if state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('Causal state/known-force channel contract differs')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]))
        fine=self.fine(torch.cat((state,features[:,:10]),dim=1))
        forces=features[:,7:]
        hidden=fine+self.forcing(forces)
        for local,global_block in zip(self.local,self.global_blocks):
            hidden=F.silu(local(hidden)+global_block(hidden,features,counts))
        decoded=self.decode(torch.cat((fine,hidden),dim=1))
        delta=self.head(torch.cat((decoded,forces),dim=1)).reshape(state.shape[0],self.horizons,5,*state.shape[2:])
        pressure=state[:,None,0]+delta[:,:,0]
        pressure=pressure-pressure[:,:,:1,:1,:1]
        _,mask=geometry(counts,str(state.device))
        fields=[pressure]
        for index in(1,2,3):
            fields.append((state[:,None,index]+delta[:,:,index])*mask[index])
        fields.append(state[:,None,4]+delta[:,:,4])
        return torch.stack(fields,dim=2)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device
        scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(state,features),scene['counts'],scales)
