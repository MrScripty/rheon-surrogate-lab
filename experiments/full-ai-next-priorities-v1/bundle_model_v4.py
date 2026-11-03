"""Compact shared state encoder; cheap joint horizon heads and learned global pressure."""
import torch
from torch import nn
from torch.nn import functional as F
from bundle_model import pressure_basis
from adaptive_pool import adaptive_pool3d
from temporal_features_v2 import geometry,packed_state,auxiliary,unpack_outputs

class BundleNet(nn.Module):
    def __init__(self,horizons,width=4):
        super().__init__()
        if horizons not in (1,2,3,4):raise ValueError('Registered horizon required')
        self.horizons=horizons;self.width=width
        def conv(inputs,outputs):return nn.Conv3d(inputs,outputs,3,padding=1,padding_mode='replicate')
        self.fine=nn.Sequential(conv(15,width),nn.SiLU(),nn.Conv3d(width,width,1),nn.SiLU())
        self.coarse=nn.Sequential(nn.Conv3d(width,2*width,1),nn.SiLU())
        self.decode=nn.Sequential(nn.Conv3d(3*width,width,1),nn.SiLU())
        self.head=nn.Conv3d(width+3*horizons,5*horizons,1)
        self.modal=nn.Sequential(nn.Linear((2*width+3*horizons)*27,32),nn.SiLU(),nn.Linear(32,63*horizons))
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.modal[-1].weight);nn.init.zeros_(self.modal[-1].bias)

    def forward(self,state,features):
        if state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:raise ValueError('Causal channel contract differs')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]))
        # Encode current state/current forcing once. Remaining prescribed force
        # fields enter inexpensive heads and global pooling, not full convolutions.
        fine=self.fine(torch.cat((state,features[:,:10]),dim=1))
        low=self.coarse(F.avg_pool3d(fine,2,ceil_mode=True))
        decoded=self.decode(torch.cat((fine,F.interpolate(low,size=fine.shape[2:],mode='nearest')),dim=1))
        future_forces=features[:,7:]
        delta=self.head(torch.cat((decoded,future_forces),dim=1)).reshape(state.shape[0],self.horizons,5,*state.shape[2:])
        pooled=torch.cat((adaptive_pool3d(low,(3,3,3)).flatten(1),
                          adaptive_pool3d(future_forces,(3,3,3)).flatten(1)),dim=1)
        coefficients=self.modal(pooled).reshape(state.shape[0],self.horizons,63)
        coefficients=torch.cat((coefficients.new_zeros((*coefficients.shape[:2],1)),coefficients),dim=2).reshape(-1,4,4,4)
        z,y,x=pressure_basis(counts,str(state.device))
        pressure=torch.einsum('mx,bklm->bklx',x,coefficients)
        pressure=torch.einsum('ly,bklx->bkyx',y,pressure)
        pressure=torch.einsum('kz,bkyx->bzyx',z,pressure).reshape(state.shape[0],self.horizons,*state.shape[2:])
        pressure=state[:,None,0]+delta[:,:,0]+pressure
        pressure=pressure-pressure[:,:,:1,:1,:1]
        _,mask=geometry(counts,str(state.device))
        fields=[pressure]
        for index in (1,2,3):fields.append((state[:,None,index]+delta[:,:,index])*mask[index])
        fields.append(state[:,None,4]+delta[:,:,4])
        return torch.stack(fields,dim=2)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(state,features),scene['counts'],scales)
