"""Causal shared-feature full-state prediction with neural displacement readouts.

All displacement coefficients come from learned networks. Native backtracing
and label-generation code are absent from this inference module.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bundle_model_coupled import LearnedFlux, LearnedMomentum
from full_learned_pressure_map import FullLearnedPressureMap
from frozen_temporal_pressure_core import ReplicatedConvolution
from displacement_readout import convex_field_readout
from temporal_features_v2 import geometry, packed_state, auxiliary, unpack_outputs
from temporal_feedback import next_input


def replicate_one(value):
    # Explicit repetition supports deterministic CUDA backward without a pad kernel.
    for axis in range(2,5):
        value=torch.cat((value.narrow(axis,0,1),value,value.narrow(axis,value.shape[axis]-1,1)),dim=axis)
    return value


class DisplacementHead(nn.Module):
    def __init__(self,width,point_weights):
        super().__init__();self.width=width
        with np.load(point_weights,allow_pickle=False)as arrays:
            if arrays.files!=['projection.weight']or arrays['projection.weight'].shape!=(12,3,3,3,3)or arrays['projection.weight'].dtype!=np.float32:
                raise ValueError('Exact learned float32 point coefficient inventory required')
            point=torch.from_numpy(arrays['projection.weight'].copy())
        self.point_weight=nn.Parameter(point,requires_grad=False)
        self.local_weight=nn.Parameter(torch.empty(2*width,3,3,3,3))
        nn.init.kaiming_uniform_(self.local_weight,a=5**.5)
        self.context=nn.Conv3d(width,3*width,1)
        nn.init.zeros_(self.context.weight);nn.init.zeros_(self.context.bias)
        self.curvature=nn.Conv3d(3*width,12,1,bias=False)
        nn.init.zeros_(self.curvature.weight)

    def forward(self,velocity,hidden,alpha):
        # One spatial projection shares the input neighborhood across frozen
        # learned point coefficients and trainable local features.
        projected=F.conv3d(replicate_one(velocity),torch.cat((self.point_weight,self.local_weight),dim=0))
        point=projected[:,:12];left=projected[:,12:12+self.width];right=projected[:,12+self.width:]
        polynomial=torch.cat((left*right,left.abs()*right,alpha*left*right.square()),dim=1)
        correction=self.curvature(polynomial*(1+self.context(hidden)))
        shift=alpha*point+alpha.square()*correction
        return shift.clamp(-1,1).reshape(1,4,3,*velocity.shape[2:])


class BundleNet(nn.Module):
    def __init__(self,horizons,width,architecture,forcing_scale,increment_scales,scales,asset_directory):
        super().__init__()
        if horizons not in(1,2,3,4)or width!=12 or architecture!='displacement':
            raise ValueError('Registered displacement K1/2/3/4 architecture required')
        self.horizons=horizons;self.width=width;self.architecture=architecture
        self.forcing_scale=forcing_scale;self.increment_scales=increment_scales;self.scales=scales
        assets=Path(asset_directory);request=json.loads((assets/'request-initialization.json').read_text())
        weights=assets/'request-weights.npz'
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=request['weights_sha256']or request['velocity_scale']!=scales['velocity']:
            raise ValueError('Managed flux and physical scale identities changed')
        points=json.loads((assets/'point-initialization.json').read_text());point_weights=assets/'point-weights.npz'
        if hashlib.sha256(point_weights.read_bytes()).hexdigest()!=points['weights_sha256']or not points['native_qualified_float32']:
            raise ValueError('Managed native-qualified point coefficients required')
        self.pressure_core=FullLearnedPressureMap(assets/'pressure',assets/'gate-bounds.json')
        with torch.random.fork_rng(devices=[]):self.flux=LearnedFlux(3,1,3,bias=False)
        with np.load(weights,allow_pickle=False)as arrays:
            with torch.no_grad():self.flux.weight.copy_(torch.from_numpy(arrays['flux.weight'].copy()))
        self.flux.weight.requires_grad_(False);self.momentum=LearnedMomentum()
        self.fluid=nn.Sequential(ReplicatedConvolution(10,width,3,padding=1,padding_mode='replicate'),nn.SiLU(),
            nn.Conv3d(width,width,1),nn.SiLU())
        self.local_update=nn.Sequential(nn.Conv3d(width+3,width,1),nn.SiLU(),nn.Conv3d(width,width,1))
        self.displacement=DisplacementHead(width,point_weights)
        self.viscous_head=ReplicatedConvolution(3,3,3,padding=1,padding_mode='replicate',bias=False)
        nn.init.zeros_(self.viscous_head.weight)

    def forward(self,state,features):
        if state.ndim!=5 or state.shape[:2]!=(1,5)or features.shape[1]!=7+3*self.horizons:
            raise ValueError('One causal full state and prescribed-force block required')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]));nx,ny,nz=counts
        force=features[:,7:].reshape(1,self.horizons,3,*state.shape[2:])
        h=features[:,6,0,0,0].double().exp()/16
        dt=torch.full_like(h,1/60);rho=torch.full_like(h,1000.);mu=torch.full_like(h,.001)
        alpha=(dt/h*self.scales['velocity']).float().reshape(-1,1,1,1,1)
        beta=(mu/rho*dt/h.square()).float().reshape(-1,1,1,1,1)
        _,mask=geometry(counts,str(state.device))
        hidden=self.fluid(torch.cat((state[:,1:4],features[:,:7]),dim=1))
        current=state;outputs=[]
        for step in range(self.horizons):
            velocity=current[:,1:4]
            hidden=hidden+self.local_update(torch.cat((hidden,velocity),dim=1))
            shift=self.displacement(velocity,hidden,alpha)
            advected=convex_field_readout(current[:,1:5],shift)
            entering=advected[:,:3]+force[:,step]
            provisional=(entering+beta*self.viscous_head(entering))*mask[1:4]
            provisional_physical=(provisional*self.scales['velocity']).float()
            q=self.flux(provisional_physical)*(rho*h/dt).reshape(-1,1,1,1,1)
            q=q-q.mean(dim=(2,3,4),keepdim=True)
            physical_pressure=self.pressure_core(q)
            physical_velocity=(provisional_physical.double()+(dt/(rho*h)).reshape(-1,1,1,1,1)*self.momentum(physical_pressure,self.flux.weight)).float()
            pressure=F.pad((physical_pressure/self.scales['pressure']).float(),(0,1,0,1,0,1),mode='replicate')
            velocity=physical_velocity/self.scales['velocity']
            output=torch.cat((pressure,velocity*mask[1:4],advected[:,3:4]),dim=1)
            outputs.append(output)
            if step+1<self.horizons:current=next_input(output[:,None],counts,self.scales)
        return torch.stack(outputs,dim=1)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(state,features),scene['counts'],scales)
