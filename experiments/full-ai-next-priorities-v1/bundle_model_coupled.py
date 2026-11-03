"""Causal full-state generation with shared encoding and coupled neural readouts.

Transport is learned locally. Pressure requests and momentum share pretrained
neural flux coefficients, and the global pressure readout reuses trained neural
unit responses. No native transport, pressure solver or external projection is
called. Every internal update and physical output belongs in block latency.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from full_learned_pressure_map import FullLearnedPressureMap
from frozen_temporal_pressure_core import ReplicatedConvolution
from tied_momentum_readout import index_readout
from bundle_model_decoder import ConvexTracerHead,donors
from temporal_features_v2 import geometry,packed_state,auxiliary,unpack_outputs
from temporal_feedback import next_input


class LearnedFlux(nn.Conv3d):
    def forward(self,velocity):
        return F.conv3d(F.pad(velocity.double(),(1,1,1,1,1,1),mode='replicate'),self.weight.double())[:,:,:-1,:-1,:-1]


class LearnedMomentum(nn.Module):
    def forward(self,pressure,flux_weight):return index_readout(pressure,flux_weight)


class MovingTracer(nn.Module):
    def __init__(self,width):
        super().__init__();self.head=ConvexTracerHead(3*width+7,1)
        # The motion gate already controls the small native-step scale.
        # Use a broad learnable proposal instead of a nearly identity proposal.
        self.head.center_prior.zero_();self.head.center_prior[:,13]=2.

    def forward(self,tracer,velocity,hidden,left,right,alpha):
        speed=velocity.abs().amax(dim=(1,2,3,4),keepdim=True)
        relative=velocity/speed.clamp_min(1e-8)
        courant=alpha*speed
        context=torch.cat((hidden,left,right,velocity,relative,courant.expand_as(tracer)),dim=1)
        proposed=self.head(donors(tracer),context)
        gate=(4*courant).clamp(max=1.)
        return (tracer.double()+gate.double()*(proposed.double()-tracer.double())).float()


class BundleNet(nn.Module):
    def __init__(self,horizons,width,architecture,forcing_scale,increment_scales,scales,asset_directory):
        super().__init__()
        if horizons not in(1,2,3,4)or width!=12 or architecture!='coupled':
            raise ValueError('Registered coupled K1/2/3/4 architecture required')
        self.horizons=horizons;self.width=width;self.architecture=architecture
        self.increment_scales=increment_scales;self.scales=scales;self.forcing_scale=forcing_scale
        assets=Path(asset_directory);provenance=json.loads((assets/'request-initialization.json').read_text())
        weights=assets/'request-weights.npz'
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=provenance['weights_sha256']or provenance['velocity_scale']!=scales['velocity']:
            raise ValueError('Managed learned flux/physical scale identity changed')
        self.pressure_core=FullLearnedPressureMap(assets/'pressure',assets/'gate-bounds.json')
        with torch.random.fork_rng(devices=[]):self.flux=LearnedFlux(3,1,3,bias=False)
        with np.load(weights,allow_pickle=False)as arrays:
            with torch.no_grad():self.flux.weight.copy_(torch.from_numpy(arrays['flux.weight'].copy()))
        self.flux.weight.requires_grad_(False);self.momentum=LearnedMomentum()
        self.fluid=nn.Sequential(ReplicatedConvolution(10,width,3,padding=1,padding_mode='replicate'),nn.SiLU(),
            nn.Conv3d(width,width,1),nn.SiLU())
        self.left=ReplicatedConvolution(3,width,3,padding=1,padding_mode='replicate',bias=False)
        self.right=ReplicatedConvolution(3,width,3,padding=1,padding_mode='replicate',bias=False)
        self.local_update=nn.Sequential(nn.Conv3d(3*width+3,width,1),nn.SiLU(),nn.Conv3d(width,width,1))
        self.transport_context=nn.Conv3d(width,3*width,1)
        nn.init.zeros_(self.transport_context.weight);nn.init.zeros_(self.transport_context.bias)
        self.transport_head=nn.Conv3d(3*width,3,1,bias=False);nn.init.zeros_(self.transport_head.weight)
        self.viscous_head=ReplicatedConvolution(3,3,3,padding=1,padding_mode='replicate',bias=False)
        nn.init.zeros_(self.viscous_head.weight)
        self.tracer_readout=MovingTracer(width)

    def forward(self,state,features):
        if state.ndim!=5 or state.shape[0]!=1 or state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('One causal physical state and prescribed forcing block required')
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
            velocity=current[:,1:4];tracer=current[:,4:5]
            left=self.left(velocity);right=self.right(velocity)
            hidden=hidden+self.local_update(torch.cat((hidden,left,right,velocity),dim=1))
            # Tracer uses entering velocity before this step's force/pressure.
            predicted_tracer=self.tracer_readout(tracer,velocity,hidden,left,right,alpha)
            polynomial=torch.cat((left*right,left.abs()*right,alpha*left*right.square()),dim=1)
            transport=alpha*self.transport_head(polynomial*(1+self.transport_context(hidden)))
            entering=velocity+force[:,step]
            provisional=(entering+transport+beta*self.viscous_head(entering))*mask[1:4]
            provisional_physical=(provisional*self.scales['velocity']).float()
            q=self.flux(provisional_physical)*(rho*h/dt).reshape(-1,1,1,1,1)
            q=q-q.mean(dim=(2,3,4),keepdim=True)
            physical_pressure=self.pressure_core(q)
            physical_velocity=(provisional_physical.double()+(dt/(rho*h)).reshape(-1,1,1,1,1)*self.momentum(physical_pressure,self.flux.weight)).float()
            predicted_velocity=physical_velocity/self.scales['velocity']
            predicted_pressure=F.pad((physical_pressure/self.scales['pressure']).float(),(0,1,0,1,0,1),mode='replicate')
            output=torch.cat((predicted_pressure,predicted_velocity*mask[1:4],predicted_tracer),dim=1)
            outputs.append(output)
            if step+1<self.horizons:current=next_input(output[:,None],counts,self.scales)
        return torch.stack(outputs,dim=1)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        state=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(state,features),scene['counts'],scales)
