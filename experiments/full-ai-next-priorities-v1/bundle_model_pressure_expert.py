"""Causal full-state bundles with one frozen learned pressure expert per block.

All spatial fluid coefficients are learned. The velocity readout is a learned
convolution conditioned on predicted pressure, not an analytical pressure
projection. Future requests/endpoints are never inference inputs. Tracer cannot
influence the fluid branch. Fixed unit conversions and prescribed forcing are
included in complete block work.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from factored_request_model import CausalRequestNet
from frozen_temporal_pressure_core import FrozenPressureCore,ReplicatedConvolution
from temporal_features_v2 import geometry,packed_state,auxiliary,unpack_outputs


def zero_head(inputs,outputs,kernel=1):
    head=ReplicatedConvolution(inputs,outputs,kernel,padding=kernel//2,padding_mode='replicate',bias=False)
    nn.init.zeros_(head.weight);return head


class BundleNet(nn.Module):
    def __init__(self,horizons,width,architecture,forcing_scale,increment_scales,scales,asset_directory):
        super().__init__()
        if horizons not in (2,3,4)or width!=16 or architecture not in ('endpoint','reuse'):
            raise ValueError('Registered pressure-expert sharing architecture required')
        self.horizons=horizons;self.width=width;self.architecture=architecture
        self.forcing_scale=float(forcing_scale);self.increment_scales=increment_scales;self.scales=scales
        assets=Path(asset_directory);provenance=json.loads((assets/'request-initialization.json').read_text())
        weights=assets/'request-weights.npz'
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=provenance['weights_sha256']:
            raise ValueError('Managed causal request initialization changed')
        if provenance['velocity_scale']!=scales['velocity']:raise ValueError('Request/full-state velocity units differ')
        self.q_scale=provenance['q_scale']
        self.pressure_core=FrozenPressureCore(assets/'pressure')
        self.request=CausalRequestNet(1,scales['velocity'],self.q_scale)
        with np.load(weights,allow_pickle=False)as arrays:
            self.request.load_state_dict({name:torch.from_numpy(arrays[name].copy())for name in arrays.files},strict=True)
        # Preserve the learned request architecture/weights, while using the
        # qualified deterministic replicate convolution implementation.
        for name in ('flux','left','right','head'):
            original=getattr(self.request,name)
            replacement=ReplicatedConvolution(original.in_channels,original.out_channels,original.kernel_size,
                padding=original.padding,padding_mode=original.padding_mode,bias=False)
            replacement.weight=original.weight;setattr(self.request,name,replacement)
        self.request.flux.weight.requires_grad_(False)
        self.fluid=nn.Sequential(ReplicatedConvolution(3+7+3*horizons,width,3,padding=1,padding_mode='replicate'),
            nn.SiLU(),nn.Conv3d(width,width,1),nn.SiLU())
        self.request_head=zero_head(width,1,3)
        self.pressure_velocity=zero_head(1,3,3)
        self.velocity_head=zero_head(width,3)
        # One passive-tracer encoder per block. Products give the decoder
        # access to transport interactions without any fixed advection stencil.
        self.tracer=nn.Sequential(ReplicatedConvolution(width+1+3+3+3+3,8,3,padding=1,padding_mode='replicate'),nn.SiLU())
        self.tracer_head=zero_head(8,1)
        if architecture=='endpoint':
            self.middle=nn.Sequential(nn.Conv3d(width+4+3*horizons,width,1),nn.SiLU())
            self.middle_p=zero_head(width,horizons-1)
            self.middle_v=zero_head(width,3*(horizons-1))
            self.middle_tracer=nn.Sequential(nn.Conv3d(8+1+1+3+3*horizons,8,1),nn.SiLU())
            self.middle_t=zero_head(8,horizons-1)
        else:
            self.latent_update=nn.Sequential(nn.Conv3d(width+4+3,width,1),nn.SiLU(),nn.Conv3d(width,width,1))
            self.later_p=zero_head(width,1)
            self.later_v=zero_head(width,3)
            self.later_tracer=nn.Sequential(nn.Conv3d(8+1+3+3,8,1),nn.SiLU())
            self.later_t=zero_head(8,1)

    def physical_scalars(self,features):
        h=features[:,6,0,0,0].double().exp()/16
        return torch.stack((h,torch.full_like(h,1/60),torch.full_like(h,1000),torch.full_like(h,.001)),dim=1)

    def velocity(self,current,forces,pressure,hidden,horizon,scalars,head):
        cumulative=forces[:,:horizon].sum(dim=1)
        h,dt,rho,_=scalars.unbind(1)
        units=(dt/(rho*h)*self.scales['pressure']/self.scales['velocity']).float().reshape(-1,1,1,1,1)
        # This convolution's coefficients start at zero and are learned from
        # full-state labels. No analytical pressure-gradient stencil is loaded.
        return current[:,1:4]+cumulative+units*self.pressure_velocity(pressure)+head(hidden)*self.increment_scales[str(horizon)]['velocity']

    @staticmethod
    def output(pressure,velocity,tracer,counts):
        _,mask=geometry(counts,str(pressure.device))
        return torch.cat((pressure,velocity*mask[1:4],tracer),dim=1)

    def forward(self,state,features):
        if state.ndim!=5 or state.shape[0]!=1 or state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('One causal full state and prescribed forcing block required')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]));nx,ny,nz=counts
        force=features[:,7:].reshape(state.shape[0],self.horizons,3,*state.shape[2:])
        known=features[:,7:]/self.forcing_scale
        hidden=self.fluid(torch.cat((state[:,1:4],features[:,:7],known),dim=1))
        scalars=self.physical_scalars(features)
        anchor=self.horizons if self.architecture=='endpoint'else 1
        q=self.request(state[:,1:4],force[:,anchor-1],scalars)[:,0]
        correction=self.request_head(hidden)[:,:,:nz,:ny,:nx]
        q=q+(correction-correction.mean(dim=(2,3,4),keepdim=True))
        pressure=self.pressure_core(q.double()*self.q_scale)/self.scales['pressure']
        pressure=F.pad(pressure.float(),(0,1,0,1,0,1),mode='replicate')
        velocity=self.velocity(state,force,pressure,hidden,anchor,scalars,self.velocity_head)
        _,mask=geometry(counts,str(state.device));velocity=velocity*mask[1:4]
        tracer_hidden=self.tracer(torch.cat((hidden,state[:,4:5],state[:,1:4],state[:,4:5]*state[:,1:4],
            velocity,state[:,4:5]*velocity),dim=1))
        tracer=state[:,4:5]+self.tracer_head(tracer_hidden)*self.increment_scales[str(anchor)]['tracer']
        first=self.output(pressure,velocity,tracer,counts)
        if self.architecture=='endpoint':
            middle=self.middle(torch.cat((hidden,pressure,velocity,known),dim=1))
            p_delta=self.middle_p(middle);v_raw=self.middle_v(middle).reshape(1,self.horizons-1,3,*state.shape[2:])
            t_hidden=self.middle_tracer(torch.cat((tracer_hidden,state[:,4:5],tracer,velocity,known),dim=1))
            t_raw=self.middle_t(t_hidden)
            result=[]
            for k in range(1,self.horizons):
                p=pressure+p_delta[:,k-1:k]
                p=p-p[:,:,:1,:1,:1]
                h,dt,rho,_=scalars.unbind(1)
                units=(dt/(rho*h)*self.scales['pressure']/self.scales['velocity']).float().reshape(-1,1,1,1,1)
                v=state[:,1:4]+force[:,:k].sum(dim=1)+units*self.pressure_velocity(p)+v_raw[:,k-1]*self.increment_scales[str(k)]['velocity']
                t=state[:,4:5]+t_raw[:,k-1:k]*self.increment_scales[str(k)]['tracer']
                result.append(self.output(p,v,t,counts))
            return torch.stack([*result,first],dim=1)
        outputs=[first];previous=first
        for k in range(2,self.horizons+1):
            known_step=force[:,k-1]/self.forcing_scale
            hidden=hidden+self.latent_update(torch.cat((hidden,previous[:,:4],known_step),dim=1))
            p=previous[:,:1]+self.later_p(hidden);p=p-p[:,:,:1,:1,:1]
            v=self.velocity(state,force,p,hidden,k,scalars,self.later_v)*mask[1:4]
            t_features=self.later_tracer(torch.cat((tracer_hidden,previous[:,4:5],v,known_step),dim=1))
            t=state[:,4:5]+self.later_t(t_features)*self.increment_scales[str(k)]['tracer']
            previous=self.output(p,v,t,counts);outputs.append(previous)
        return torch.stack(outputs,dim=1)

    def predict(self,fields,scene,step,normalization):
        device=next(self.parameters()).device;scales=normalization['scales']
        current=packed_state(fields,tuple(scene['counts']),scales,device)
        features=auxiliary(scene,step,self.horizons,scales,device)
        return unpack_outputs(self(current,features),scene['counts'],scales)
