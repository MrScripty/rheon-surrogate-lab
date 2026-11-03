"""Lead-conditioned fluid decoding and learned convex passive-tracer readouts.

Pressure solves and post-projections remain absent. Tracer interpolation weights
are predicted by neural networks from fluid features only; no native backtrace
or analytical advection rule supplies them. Fixed local gathers are neural
readout support, and all costs belong to complete block inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from bundle_model_pressure_expert import BundleNet as ExpertBundle
from frozen_temporal_pressure_core import ReplicatedConvolution
from temporal_features_v2 import geometry


class ZeroDCConvolution(ReplicatedConvolution):
    def _conv_forward(self,value,weight,bias):
        weight=weight-weight.mean(dim=(2,3,4),keepdim=True)
        return super()._conv_forward(value,weight,bias)


def donors(value):
    """One shared 27-donor representation of a tracer field."""
    padded=F.pad(value,(1,1,1,1,1,1),mode='replicate');z,y,x=value.shape[2:]
    return torch.cat([padded[:,:,a:a+z,b:b+y,c:c+x]
        for a in range(3)for b in range(3)for c in range(3)],dim=1)


class ConvexTracerHead(nn.Module):
    def __init__(self,inputs,groups):
        super().__init__()
        if groups not in (1,2):raise ValueError('Current and optional predicted tracer donors required')
        channels=27*groups
        self.coefficients=nn.Sequential(nn.Conv3d(inputs,16,1),nn.SiLU(),nn.Conv3d(16,channels,1))
        nn.init.zeros_(self.coefficients[-1].weight);nn.init.zeros_(self.coefficients[-1].bias)
        prior=torch.zeros(1,channels,1,1,1)
        prior[:,13]=10.
        if groups==2:prior[:,40]=9.
        self.register_buffer('center_prior',prior,persistent=False)

    def forward(self,values,context):
        logits=self.coefficients(context)+self.center_prior
        weights=logits.double().softmax(dim=1)
        return (weights*values.double()).sum(dim=1,keepdim=True).float()


def centered(pressure,counts):
    nx,ny,nz=counts
    mean=pressure[:,:,:nz,:ny,:nx].double().mean(dim=(2,3,4),keepdim=True)
    return (pressure.double()-mean).float()


class BundleNet(ExpertBundle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep the seeded fluid/request parameter tensors of the prior model.
        # The passive-tracer architecture changes and is reported separately.
        with torch.random.fork_rng(devices=[]):
            replacement=ZeroDCConvolution(1,3,3,padding=1,padding_mode='replicate',bias=False)
            replacement.weight=self.pressure_velocity.weight;self.pressure_velocity=replacement
            del self.tracer,self.tracer_head
            inputs=self.width+6+3*self.horizons
            self.tracer_anchor=ConvexTracerHead(inputs,1)
            if self.architecture=='endpoint':
                del self.middle_tracer,self.middle_t
                self.tracer_middle=ConvexTracerHead(inputs+1,2)
            else:
                del self.later_tracer,self.later_t
                self.tracer_later=ConvexTracerHead(self.width+6+3+1,2)

    def velocity(self,current,forces,pressure,hidden,horizon,scalars,head):
        cumulative=forces[:,:horizon].sum(dim=1);h,dt,rho,_=scalars.unbind(1)
        units=(horizon*dt/(rho*h)*self.scales['pressure']/self.scales['velocity']).float().reshape(-1,1,1,1,1)
        return current[:,1:4]+cumulative+units*self.pressure_velocity(pressure)+head(hidden)*self.increment_scales[str(horizon)]['velocity']

    def forward(self,state,features):
        if state.ndim!=5 or state.shape[0]!=1 or state.shape[1]!=5 or features.shape[1]!=7+3*self.horizons:
            raise ValueError('One causal full state and prescribed forcing block required')
        counts=tuple(int(n)-1 for n in reversed(state.shape[2:]));nx,ny,nz=counts
        force=features[:,7:].reshape(1,self.horizons,3,*state.shape[2:]);known=features[:,7:]/self.forcing_scale
        hidden=self.fluid(torch.cat((state[:,1:4],features[:,:7],known),dim=1))
        scalars=self.physical_scalars(features);anchor=self.horizons if self.architecture=='endpoint'else 1
        q=self.request(state[:,1:4],force[:,anchor-1],scalars)[:,0]
        correction=self.request_head(hidden)[:,:,:nz,:ny,:nx]
        q=q+(correction-correction.mean(dim=(2,3,4),keepdim=True))
        pressure=self.pressure_core(q.double()*self.q_scale)/self.scales['pressure']
        pressure=centered(pressure,counts)
        pressure=F.pad(pressure,(0,1,0,1,0,1),mode='replicate')
        _,mask=geometry(counts,str(state.device))
        velocity=self.velocity(state,force,pressure,hidden,anchor,scalars,self.velocity_head)*mask[1:4]
        initial_donors=donors(state[:,4:5])
        tracer=self.tracer_anchor(initial_donors,torch.cat((hidden,state[:,1:4],velocity,known),dim=1))
        first=self.output(pressure,velocity,tracer,counts)
        if self.architecture=='endpoint':
            middle=self.middle(torch.cat((hidden,pressure,velocity,known),dim=1))
            p_delta=self.middle_p(middle);v_raw=self.middle_v(middle).reshape(1,self.horizons-1,3,*state.shape[2:])
            tracer_values=torch.cat((initial_donors,donors(tracer)),dim=1)
            outputs=[]
            for k in range(1,self.horizons):
                p=centered(pressure+p_delta[:,k-1:k],counts)
                h,dt,rho,_=scalars.unbind(1)
                units=(k*dt/(rho*h)*self.scales['pressure']/self.scales['velocity']).float().reshape(-1,1,1,1,1)
                v=state[:,1:4]+force[:,:k].sum(dim=1)+units*self.pressure_velocity(p)+v_raw[:,k-1]*self.increment_scales[str(k)]['velocity']
                v=v*mask[1:4];lead=torch.full_like(p,k/self.horizons)
                t=self.tracer_middle(tracer_values,torch.cat((middle,velocity,v,known,lead),dim=1))
                outputs.append(self.output(p,v,t,counts))
            return torch.stack([*outputs,first],dim=1)
        outputs=[first];previous=first
        for k in range(2,self.horizons+1):
            known_step=force[:,k-1]/self.forcing_scale
            hidden=hidden+self.latent_update(torch.cat((hidden,previous[:,:4],known_step),dim=1))
            p=centered(previous[:,:1]+self.later_p(hidden),counts)
            v=self.velocity(state,force,p,hidden,k,scalars,self.later_v)*mask[1:4]
            lead=torch.full_like(p,k/self.horizons)
            values=torch.cat((initial_donors,donors(previous[:,4:5])),dim=1)
            t=self.tracer_later(values,torch.cat((hidden,previous[:,1:4],v,known_step,lead),dim=1))
            previous=self.output(p,v,t,counts);outputs.append(previous)
        return torch.stack(outputs,dim=1)
