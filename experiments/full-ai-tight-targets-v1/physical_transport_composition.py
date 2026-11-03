"""Frozen neural transport with physical carrier arithmetic and double pressure.

Only learned neural displacement/viscosity/flux/momentum coefficients are used.
No native backtrace, numerical pressure solve, reference or projection call.
"""
import torch
from torch import nn
from torch.nn import functional as F
from bundle_model_fast_preparation import displacement
from fused_field_readout import convex_field_readout
from fused_force_construction import auxiliary
from temporal_features_v2 import geometry


def physical_carriers(fields,counts):
    nx,ny,nz=counts;values=[]
    shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))
    for name,shape in zip(('vx','vy','vz','tracer'),shapes):
        value=torch.as_tensor(fields[name]).float().reshape(1,1,*shape)
        values.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(values,dim=1)


def physical_point_shift(head,velocity,normalized,hidden,alpha,dt_over_h):
    projected=F.conv3d(F.pad(normalized,(1,1,1,1,1,1),mode='replicate'),head.local_weight)
    left=projected[:,:head.width];right=projected[:,head.width:]
    polynomial=torch.cat((left*right,left.abs()*right,alpha*left*right.square()),dim=1)
    correction=head.curvature(polynomial*(1+head.context(hidden)))
    point=F.conv3d(F.pad(velocity,(1,1,1,1,1,1),mode='replicate'),head.point_weight)
    shift=dt_over_h*point.double()+alpha.double().square()*correction.double()
    return shift.clamp(-1,1).reshape(1,4,3,*velocity.shape[2:])


class PhysicalTransportComposition(nn.Module):
    def __init__(self,parent,mode):
        super().__init__()
        if mode not in ('physical_carriers','physical_carriers_and_point'):
            raise ValueError('Exactly two registered arithmetic hypotheses')
        self.parent=parent;self.mode=mode;self.horizons=parent.horizons

    def predict(self,fields,scene,step,normalization):
        network=self.parent;scales=normalization['scales'];counts=tuple(scene['counts']);nx,ny,nz=counts
        current=physical_carriers(fields,counts)
        features=auxiliary(scene,step,self.horizons,dict(scales,velocity=1.),'cpu')
        force=features[:,7:].reshape(1,self.horizons,3,*current.shape[2:])
        h=1./scene['cells_per_metre'];dt=1./60.;rho=1000.;mu=.001
        alpha=torch.tensor(dt/h*scales['velocity'],dtype=torch.float32).reshape(1,1,1,1,1)
        beta=torch.tensor(mu/rho*dt/h**2,dtype=torch.float32).reshape(1,1,1,1,1)
        _,mask=geometry(counts,'cpu')
        hidden=network.fluid(torch.cat((current[:,:3]/scales['velocity'],features[:,:7]),dim=1))
        results=[]
        for lead in range(self.horizons):
            normalized=current[:,:3]/scales['velocity']
            hidden=hidden+network.local_update(torch.cat((hidden,normalized),dim=1))
            if self.mode=='physical_carriers':shift=displacement(network.displacement,normalized,hidden,alpha)
            else:shift=physical_point_shift(network.displacement,current[:,:3],normalized,hidden,alpha,dt/h)
            advected=convex_field_readout(current,shift)
            entering=advected[:,:3]+force[:,lead]
            provisional=(entering+beta*network.viscous_head(entering))*mask[1:4]
            q=network.flux(provisional)*(rho*h/dt);q=q-q.mean(dim=(2,3,4),keepdim=True)
            pressure=network.pressure_core(q)
            velocity=(provisional.double()+dt/(rho*h)*network.momentum(pressure,network.flux.weight)).float()*mask[1:4]
            p=pressure-pressure[:,:,:1,:1,:1]
            values={'pressure':p.contiguous().flatten(),
                'vx':velocity[0,0,:nz,:ny,:nx+1].contiguous().flatten(),
                'vy':velocity[0,1,:nz,:ny+1,:nx].contiguous().flatten(),
                'vz':velocity[0,2,:nz+1,:ny,:nx].contiguous().flatten(),
                'tracer':advected[0,3,:nz,:ny,:nx].contiguous().flatten()}
            results.append(values)
            if lead+1<self.horizons:current=physical_carriers(values,counts)
        return results
