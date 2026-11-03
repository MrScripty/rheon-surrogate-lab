"""Joint horizon supervision and physical diagnostics; no inference pressure solve."""
import torch
from torch.nn import functional as F

def laplacian(value):
    result=torch.zeros_like(value)
    for axis in (-3,-2,-1):
        left=[slice(None)]*value.ndim;right=left.copy()
        left[axis]=slice(None,-1);right[axis]=slice(1,None)
        difference=value[tuple(left)]-value[tuple(right)]
        result[tuple(left)]+=difference;result[tuple(right)]-=difference
    return result

def errors(predicted,target,counts):
    nx,ny,nz=counts
    p=predicted[:,:,0,:nz,:ny,:nx];reference=target[:,:,0,:nz,:ny,:nx]
    p=p-p.mean(dim=(-3,-2,-1),keepdim=True)
    reference=reference-reference.mean(dim=(-3,-2,-1),keepdim=True)
    def relative_square(difference,truth):
        numerator=difference.flatten(2).square().sum(2)
        denominator=truth.flatten(2).square().sum(2).clamp_min(truth[0,0].numel()*1e-16)
        return numerator/denominator
    pressure=relative_square(p-reference,reference)
    equation=relative_square(laplacian(p)-laplacian(reference),laplacian(reference))
    velocity=[];reference_velocity=[]
    for component,slices in ((1,(slice(0,nz),slice(0,ny),slice(0,nx+1))),
                              (2,(slice(0,nz),slice(0,ny+1),slice(0,nx))),
                              (3,(slice(0,nz+1),slice(0,ny),slice(0,nx)))):
        velocity.append(predicted[:,:,component][(slice(None),slice(None),*slices)].flatten(2))
        reference_velocity.append(target[:,:,component][(slice(None),slice(None),*slices)].flatten(2))
    v=torch.cat(velocity,dim=2);rv=torch.cat(reference_velocity,dim=2)
    velocity_error=relative_square(v-rv,rv)
    tracer=predicted[:,:,4,:nz,:ny,:nx];rt=target[:,:,4,:nz,:ny,:nx]
    tracer_error=relative_square(tracer-rt,rt)
    def divergence(state):
        vx=state[:,:,1,:nz,:ny,:nx+1];vy=state[:,:,2,:nz,:ny+1,:nx];vz=state[:,:,3,:nz+1,:ny,:nx]
        return vx[...,1:]-vx[...,:-1]+vy[:,:,:,1:,:]-vy[:,:,:,:-1,:]+vz[:,:,1:,:,:]-vz[:,:,:-1,:,:]
    div=divergence(predicted)-divergence(target)
    divergence_error=div.flatten(2).square().sum(2)/rv.square().sum(2).clamp_min(rv.shape[2]*1e-16)
    mass=(tracer.flatten(2).sum(2)-rt.flatten(2).sum(2)).square()/rt.flatten(2).sum(2).square().clamp_min(1e-16)
    energy=(v.square().sum(2)-rv.square().sum(2)).square()/rv.square().sum(2).square().clamp_min(1e-16)
    lower=F.relu(-tracer).flatten(2).square().mean(2)
    return dict(pressure=pressure,equation=equation,velocity=velocity_error,tracer=tracer_error,
                divergence=divergence_error,mass=mass,energy=energy,lower_bound=lower)

def loss(predicted,target,counts,tracer_scale):
    values=errors(predicted,target,counts)
    field=sum(values[name]for name in ('pressure','velocity','tracer'))/3
    nx,ny,nz=counts
    physical_tracer=predicted[:,:,4,:nz,:ny,:nx]*tracer_scale
    bounds=F.relu(-physical_tracer).square().mean()+F.relu(physical_tracer-1).square().mean()
    physical=.1*values['divergence']+.05*values['mass']+.02*values['energy']
    return (field+values['equation']+physical).mean()+.01*bounds
