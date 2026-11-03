"""Controlled MSE loss: full-state relative fields versus fixed step-unit errors.

Inference, targets and evaluation errors are unchanged. The increment arm scales
velocity/tracer *prediction errors* by fixed aggregate training increment RMS;
it does not substitute reference history or ignore accumulated state error.
"""
import torch
from torch.nn import functional as F
from temporal_objective import errors


def loss(predicted,target,counts,tracer_scale,increment_scales,mode):
    if mode not in ('full_state','increments'):raise ValueError('Registered supervision arm required')
    values=errors(predicted,target,counts)
    if mode=='increments':
        nx,ny,nz=counts;velocity=[]
        for component,selection in ((1,(slice(0,nz),slice(0,ny),slice(0,nx+1))),
                                     (2,(slice(0,nz),slice(0,ny+1),slice(0,nx))),
                                     (3,(slice(0,nz+1),slice(0,ny),slice(0,nx)))):
            difference=(predicted[:,:,component]-target[:,:,component])[(slice(None),slice(None),*selection)]
            velocity.append(difference.flatten(2))
        velocity_error=torch.cat(velocity,dim=2).square().mean(2)
        tracer_error=(predicted[:,:,4,:nz,:ny,:nx]-target[:,:,4,:nz,:ny,:nx]).square().flatten(2).mean(2)
        def squared_scales(field):
            return predicted.new_tensor([increment_scales[str(k)][field]**2 for k in range(1,predicted.shape[1]+1)]).reshape(1,-1)
        values['velocity']=velocity_error/squared_scales('velocity')
        values['tracer']=tracer_error/squared_scales('tracer')
    # All remaining terms and their coefficients match the existing relative
    # MSE control exactly. This distinguishes robust-loss changes from the
    # specific choice of full-state versus step-sized field supervision.
    field=sum(values[name]for name in ('pressure','velocity','tracer'))/3
    nx,ny,nz=counts
    physical_tracer=predicted[:,:,4,:nz,:ny,:nx]*tracer_scale
    bounds=F.relu(-physical_tracer).square().mean()+F.relu(physical_tracer-1).square().mean()
    physical=.1*values['divergence']+.05*values['mass']+.02*values['energy']
    return (field+values['equation']+physical).mean()+.01*bounds
