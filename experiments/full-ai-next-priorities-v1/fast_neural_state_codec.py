"""Preserve V/T encoding exactly while omitting unused pressure-input arithmetic."""
import torch
from torch.nn import functional as F
from temporal_features_v2 import FIELD_NAMES


def packed_state(fields,counts,scales,device='cpu'):
    if set(fields)!=set(FIELD_NAMES):raise ValueError('Only current physical state fields may enter prediction')
    nx,ny,nz=counts
    shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))
    padded=[torch.zeros((1,1,nz+1,ny+1,nx+1),device=device,dtype=torch.float32)]
    for name,shape in zip(FIELD_NAMES[1:],shapes):
        value=torch.as_tensor(fields[name],device=device).reshape(1,1,*shape).float()
        scale=scales['tracer'if name=='tracer'else'velocity']
        value=value/scale
        padded.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(padded,dim=1)


def next_input(predicted,counts,scales):
    if predicted.shape[0]!=1:raise ValueError('Registered temporal inference uses batch size one')
    nx,ny,nz=counts
    shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))
    padded=[torch.zeros((1,1,nz+1,ny+1,nx+1),device=predicted.device,dtype=torch.float32)]
    for field,shape in enumerate(shapes,1):
        scale=scales['tracer'if field==4 else'velocity']
        # Keep physical float32 multiplication and re-encoding division as two
        # separate operations, matching the deployed codec's two roundings.
        value=predicted[:,-1:,field,:shape[0],:shape[1],:shape[2]]*scale
        value=value/scale
        padded.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(padded,dim=1)
