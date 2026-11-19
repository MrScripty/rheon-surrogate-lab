"""Generate velocity increments from learned edge potentials and a fixed readout.

This is a representation of neural outputs, not a projection of an independently
predicted velocity. No pressure solver, inverse or numerical correction is
called. Pressure/tracer heads and the shared spatial encoder remain unchanged.
"""
from functools import lru_cache
import torch
from torch.nn import functional as F
from bundle_model_step_scaled import BundleNet as DirectBundleNet
from temporal_features_v2 import geometry


@lru_cache(maxsize=64)
def edge_masks(counts,device):
    nx,ny,nz=counts
    with torch.inference_mode(False):
        mask=torch.zeros((3,nz+1,ny+1,nx+1),dtype=torch.float64,device=device)
        # Tangential potential vanishes on each wall. The normal component is
        # cell-centred along its own axis and face-centred on the other two.
        mask[0,1:nz,1:ny,:nx]=1
        mask[1,1:nz,:ny,1:nx]=1
        mask[2,:nz,1:ny,1:nx]=1
    return mask


def velocity_curl(potential,counts):
    nx,ny,nz=counts
    if potential.shape[2:]!=(3,nz+1,ny+1,nx+1):raise ValueError('Complete edge-potential lattice required')
    fields=potential*edge_masks(tuple(counts),str(potential.device)).to(potential.dtype)[None,None]
    ax,ay,az=fields.unbind(2)
    vx=(az[:,:,:nz,1:ny+1,:nx+1]-az[:,:,:nz,:ny,:nx+1]
        -ay[:,:,1:nz+1,:ny,:nx+1]+ay[:,:,:nz,:ny,:nx+1])
    vy=(ax[:,:,1:nz+1,:ny+1,:nx]-ax[:,:,:nz,:ny+1,:nx]
        -az[:,:,:nz,:ny+1,1:nx+1]+az[:,:,:nz,:ny+1,:nx])
    vz=(ay[:,:,:nz+1,:ny,1:nx+1]-ay[:,:,:nz+1,:ny,:nx]
        -ax[:,:,:nz+1,1:ny+1,:nx]+ax[:,:,:nz+1,:ny,:nx])
    return vx,vy,vz


class BundleNet(DirectBundleNet):
    def reconstruct(self,raw,current,counts,physical_horizons):
        nx,ny,nz=counts
        pressure=raw[:,:,0]-raw[:,:,0,:1,:1,:1]
        _,mask=geometry(tuple(counts),str(current.device))
        gains={name:raw.new_tensor([self.increment_scales[str(k)][name]for k in physical_horizons]).reshape(1,-1,1,1,1)
               for name in ('velocity','tracer')}
        # A common domain-length factor conditions the potential readout and
        # preserves the discrete divergence identity on rectangular grids.
        delta=velocity_curl(raw[:,:,1:4].double(),counts)
        shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))
        fields=[pressure]
        for component,(value,shape)in enumerate(zip(delta,shapes),1):
            initial=current[:,None,component,:shape[0],:shape[1],:shape[2]].double()
            velocity=(initial+gains['velocity'].double()*min(counts)*value).float()
            velocity=F.pad(velocity,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]))
            fields.append(velocity*mask[component])
        fields.append(current[:,None,4]+raw[:,:,4]*gains['tracer'])
        return torch.stack(fields,dim=2)
