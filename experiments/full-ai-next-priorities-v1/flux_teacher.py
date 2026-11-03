"""Offline synthetic linear flux labels; never imported by candidate inference."""
import torch


def dimensionless_rhs(velocity):
    nz,ny,nx=(n-1 for n in velocity.shape[2:])
    result=velocity.new_zeros((velocity.shape[0],1,nz,ny,nx))
    result+=velocity[:,0:1,:nz,:ny,:nx]-velocity[:,0:1,:nz,:ny,1:nx+1]
    result+=velocity[:,1:2,:nz,:ny,:nx]-velocity[:,1:2,:nz,1:ny+1,:nx]
    result+=velocity[:,2:3,:nz,:ny,:nx]-velocity[:,2:3,1:nz+1,:ny,:nx]
    return result
