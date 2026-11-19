"""Bounded variable coarse grids and a purely numerical two-level PCG control."""
from functools import lru_cache
import math
import torch
import coarse
from numerics import fine_action

@lru_cache(maxsize=32)
def geometry(shape,nodes,device):
    if nodes not in (2,3,4,6):raise ValueError('Declared coarse node count required')
    axes=[];mass=[];stiffness=[]
    for n in shape:
        count=min(nodes,n)
        centers=(torch.arange(count,dtype=torch.float64)+.5)*n/count-.5
        p=coarse.interpolate(n,centers)
        axes.append(p.T.contiguous().to(device));mass.append(p.T@p)
        difference=torch.diff(p,dim=0);stiffness.append(difference.T@difference)
    mz,my,mx=mass;kz,ky,kx=stiffness
    matrix=torch.kron(torch.kron(kz,my),mx)+torch.kron(torch.kron(mz,ky),mx)+torch.kron(torch.kron(mz,my),kx)
    factor=torch.linalg.cholesky(matrix[1:,1:]).to(device) if matrix.shape[0]>1 else None
    return tuple(axes),factor

def validate(residual):
    if (residual.dtype!=torch.float64 or residual.ndim!=5 or residual.shape[:2]!=(1,1)
            or min(residual.shape[2:])<1 or max(residual.shape[2:])>256
            or residual.numel()>256**3 or not bool(torch.isfinite(residual).all())):
        raise ValueError('Bounded finite f64 residual required')
    norm=float(torch.linalg.vector_norm(residual))
    dc=abs(float(residual.mean()))*math.sqrt(residual.numel())
    if dc>max(1e-12,norm*1e-8):raise ValueError('Incompatible sealed RHS; no mean subtraction')

def correction(residual,h,rho,nodes=4):
    validate(residual)
    if any(value.shape!=(1,) or value.dtype!=torch.float64 or value.device!=residual.device
           or not bool(torch.isfinite(value).all() and (value>0).all()) for value in (h,rho)):
        raise ValueError('Positive colocated f64 spacing and density required')
    axes,factor=geometry(tuple(residual.shape[2:]),nodes,str(residual.device))
    restricted=coarse.restrict(residual*(rho/h).reshape(-1,1,1,1,1),axes)
    flat=restricted.flatten(1);solved=torch.zeros_like(flat)
    if factor is not None:solved[:,1:]=torch.cholesky_solve(flat[:,1:].T,factor).T
    pressure=coarse.prolong(solved.reshape(restricted.shape),axes)
    return pressure-pressure[:,:,:1,:1,:1].clone()

@lru_cache(maxsize=16)
def diagonal(shape,device):
    result=torch.zeros((1,1,*shape),dtype=torch.float64,device=device)
    for index,n in enumerate(shape):
        if n==1:continue
        value=torch.full((n,),2.,dtype=torch.float64,device=device);value[0]=value[-1]=1.
        dimensions=[1]*5;dimensions[index+2]=n
        result+=value.reshape(dimensions)
    return result.clamp_min(1.)

def pcg(sample,nodes=4,max_iterations=256,tolerance=1e-6):
    """Additive Galerkin coarse + diagonal SPD preconditioner; no neural weights."""
    h=sample['cell_size_m'];rho=sample['rho'];base=sample['base']
    initial=sample['native_input_residual'];validate(initial)
    scaling=(rho/h).reshape(1,1,1,1,1)
    r=initial*scaling;x=torch.zeros_like(r);one=torch.ones_like(h)
    target=max(float(torch.linalg.vector_norm(sample['rhs']*scaling))*tolerance,
               1e-12*float(rho/h))
    def action(p):return fine_action(p,one,one)
    def precondition(value):return value/diagonal(tuple(value.shape[2:]),str(value.device))+correction(value,one,one,nodes)
    if float(torch.linalg.vector_norm(r))<=target:return base.clone(),0
    z=precondition(r);direction=z.clone();rz=torch.sum(r*z)
    for iteration in range(1,max_iterations+1):
        ad=action(direction);denominator=torch.sum(direction*ad)
        if not bool(torch.isfinite(denominator)) or float(denominator)<=0:
            raise ValueError('Nonpositive numerical CG curvature')
        alpha=rz/denominator;x=x+alpha*direction;r=r-alpha*ad
        if iteration%8==0 or float(torch.linalg.vector_norm(r))<=target:
            r=initial*scaling-action(x)
        if float(torch.linalg.vector_norm(r))<=target:break
        z=precondition(r);next_rz=torch.sum(r*z)
        direction=z+(next_rz/rz)*direction;rz=next_rz
    pressure=base+x;pressure=pressure-pressure[:,:,:1,:1,:1].clone()
    return pressure,iteration
