"""Literal coarse Galerkin solves and globally computed sparse pressure anchors.

Both methods use the same sealed fine-grid equation. Sparse anchors require a
global forward transform and selected inverse; their entire cost must count.
Geometry caches contain no sample fields or pressure labels.
"""
from functools import lru_cache
import math
import torch


def interpolate(n, positions):
    positions = positions.to(torch.float64)
    if positions.numel() == 1:
        return torch.ones((n, 1), dtype=torch.float64)
    cells = torch.arange(n, dtype=torch.float64).clamp(float(positions[0]), float(positions[-1]))
    right = torch.searchsorted(positions, cells, right=True).clamp(1, positions.numel()-1)
    left = right - 1
    weight = (cells - positions[left]) / (positions[right] - positions[left])
    result = torch.zeros((n, positions.numel()), dtype=torch.float64)
    result[torch.arange(n), left] = 1-weight
    result[torch.arange(n), right] += weight
    return result


@lru_cache(maxsize=64)
def coarse_geometry(shape, device):
    axes, mass, stiffness = [], [], []
    for n in shape:
        count = min(6,n)
        centers = (torch.arange(count,dtype=torch.float64)+.5)*n/count-.5
        p = interpolate(n,centers)
        axes.append(p.T.contiguous().to(device))
        mass.append(p.T @ p)
        differences = torch.diff(p,dim=0)
        stiffness.append(differences.T @ differences)
    mz,my,mx = mass
    kz,ky,kx = stiffness
    matrix = (torch.kron(torch.kron(kz,my),mx) + torch.kron(torch.kron(mz,ky),mx)
              + torch.kron(torch.kron(mz,my),kx))
    factor = torch.linalg.cholesky(matrix[1:,1:]).to(device) if matrix.shape[0] > 1 else None
    return tuple(axes), factor


@lru_cache(maxsize=64)
def anchor_geometry(shape, device):
    axes, selected_axes, prolongation, eigenvalues, indices = [], [], [], [], []
    for n in shape:
        modes = torch.arange(n,dtype=torch.float64)
        cells = torch.arange(n,dtype=torch.float64)
        basis = torch.cos(math.pi/n*modes[:,None]*(cells[None]+.5))*math.sqrt(2/n)
        basis[0] /= math.sqrt(2)
        selected = torch.linspace(0,n-1,min(4,n)).round().to(torch.int64)
        axes.append(basis.to(device))
        selected_axes.append(basis[:,selected].contiguous().to(device))
        prolongation.append(interpolate(n,selected).T.contiguous().to(device))
        eigenvalues.append(4*torch.sin(math.pi*modes/(2*n)).square())
        indices.append(selected.to(device))
    z,y,x = eigenvalues
    values = z[:,None,None]+y[None,:,None]+x[None,None,:]
    values[0,0,0] = 1
    return tuple(axes), tuple(selected_axes), tuple(prolongation), values.to(device), tuple(indices)


def restrict(value, axes):
    z,y,x = axes
    result = torch.einsum('kz,bzyx->bkyx',z,value[:,0])
    result = torch.einsum('ly,bkyx->bklx',y,result)
    return torch.einsum('mx,bklx->bklm',x,result)


def prolong(value, axes):
    z,y,x = axes
    result = torch.einsum('mx,bklm->bklx',x,value)
    result = torch.einsum('ly,bklx->bkyx',y,result)
    return torch.einsum('kz,bkyx->bzyx',z,result)[:,None]


def validate(residual,h,rho):
    if (residual.ndim != 5 or residual.shape[1] != 1 or residual.dtype != torch.float64
            or not 1 <= residual.shape[0] <= 4 or min(residual.shape[2:]) < 1
            or max(residual.shape[2:]) > 128 or math.prod(residual.shape[2:]) > 2097152
            or not bool(torch.isfinite(residual).all())):
        raise ValueError('Finite bounded float64 fine residual required')
    for value in (h,rho):
        if (value.shape != (residual.shape[0],) or value.device != residual.device
                or value.dtype != torch.float64 or not bool(torch.isfinite(value).all() and (value > 0).all())):
            raise ValueError('Colocated positive float64 spacing/density required')
    norm = torch.linalg.vector_norm(residual.flatten(1),dim=1)
    constant = residual.flatten(1).mean(1).abs()*math.sqrt(residual[0].numel())
    if bool((constant > torch.maximum(norm*1e-8,norm.new_full(norm.shape,1e-12))).any()):
        raise ValueError('Incompatible sealed RHS; no mean subtraction allowed')


def numerical(residual,h,rho):
    validate(residual,h,rho)
    axes,factor = coarse_geometry(tuple(residual.shape[2:]),str(residual.device))
    rhs = restrict(residual*(rho/h).reshape(-1,1,1,1,1),axes)
    flat = rhs.flatten(1)
    solution = torch.zeros_like(flat)
    if factor is not None:
        solution[:,1:] = torch.cholesky_solve(flat[:,1:].T,factor).T
    correction = prolong(solution.reshape(rhs.shape),axes)
    return correction-correction[:,:,:1,:1,:1].clone()


def sparse(residual,h,rho):
    validate(residual,h,rho)
    axes,selected,interpolation,eigenvalues,_ = anchor_geometry(tuple(residual.shape[2:]),str(residual.device))
    coefficients = restrict(residual*(rho/h).reshape(-1,1,1,1,1),axes)/eigenvalues
    coefficients = coefficients.clone()
    coefficients[:,0,0,0] = 0
    anchors = prolong(coefficients,selected)[:,0]
    anchors = anchors-anchors[:,:1,:1,:1].clone()
    result = prolong(anchors,interpolation)
    return result-result[:,:,:1,:1,:1].clone()


def pin_neural_correction(delta):
    """The learned infill must not change exact anchor values, including gauge."""
    *_,indices = anchor_geometry(tuple(delta.shape[2:]),str(delta.device))
    z,y,x = indices
    mask = torch.ones_like(delta)
    mask[:,:,z[:,None,None],y[None,:,None],x[None,None,:]] = 0
    return delta*mask
