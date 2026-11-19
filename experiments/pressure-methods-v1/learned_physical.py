"""Learned low-frequency gains with a fine-equation contraction gate per pass."""
import math
from functools import lru_cache
from historical_physical import *
import historical_physical
import coarse


@lru_cache(maxsize=64)
def low_basis(shape,device):
    result = []
    for n in shape:
        modes = torch.arange(min(4,n),dtype=torch.float64)
        cells = torch.arange(n,dtype=torch.float64)
        axis = torch.cos(math.pi/n*modes[:,None]*(cells[None]+.5))*math.sqrt(2/n)
        axis[0] /= math.sqrt(2)
        result.append(axis.to(device))
    return tuple(result)


def predict_pressure(network,sample,normalization,owner,correction,intermediates=None):
    base = sample['base']
    initial = sample['native_input_residual']
    pressure = base.clone()
    axes = low_basis(tuple(base.shape[2:]),str(base.device))
    cap = max(base.shape[2:])**2/2
    positive = cap*torch.sigmoid(network.coarse_logits.to(torch.float64))
    gains = torch.cat((positive.new_zeros(1),positive)).reshape(4,4,4)
    gains = gains[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
    for stage in range(correction['steps']):
        residual = initial if stage == 0 else initial-owner.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
        coefficients = coarse.restrict(residual*(sample['rho']/sample['cell_size_m']).reshape(-1,1,1,1,1),axes)
        proposal = coarse.prolong(coefficients*gains,axes)
        proposal = proposal-proposal[:,:,:1,:1,:1].clone()
        remaining = residual-owner.full_operator(proposal,sample['cell_size_m'],sample['rho'])
        accept = (remaining.flatten(1).square().sum(1) <= residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
        proposal = torch.where(accept,proposal,torch.zeros_like(proposal))
        remaining = torch.where(accept,remaining,residual)
        pressure = historical_physical.predict_pressure(network,
            {**sample,'base':pressure+proposal,'native_input_residual':remaining},normalization,owner,{**correction,'steps':1})
        if intermediates is not None:
            intermediates.append(pressure)
    return pressure
