"""One additional learned global block after the existing two fine corrections."""
import math
import torch


def append(state,sample,pressure):
    import physical_residual as physical
    network=state.network
    counts=tuple(pressure.shape[2:])
    if max(counts)>16:
        return pressure
    axes=physical.low_basis(counts,str(pressure.device))
    features=torch.tensor([math.log(n/16)for n in counts],dtype=torch.float32,device=pressure.device)
    logits=network.logits_for_shape(features,counts)
    positive=max(counts)**2/2*torch.sigmoid(logits.to(torch.float64))
    gains=torch.cat((positive.new_zeros(1),positive)).reshape(4,4,4)
    gains=gains[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
    return apply_gains(state,sample,pressure,axes,gains)


def apply_gains(state,sample,pressure,axes,gains):
    import physical_residual as physical
    residual=sample['native_input_residual']-state.model.owner.full_operator(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
    coefficients=physical.coarse.restrict(residual*(sample['rho']/sample['cell_size_m']).reshape(-1,1,1,1,1),axes)
    proposal=physical.coarse.prolong(coefficients*gains,axes)
    proposal=proposal-proposal[:,:,:1,:1,:1].clone()
    remaining=residual-state.model.owner.full_operator(proposal,sample['cell_size_m'],sample['rho'])
    accept=(remaining.flatten(1).square().sum(1)<=residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
    return pressure+torch.where(accept,proposal,torch.zeros_like(proposal))
