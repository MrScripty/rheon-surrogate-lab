"""Frozen-weight C107 execution experiments; no numerical pressure inverse."""
import copy
import math
import torch

class CachedPrediction:
    def __init__(self, state, *, buffers=False, channels_last=False):
        self.state = state
        self.network = copy.deepcopy(state.network).eval() if channels_last else state.network
        if channels_last:
            self.network.to(memory_format=torch.channels_last_3d)
        self.buffers = buffers
        self.channels_last = channels_last
        self.cache = {}
        self.scratch = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, sample):
        import physical_residual as physical
        state = self.state
        network = self.network
        base = sample['base']
        key = (tuple(base.shape), str(base.device), str(base.dtype),
               tuple(p._version for p in network.parameters()),
               tuple(float(sample[n][0]) for n in ('cell_size_m','rho','dt')))
        if key not in self.cache:
            if len(self.cache) >= 64:
                self.cache.clear()
                self.scratch.clear()
            axes = physical.low_basis(tuple(base.shape[2:]), str(base.device))
            features = torch.tensor([math.log(n/16) for n in base.shape[2:]],dtype=torch.float32,device=base.device)
            positive = (max(base.shape[2:])**2/2)*torch.sigmoid((network.coarse_logits+network.shape_head(features)).to(torch.float64))
            gains = torch.cat((positive.new_zeros(1),positive)).reshape(4,4,4)
            gains = gains[:axes[0].shape[0],:axes[1].shape[0],:axes[2].shape[0]]
            self.cache[key] = (axes,gains)
            self.misses += 1
        else:
            self.hits += 1
        axes,gains = self.cache[key]
        pressure = base.clone()
        initial = sample['native_input_residual']
        for stage in range(state.model.CORRECTION['steps']):
            residual = initial if stage==0 else initial-state.model.owner.full_operator(pressure-base,sample['cell_size_m'],sample['rho'])
            ratio = (sample['rho']/sample['cell_size_m']).reshape(-1,1,1,1,1)
            if self.buffers:
                if key not in self.scratch:
                    self.scratch[key] = torch.empty_like(residual)
                scaled = torch.mul(residual,ratio,out=self.scratch[key])
            else:
                scaled = residual*ratio
            coefficients = physical.coarse.restrict(scaled,axes)
            proposal = physical.coarse.prolong(coefficients*gains,axes)
            proposal = proposal-proposal[:,:,:1,:1,:1].clone()
            remaining = residual-state.model.owner.full_operator(proposal,sample['cell_size_m'],sample['rho'])
            accept = (remaining.flatten(1).square().sum(1)<=residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
            proposal = torch.where(accept,proposal,torch.zeros_like(proposal))
            remaining = torch.where(accept,remaining,residual)
            pressure = physical.historical_physical.predict_pressure(network,
                {**sample,'base':pressure+proposal,'native_input_residual':remaining},state.normalization,
                state.model.owner,{**state.model.CORRECTION,'steps':1})
        return pressure
