"""Compile the complete fixed-weight small-grid pressure graph, retaining its policy."""
import math
import time
import warnings
import torch
from torch import nn


class PressureGraph(nn.Module):
    def __init__(self,state,counts):
        super().__init__()
        import physical_residual as physical
        self.network=state.network
        self.owner=state.model.owner
        self.physical=physical
        self.counts=counts
        self.axes=physical.low_basis(counts,'cpu')
        self.steps=state.model.CORRECTION['steps']
        self.scales=physical.historical_physical.scales(state.normalization)
        self.output_scale=self.scales[state.normalization.get('output_scale','pressure')]

    def forward(self,base,initial,h,rho,dt):
        features=torch.tensor([math.log(n/16)for n in self.counts],dtype=torch.float32)
        if hasattr(self.network,'logits_for_shape'):
            logits=self.network.logits_for_shape(features,self.counts)
        else:
            logits=self.network.coarse_logits+self.network.shape_head(features)
        positive=max(self.counts)**2/2*torch.sigmoid(logits.to(torch.float64))
        gains=torch.cat((positive.new_zeros(1),positive)).reshape(4,4,4)
        gains=gains[:self.axes[0].shape[0],:self.axes[1].shape[0],:self.axes[2].shape[0]]
        pressure=base.clone()
        for stage in range(self.steps):
            residual=initial if stage==0 else initial-self.owner.full_operator(pressure-base,h,rho)
            coefficients=self.physical.coarse.restrict(residual*(rho/h).reshape(-1,1,1,1,1),self.axes)
            proposal=self.physical.coarse.prolong(coefficients*gains,self.axes)
            proposal=proposal-proposal[:,:,:1,:1,:1].clone()
            remaining=residual-self.owner.full_operator(proposal,h,rho)
            accept=(remaining.flatten(1).square().sum(1)<=residual.flatten(1).square().sum(1)).reshape(-1,1,1,1,1)
            proposal=torch.where(accept,proposal,torch.zeros_like(proposal))
            remaining=torch.where(accept,remaining,residual)
            values=(remaining*(rho/h).reshape(-1,1,1,1,1)/self.scales['laplacian_rhs']).to(torch.float32)
            raw=self.network(values)
            delta=self.owner.frozen.gauge_correction(raw,self.output_scale)
            nonzero=(remaining.flatten(1)!=0).any(dim=1).reshape(-1,1,1,1,1)
            pressure=pressure+proposal+torch.where(nonzero,delta,torch.zeros_like(delta))
        return pressure


class WholePrediction:
    def __init__(self,state):
        self.state=state
        self.cache={}
        self.versions=None
        self.normalization=None
        self.preparation=[]
        self.traces=0

    def __call__(self,sample):
        if torch.is_grad_enabled():
            raise ValueError('Frozen complete graph is inference only')
        state=self.state
        base=sample['base']
        state.model.owner.field_shape_contract(base)
        if base.shape[0]!=1:
            raise ValueError('Complete-graph research scope is batch1')
        if max(base.shape[2:])>16:
            return state.model.predict_pressure(state.network,sample,state.normalization)
        arguments=tuple(sample[name]for name in('base','native_input_residual','cell_size_m','rho','dt'))
        for value in arguments:
            if value.device.type!='cpu'or value.dtype!=torch.float64 or not bool(torch.isfinite(value).all()):
                raise ValueError('Original finite CPU float64 physical inputs required')
        if arguments[1].shape!=base.shape or any(value.shape!=(1,)or not bool((value>0).all())for value in arguments[2:]):
            raise ValueError('Colocated residual and positive physical scalars required')
        if float(base[0,0,0,0,0])!=0:
            raise ValueError('Cell-zero input pressure gauge required')
        import physical_residual as physical
        scales=physical.historical_physical.scales(state.normalization)
        normalization=(state.normalization.get('output_scale','pressure'),tuple(sorted(scales.items())))
        versions=tuple(parameter._version for parameter in state.network.parameters())
        if versions!=self.versions or normalization!=self.normalization:
            self.cache.clear()
            self.versions=versions
            self.normalization=normalization
        key=tuple((tuple(value.shape),tuple(value.stride()))for value in arguments)
        if key not in self.cache:
            if len(self.cache)>=128:
                raise ValueError('Complete-graph128-shape cache exhausted')
            tick=time.perf_counter()
            module=PressureGraph(state,tuple(base.shape[2:])).eval()
            with warnings.catch_warnings(record=True)as observed:
                warnings.simplefilter('always')
                compiled=torch.jit.trace(module,arguments,check_trace=True)
                compiled=torch.jit.freeze(compiled.eval())
            self.cache[key]=compiled
            self.traces+=1
            self.preparation.append(dict(shape=list(base.shape),seconds=time.perf_counter()-tick,warnings=len(observed),
                specialization='Shape/stride and frozen weights/normalization; data-dependent gates remain tensor operations'))
        result=self.cache[key](*arguments)
        if not bool(torch.isfinite(result).all())or float(result[0,0,0,0,0])!=0:
            raise ValueError('Finite cell-zero pressure output required')
        return result
