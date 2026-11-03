"""Compile the actual two-fine/three-global learned-tail pressure graph."""
import time
import warnings
import torch
from whole_pressure_trace import PressureGraph


class TailPressureGraph(PressureGraph):
    def forward(self,base,initial,h,rho,dt):
        import tail_physical
        pressure=super().forward(base,initial,h,rho,dt)
        sample=dict(base=base,native_input_residual=initial,cell_size_m=h,rho=rho,dt=dt)
        return tail_physical.append(self.network,sample,pressure,self.owner)


class TailWholePrediction:
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
            module=TailPressureGraph(state,tuple(base.shape[2:])).eval()
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
