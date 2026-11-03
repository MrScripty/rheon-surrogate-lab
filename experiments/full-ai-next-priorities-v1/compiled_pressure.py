"""Frozen CPU fine-network traces; retain the checkpoint's learned global branch."""
import math
import time
import warnings
import torch


def fine_name(name):return name!='coarse_logits'and not name.startswith('shape_head.')


class FineTraces:
    def __init__(self,network):
        self.network=network;self.cache={};self.versions=None;self.traces=0;self.calls=0;self.preparation=[]

    def __getattr__(self,name):return getattr(self.network,name)

    def __call__(self,values):
        if torch.is_grad_enabled():raise ValueError('Frozen traces are inference-only')
        if values.ndim!=5 or values.shape[1]!=1 or not 1<=values.shape[0]<=4 or values.dtype!=torch.float32 or values.device.type!='cpu':
            raise ValueError('Original CPU float32 N,1,Z,Y,X fine-input contract required')
        if any(n<1 for n in values.shape[2:])or math.prod(values.shape[2:])>2097152 or not bool(torch.isfinite(values).all()):
            raise ValueError('Finite fine input inside the original cell budget required')
        versions=tuple(p._version for name,p in self.network.named_parameters()if fine_name(name))
        if versions!=self.versions:self.cache.clear();self.versions=versions
        key=(tuple(values.shape),tuple(values.stride()),str(values.dtype))
        self.calls+=1
        if key not in self.cache:
            if len(self.cache)>=128:raise ValueError('Bounded128-shape trace cache exhausted')
            started=time.perf_counter()
            with warnings.catch_warnings(record=True)as observed:
                warnings.simplefilter('always')
                traced=torch.jit.trace(self.network,values,check_trace=True)
                traced=torch.jit.freeze(traced.eval())
            self.cache[key]=traced;self.traces+=1
            self.preparation.append(dict(shape=list(values.shape),seconds=time.perf_counter()-started,warnings=len(observed),
                note='Input validation remains outside the trace; trace branches specialize to the cached shape'))
        return self.cache[key](values)


class CompiledPrediction:
    def __init__(self,state):self.state=state;self.fine=FineTraces(state.network)

    def __call__(self,sample):
        return self.state.model.predict_pressure(self.fine,sample,self.state.normalization)
