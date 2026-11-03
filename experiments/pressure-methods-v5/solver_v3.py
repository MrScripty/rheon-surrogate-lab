"""Matched multilevel policies with explicit local-model and recycling work."""
import time
from v5_support import ROOT,read
import torch
from recycling_v3 import RecycledGCR

def load(entry_path):
    import inference
    entry=read(entry_path);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    if entry['id'].startswith('v5-'):wrapper=state.network
    else:
        import scalable
        wrapper,_=scalable.qualify(state)
    return state,wrapper

class Policy:
    def __init__(self,state,wrapper,*,method='gcr',neural=True,reuse=0,optimized=False,channels_last=False,retention="contribution"):
        if method not in ['gcr','vcycle','spectral']:raise ValueError('Declared method required')
        if method!='gcr'and reuse:raise ValueError('Only the GCR method owns recycled directions')
        self.state=state;self.wrapper=wrapper;self.method=method;self.neural=neural;self.optimized=optimized;self.channels_last=channels_last
        self.recycler=RecycledGCR(reuse,strategy=retention)
        if channels_last:
            import copy
            self.wrapper=copy.deepcopy(wrapper).to(memory_format=torch.channels_last_3d).eval()
        self.tiny=state.entry['id'].startswith('v5-')
    def commit(self,advanced):
        if self.method=='gcr':self.recycler.commit(advanced)
    def solve(self,sample,context,*,max_passes=8):
        import numerics,multilevel
        base=sample['base'];initial=sample['native_input_residual'];h=sample['cell_size_m'];rho=sample['rho']
        multilevel.validate(initial)
        if self.optimized:
            import fast_multilevel
            coarse=fast_multilevel.correction
        else:coarse=multilevel.correction
        work=dict(network_calls=0,coarse_calls=0,operator_applications=0,operator_seconds=0.,coarse_seconds=0.,network_seconds=0.)
        denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300);limit=max(1e-12,denominator*1e-6)
        def action(value):
            tick=time.perf_counter();out=numerics.fine_action(value,h,rho);work['operator_seconds']+=time.perf_counter()-tick;work['operator_applications']+=1;return out
        def accepted(r):return float(torch.linalg.vector_norm(r))<=limit and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
        def returned_residual(delta):return sample['rhs']-action(base+delta)
        initial=sample['rhs']-action(base)
        def local(r):
            tick=time.perf_counter();zero=torch.zeros_like(r)
            if self.tiny:
                # Both data conditioning and deployment call the captured single
                # local correction. Pressure labels never enter this call.
                if self.channels_last:
                    import historical_physical
                    values=historical_physical.build_inputs(r,zero,h,rho,sample['dt'],self.state.normalization,self.state.model.owner)
                    raw=self.wrapper(values.contiguous(memory_format=torch.channels_last_3d))
                    delta=raw.to(torch.float64)*historical_physical.scales(self.state.normalization)['laplacian_rhs'];delta=delta-delta[:,:,:1,:1,:1].clone()
                else:delta=self.state.model.predict_pressure(self.wrapper,dict(sample,base=zero,native_input_residual=r),self.state.normalization)
            else:
                import scalable
                if self.channels_last:
                    class Layout:
                        def __call__(_,v):return self.wrapper(v.contiguous(memory_format=torch.channels_last_3d))
                    wrapper=Layout()
                else:wrapper=self.wrapper
                delta=scalable.fine_step(wrapper,dict(sample,base=zero,native_input_residual=r),self.state.normalization,zero)
            work['network_seconds']+=time.perf_counter()-tick;work['network_calls']+=1;return delta
        def proposal(r):
            tick=time.perf_counter();delta=coarse(r,h,rho);work['coarse_seconds']+=time.perf_counter()-tick;work['coarse_calls']+=1
            if self.neural:delta=delta+local(r-action(delta))
            return delta
        if self.method=='spectral':
            tick=time.perf_counter();delta=numerics.correction(initial,h,rho,None);work['coarse_seconds']=time.perf_counter()-tick;work['coarse_calls']=1
            extra=dict(iterations=1,cache_in=0,cache_bytes=0,breakdown=None)
        elif self.method=='gcr':
            key=(context,tuple(initial.shape),float(h[0]),float(rho[0]))
            delta,extra=self.recycler.solve(initial,action,proposal,accepted,key,max_iterations=max_passes,residual_of=returned_residual)
            extra=dict(extra);extra['gcr_operator_applications']=extra.pop('operator_applications')
        else:
            delta=torch.zeros_like(base);iterations=0
            for _ in range(max_passes):
                r=returned_residual(delta)
                if accepted(r):break
                tick=time.perf_counter();delta=delta+coarse(r,h,rho);work['coarse_seconds']+=time.perf_counter()-tick;work['coarse_calls']+=1;iterations+=1
                r=returned_residual(delta)
                if accepted(r):break
                if self.neural:delta=delta+local(r)
            extra=dict(iterations=iterations,cache_in=0,cache_bytes=0,breakdown=None)
        residual=returned_residual(delta)
        return base+delta,dict(**work,**extra,qualified=accepted(residual),relative_residual=float(torch.linalg.vector_norm(residual))/denominator,
            predicted_divergence=float(sample['dt'][0]*residual.abs().max()/h[0]**3))

