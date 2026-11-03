"""Matched multilevel policies with explicit local-model and recycling work."""
import time
from v6_support import ROOT,read
import torch
from recycling import RecycledGCR

def load(entry_path):
    import inference
    entry=read(entry_path);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    if entry['id'].startswith(('v5-','v6-')):wrapper=state.network
    else:
        import scalable
        wrapper,_=scalable.qualify(state)
    return state,wrapper

class Policy:
    def __init__(self,state,wrapper,*,method='gcr',neural=True,reuse=0,optimized=False,channels_last=False,retention="contribution",variant="all",local_execution="torch",adaptive_reuse=False):
        if method not in ['gcr','vcycle','spectral']:raise ValueError('Declared method required')
        if method!='gcr'and reuse:raise ValueError('Only the GCR method owns recycled directions')
        self.state=state;self.wrapper=wrapper;self.method=method;self.neural=neural;self.optimized=optimized;self.channels_last=channels_last
        self.recycler=RecycledGCR(reuse,strategy=retention)
        if channels_last:
            import copy
            self.wrapper=copy.deepcopy(wrapper).to(memory_format=torch.channels_last_3d).eval()
        self.tiny=state.entry['id'].startswith(('v5-','v6-'))
        if not self.tiny:raise ValueError('This study uses explicitly tiny local models')
        if variant not in ['all','first','first2','stagnation','coarse1']:raise ValueError('Declared neural placement required')
        if local_execution not in ['torch','native','streaming']:raise ValueError('Declared local execution required')
        self.variant=variant;self.local_execution=local_execution;self.adaptive_reuse=adaptive_reuse
        self.last_context=None;self.last_sketch=None;self.last_iterations=None;self.pending_metadata=None;self.native=None
        if local_execution in ['native','streaming']:
            from native_local import NativeLocal
            self.native=NativeLocal(wrapper,state.normalization,streaming=local_execution=='streaming')

    def commit(self,advanced):
        if self.method=='gcr':self.recycler.commit(advanced)
        if advanced and self.pending_metadata is not None:self.last_context,self.last_sketch,self.last_iterations=self.pending_metadata
        self.pending_metadata=None
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
        decisions=[];input_norms=[]
        def local(r):
            tick=time.perf_counter()
            import historical_physical
            scale=historical_physical.scales(self.state.normalization)['laplacian_rhs']
            hh=h;rr=rho;level=None
            if self.variant=='coarse1':
                level=multilevel.geometry(tuple(r.shape[2:]))
                if level.child is None:return torch.zeros_like(r)
                q=level.restrict(r*(rho/h).reshape(1,1,1,1,1))
                r=6*q/level.child.diagonal;hh=torch.ones_like(h);rr=torch.ones_like(rho)
            if self.native is not None:delta=self.native(r,hh,rr)
            else:
                values=((r*(rr/hh).reshape(1,1,1,1,1))/scale).to(torch.float32)
                if self.channels_last:values=values.contiguous(memory_format=torch.channels_last_3d)
                raw=self.wrapper(values);delta=raw.to(torch.float64)*scale;delta=delta-delta[:,:,:1,:1,:1].clone()
            if level is not None:delta=level.prolong(delta);delta=delta-delta[:,:,:1,:1,:1].clone()
            work['network_seconds']+=time.perf_counter()-tick;work['network_calls']+=1
            return delta
        def proposal(r):
            index=work['coarse_calls'];norm=float(torch.linalg.vector_norm(r))
            stagnating=bool(input_norms and norm/max(input_norms[-1],1e-300)>.25)
            use=self.neural and (self.variant in ['all','coarse1']or(self.variant=='first'and index==0)or(self.variant=='first2'and index<2)or(self.variant=='stagnation'and stagnating))
            input_norms.append(norm);decisions.append(dict(iteration=index+1,neural=use,stagnating=stagnating))
            tick=time.perf_counter();delta=coarse(r,h,rho);work['coarse_seconds']+=time.perf_counter()-tick;work['coarse_calls']+=1
            if use:delta=delta+local(r-action(delta))
            return delta
        adaptive=None;sketch=None
        if self.adaptive_reuse and self.method=='gcr':
            # A bounded deterministic subsample avoids retaining another full
            # field. Admission thresholds are development-only configuration.
            stride=[max(1,n//8)for n in initial.shape[2:]]
            sketch=initial[:,:,::stride[0],::stride[1],::stride[2]].flatten().clone()
            key=(context,tuple(initial.shape),float(h[0]),float(rho[0]))
            cosine=None
            if key==self.last_context and self.last_sketch is not None and sketch.shape==self.last_sketch.shape:
                cosine=float(torch.dot(sketch,self.last_sketch)/(torch.linalg.vector_norm(sketch)*torch.linalg.vector_norm(self.last_sketch)).clamp_min(1e-300))
            admitted=bool(cosine is not None and cosine>=.9 and self.last_iterations>=4 and initial.numel()>=64**3)
            adaptive=dict(admitted=admitted,sketch_cosine=cosine,previous_iterations=self.last_iterations,minimum_iterations=4,minimum_cosine=.9)
            if not admitted:self.recycler.basis=[]
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
        if adaptive is not None:
            self.pending_metadata=(key,sketch,work['coarse_calls'])
        work.update(neural_decisions=decisions,adaptive_reuse=adaptive)
        return base+delta,dict(**work,**extra,qualified=accepted(residual),relative_residual=float(torch.linalg.vector_norm(residual))/denominator,
            predicted_divergence=float(sample['dt'][0]*residual.abs().max()/h[0]**3))

