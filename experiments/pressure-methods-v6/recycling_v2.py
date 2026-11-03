"""Bounded inter-request GCR recycling; commit only after an accepted host step."""
import math,time
import torch

class RecycledGCR:
    def __init__(self,capacity=4,strategy="contribution"):
        if strategy not in ["tail","contribution","coefficient"]:raise ValueError("Declared retention strategy required")
        self.strategy=strategy
        if capacity not in [0,2,4]:raise ValueError('Declared capacity 0, 2 or 4 required')
        self.capacity=capacity;self.context=None;self.basis=[];self.pending=None
    def clear(self):self.context=None;self.basis=[];self.pending=None
    def commit(self,advanced):
        if advanced and self.pending is not None:self.basis=self.pending
        self.pending=None
    def solve(self,initial,action,precondition,accepted,context,*,max_iterations=8,residual_of=None):
        if not 1<=max_iterations<=8 or initial.dtype!=torch.float64 or initial.device.type!='cpu' or not bool(torch.isfinite(initial).all()):
            raise ValueError('Bounded finite CPU float64 equation required')
        if self.pending is not None:raise ValueError('Commit or reject the previous host step before solving another')
        changed=context!=self.context
        if changed:self.basis=[];self.context=context
        if any(d.shape!=initial.shape or q.shape!=initial.shape for d,q in self.basis):raise ValueError('Context omitted changed field shape')
        cached=list(self.basis);cached_coefficients=[0. for _ in cached];new_coefficients=[];out=torch.zeros_like(initial);residual=initial.clone();directions=[];images=[];trace=[];applications=0;breakdown=None
        work=dict(cache_in=len(cached),reuse_capacity=self.capacity,context_reset=changed,recycle_projection_rejected=False,recycled_projection_sweeps=0)
        if accepted(residual):
            self.pending=cached
            return out,dict(**work,iterations=0,operator_applications=0,breakdown=None,trace=[],cache_bytes=sum(d.numel()*d.element_size()+q.numel()*q.element_size()for d,q in cached))
        norm=float(torch.linalg.vector_norm(initial))
        projection_seconds=0.;cached_orthogonalization_seconds=0.;retention_seconds=0.
        if cached:
            projection_tick=time.perf_counter()
            for _ in range(2):
                for k,(direction,image)in enumerate(cached):
                    alpha=torch.sum(residual*image);cached_coefficients[k]+=float(alpha);out=out+alpha*direction;residual=residual-alpha*image
            residual=(initial-action(out))if residual_of is None else residual_of(out);applications+=1;work['recycled_projection_sweeps']=2
            if not bool(torch.isfinite(residual).all())or float(torch.linalg.vector_norm(residual))>norm*(1+1e-12):
                out=torch.zeros_like(initial);residual=initial.clone();cached=[];cached_coefficients=[];work['recycle_projection_rejected']=True
            projection_seconds=time.perf_counter()-projection_tick
        for iteration in range(max_iterations):
            if accepted(residual):break
            direction=precondition(residual);image=action(direction);applications+=1
            for _ in range(2):
                cache_tick=time.perf_counter()
                for prior,prior_image in cached:
                    beta=torch.sum(image*prior_image);direction=direction-beta*prior;image=image-beta*prior_image
                cached_orthogonalization_seconds+=time.perf_counter()-cache_tick
                for prior,prior_image in zip(directions,images):
                    beta=torch.sum(image*prior_image);direction=direction-beta*prior;image=image-beta*prior_image
            length=torch.linalg.vector_norm(image)
            if not bool(torch.isfinite(length))or float(length)<=1e-30*max(norm,1e-300):
                breakdown='nonfinite_or_dependent_preconditioned_direction';break
            direction=direction/length;image=image/length;alpha=torch.sum(residual*image);out=out+alpha*direction
            residual=(initial-action(out))if residual_of is None else residual_of(out);applications+=1
            if not bool(torch.isfinite(residual).all()):breakdown='nonfinite_true_residual';break
            directions.append(direction);images.append(image);new_coefficients.append(float(alpha))
            trace.append(dict(iteration=iteration+1,relative_input_residual=float(torch.linalg.vector_norm(residual))/max(norm,1e-300)))
        qualified=accepted(residual)
        retention_tick=time.perf_counter()
        if qualified:
            all_pairs=[*cached,*zip(directions,images)]
            if self.capacity and self.strategy in ['contribution','coefficient']:
                coefficients=cached_coefficients+new_coefficients
                scores=([abs(a)for a in coefficients]if self.strategy=='coefficient'else[abs(a)*float(torch.linalg.vector_norm(d-d.mean()))for a,(d,q)in zip(coefficients,all_pairs)])
                selected=sorted(range(len(all_pairs)),key=lambda i:(scores[i],i),reverse=True)[:self.capacity]
                retained=[all_pairs[i]for i in selected]
                work['retained_contribution_scores']=[scores[i]for i in selected];work['retention_score_kind']='equation-image-coefficient'if self.strategy=='coefficient'else'pressure-contribution'
            else:retained=all_pairs[-self.capacity:]if self.capacity else[]
            self.pending=[(d.detach(),q.detach())for d,q in retained] # Every solver update is out of place; published basis tensors are immutable.
        else:self.pending=[]
        retention_seconds=time.perf_counter()-retention_tick
        return out,dict(**work,recycle_projection_seconds=projection_seconds,cached_orthogonalization_seconds=cached_orthogonalization_seconds,retention_seconds=retention_seconds,iterations=len(trace),operator_applications=applications,breakdown=breakdown,trace=trace,
            cache_bytes=sum(d.numel()*d.element_size()+q.numel()*q.element_size()for d,q in self.pending))

