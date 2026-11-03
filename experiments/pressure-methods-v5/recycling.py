"""Bounded inter-request GCR recycling; commit only after an accepted host step."""
import math
import torch

class RecycledGCR:
    def __init__(self,capacity=4):
        if capacity not in [0,2,4]:raise ValueError('Declared capacity 0, 2 or 4 required')
        self.capacity=capacity;self.context=None;self.basis=[];self.pending=None
    def clear(self):self.context=None;self.basis=[];self.pending=None
    def commit(self,advanced):
        if advanced and self.pending is not None:self.basis=self.pending
        self.pending=None
    def solve(self,initial,action,precondition,accepted,context,*,max_iterations=8):
        if not 1<=max_iterations<=8 or initial.dtype!=torch.float64 or initial.device.type!='cpu' or not bool(torch.isfinite(initial).all()):
            raise ValueError('Bounded finite CPU float64 equation required')
        if self.pending is not None:raise ValueError('Commit or reject the previous host step before solving another')
        changed=context!=self.context
        if changed:self.basis=[];self.context=context
        if any(d.shape!=initial.shape or q.shape!=initial.shape for d,q in self.basis):raise ValueError('Context omitted changed field shape')
        cached=list(self.basis);out=torch.zeros_like(initial);residual=initial.clone();directions=[];images=[];trace=[];applications=0;breakdown=None
        work=dict(cache_in=len(cached),reuse_capacity=self.capacity,context_reset=changed,recycle_projection_rejected=False,recycled_projection_sweeps=0)
        if accepted(residual):
            self.pending=cached
            return out,dict(**work,iterations=0,operator_applications=0,breakdown=None,trace=[],cache_bytes=sum(d.numel()*d.element_size()+q.numel()*q.element_size()for d,q in cached))
        norm=float(torch.linalg.vector_norm(initial))
        if cached:
            for _ in range(2):
                for direction,image in cached:
                    alpha=torch.sum(residual*image);out=out+alpha*direction;residual=residual-alpha*image
            residual=initial-action(out);applications+=1;work['recycled_projection_sweeps']=2
            if not bool(torch.isfinite(residual).all())or float(torch.linalg.vector_norm(residual))>norm*(1+1e-12):
                out=torch.zeros_like(initial);residual=initial.clone();cached=[];work['recycle_projection_rejected']=True
        for iteration in range(max_iterations):
            if accepted(residual):break
            direction=precondition(residual);image=action(direction);applications+=1
            for _ in range(2):
                for prior,prior_image in [*cached,*zip(directions,images)]:
                    beta=torch.sum(image*prior_image);direction=direction-beta*prior;image=image-beta*prior_image
            length=torch.linalg.vector_norm(image)
            if not bool(torch.isfinite(length))or float(length)<=1e-30*max(norm,1e-300):
                breakdown='nonfinite_or_dependent_preconditioned_direction';break
            direction=direction/length;image=image/length;alpha=torch.sum(residual*image);out=out+alpha*direction
            residual=initial-action(out);applications+=1
            if not bool(torch.isfinite(residual).all()):breakdown='nonfinite_true_residual';break
            directions.append(direction);images.append(image)
            trace.append(dict(iteration=iteration+1,relative_input_residual=float(torch.linalg.vector_norm(residual))/max(norm,1e-300)))
        qualified=accepted(residual)
        if qualified:
            all_pairs=[*cached,*zip(directions,images)]
            self.pending=[(d.detach().clone(),q.detach().clone())for d,q in all_pairs[-self.capacity:]]if self.capacity else[]
        else:self.pending=[]
        return out,dict(**work,iterations=len(trace),operator_applications=applications,breakdown=breakdown,trace=trace,
            cache_bytes=sum(d.numel()*d.element_size()+q.numel()*q.element_size()for d,q in self.pending))

