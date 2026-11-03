"""Flexible GCR: residual minimization with explicitly counted neural proposals.

Unlike ordinary PCG this does not assume a neural proposal is SPD. Restarted,
twice-orthogonalized directions support nonstationary or nonlinear proposals.
All acceptance decisions use a recomputed original fine-grid residual.
"""
import torch

def gcr(initial,action,precondition,accepted,*,max_iterations=8,restart=8):
    if not 1<=max_iterations<=64 or not 1<=restart<=8:raise ValueError('Bounded GCR work required')
    out=torch.zeros_like(initial);residual=initial.clone();directions=[];images=[];trace=[]
    norm=max(float(torch.linalg.vector_norm(initial)),1e-300);applications=0;breakdown=None
    for iteration in range(max_iterations):
        if accepted(residual):break
        if len(directions)==restart:directions.clear();images.clear()
        direction=precondition(residual);image=action(direction);applications+=1
        for _ in range(2):
            for prior,image_prior in zip(directions,images):
                beta=torch.sum(image*image_prior)
                direction=direction-beta*prior;image=image-beta*image_prior
        length=torch.linalg.vector_norm(image)
        if not bool(torch.isfinite(length))or float(length)<=1e-30*norm:
            breakdown='nonfinite_or_dependent_preconditioned_direction';break
        direction=direction/length;image=image/length
        alpha=torch.sum(residual*image);out=out+alpha*direction
        # Recompute instead of accepting an optimistically updated recurrence.
        residual=initial-action(out);applications+=1
        if not bool(torch.isfinite(residual).all()):breakdown='nonfinite_true_residual';break
        directions.append(direction);images.append(image)
        trace.append(dict(iteration=iteration+1,relative_input_residual=float(torch.linalg.vector_norm(residual))/norm))
    return out,dict(iterations=len(trace),operator_applications=applications,breakdown=breakdown,trace=trace)

def solve(wrapper,sample,normalization,branch,*,kind='learned',max_iterations=8):
    import numerics,scalable,multilevel
    from frozen_matrix import run_policy
    base=sample['base'];h=sample['cell_size_m'];rho=sample['rho'];initial=sample['native_input_residual']
    multilevel.validate(initial)
    denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
    limit=max(1e-12,denominator*1e-6);network_calls=0;coarse_calls=0
    def accepted(r):return float(torch.linalg.vector_norm(r))<=limit and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
    def action(p):return numerics.fine_action(p,h,rho)
    def precondition(r):
        nonlocal network_calls,coarse_calls
        if kind=='learned':
            zero=torch.zeros_like(r)
            pressure,work=run_policy(wrapper,dict(sample,base=zero,rhs=r,native_input_residual=r),normalization,branch,'learned',max_passes=1)
            network_calls+=work['network_calls'];coarse_calls+=work['coarse_calls']
            return pressure
        elif kind in ['multilevel','multilevel-neural']:
            out=multilevel.correction(r,h,rho);coarse_calls+=1
            if kind=='multilevel-neural':
                remaining=r-action(out);zero=torch.zeros_like(r)
                out=out+scalable.fine_step(wrapper,dict(sample,base=zero,native_input_residual=remaining),normalization,zero)
                network_calls+=1
            return out
        raise ValueError('Declared preconditioner required')
    delta,work=gcr(initial,action,precondition,accepted,max_iterations=max_iterations)
    r=initial-action(delta)
    return base+delta,dict(network_calls=network_calls,coarse_calls=coarse_calls,qualified=accepted(r),
        relative_residual=float(torch.linalg.vector_norm(r))/denominator,
        predicted_divergence=float(sample['dt'][0]*r.abs().max()/h[0]**3),**work)
