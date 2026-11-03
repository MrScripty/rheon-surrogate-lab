"""Signed spatial inputs, at most 32 captured complete pressure proposals.

No numerical hierarchy, optimizer or simulator is invoked by this diagnostic.
"""
import os
import math
import audit
from learned_iteration import equation_quality, stopping_policy

def probe(task):
    torch=audit.TORCH;counts=task['counts_xyz'];shape=counts[::-1]
    pressure=torch.zeros((1,1,*shape),dtype=torch.float64);laplacian=torch.zeros_like(pressure)
    for frequencies,weight in task['components_xyz']:
        factors=[torch.cos(torch.pi*k*(torch.arange(n,dtype=torch.float64)+.5)/n) for n,k in zip(shape,frequencies[::-1])]
        mode=torch.einsum('z,y,x->zyx',*factors)[None,None]
        eigenvalue=math.fsum(2-2*math.cos(math.pi*k/n) for n,k in zip(counts,frequencies))
        pressure+=weight*mode;laplacian+=(weight*eigenvalue)*mode
    pressure-=pressure[:,:,:1,:1,:1].clone()
    gain=task['amplitude_pa']/float(pressure.square().mean().sqrt());pressure*=gain;laplacian*=gain
    sample={'base':torch.zeros_like(pressure),'reference':pressure,
        'cell_size_m':torch.tensor([task['cell_size_m']],dtype=torch.float64),
        'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}
    sample.update(rhs=laplacian*(task['cell_size_m']/1000.))
    sample['native_input_residual']=sample['rhs'].clone()
    ideal=audit.ratio(pressure,sample)
    if not ideal<1e-11:raise ValueError('independent eigenvalue oracle disagrees with implemented operator')
    policy=stopping_policy({'schema':'rheon-learned-stopping-policy-v1','relative_residual':1e-6,'absolute_residual':1e-12,'predicted_divergence':1e-7})
    with torch.no_grad():
        trace=[];prediction=sample['base']
        for i in range(32):
            residual=sample['rhs']-audit.MODEL.full_operator(prediction,sample['cell_size_m'],sample['rho'])
            prediction=audit.MODEL.predict_pressure(audit.NETWORK,{**sample,'base':prediction,'native_input_residual':residual},audit.NORM)
            quality=equation_quality(audit.MODEL,prediction,sample,sample['rhs'],policy)
            trace.append(dict(proposals=i+1,**quality))
            if quality['equation_passed']:break
    return {**task,'trace':trace,'ideal_f64_relative_residual':ideal,
        'first_relative_residual':trace[0]['relative_residual'],'final_relative_residual':audit.ratio(prediction,sample),
        'network_evaluations':trace[-1]['proposals'],
        'fine_network_evaluations':trace[-1]['proposals']*audit.MODEL.ARCHITECTURE['correction']['steps'],
        'work_unit':'captured pressure proposals','equation_passed':trace[-1]['equation_passed'],
        'affinity':sorted(os.sched_getaffinity(0)),'cuda_initialized':torch.cuda.is_initialized()}

