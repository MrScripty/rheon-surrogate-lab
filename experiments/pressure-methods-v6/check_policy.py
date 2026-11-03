"""Check selective dispatch and host-committed adaptive state independently."""
import argparse,os
from pathlib import Path
from v6_support import ROOT,HERE,BASE,write,sha
from solver import load,Policy
def run(args):
    if args.output.exists():raise ValueError('Fresh policy check required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    import numerics
    t.manual_seed(2026101913)
    def sample(shape,h=.0625,sign=1):
        p=t.randn((1,1,*shape),dtype=t.float64)*sign;p-=p[:,:,:1,:1,:1].clone()
        hh=t.tensor([h],dtype=t.float64);rho=t.tensor([1000.],dtype=t.float64);rhs=numerics.fine_action(p,hh,rho)
        return dict(base=t.zeros_like(p),reference=p,rhs=rhs,native_input_residual=rhs,cell_size_m=hh,rho=rho,dt=t.tensor([1/60],dtype=t.float64))
    def check(policy,s,context,commit=True):
        p,w=policy.solve(s,context);r=s['rhs']-numerics.fine_action(p,s['cell_size_m'],s['rho'])
        actual=float(t.linalg.vector_norm(r))<=max(1e-12,float(t.linalg.vector_norm(s['rhs']))*1e-6)and float(s['dt'][0]*r.abs().max()/s['cell_size_m'][0]**3)<=1e-7
        if actual!=w['qualified']:raise ValueError('Policy gate disagrees with independently recomputed returned-field equation')
        policy.commit(commit and actual);return w
    rows=[]
    with t.inference_mode():
        s=sample((17,13,9))
        for variant in ['all','first','first2','stagnation','coarse1']:
            policy=Policy(state,network,optimized=True,variant=variant,channels_last=True)
            w=check(policy,s,'dispatch')
            if variant=='first'and w['network_calls']>1:raise ValueError('First-iteration policy invoked extra neural corrections')
            if variant=='first2'and w['network_calls']>2:raise ValueError('Two-iteration policy invoked extra neural corrections')
            rows.append(dict(kind=variant,**w))
        for execution in ['native','streaming']:
            rows.append(dict(kind=execution,**check(Policy(state,network,optimized=True,local_execution=execution),s,'native')))
        s=sample((64,64,64));policy=Policy(state,network,neural=False,optimized=True,reuse=4,retention='coefficient',adaptive_reuse=True)
        w=check(policy,s,'sequence',commit=False)
        if policy.last_iterations is not None or policy.recycler.basis:raise ValueError('Rejected host step committed adaptive state')
        rows.append(dict(kind='host-rejected',**w))
        w=check(policy,s,'sequence');rows.append(dict(kind='cold-accepted',**w))
        expected=w['coarse_calls']>=4
        w=check(policy,s,'sequence');rows.append(dict(kind='repeat-accepted',**w))
        if w['adaptive_reuse']['admitted']!=expected:raise ValueError('Repeated-input adaptive admission disagrees with declared work threshold')
        reversed_sample={**s,'rhs':-s['rhs'],'native_input_residual':-s['native_input_residual'],'reference':-s['reference']}
        w=check(policy,reversed_sample,'sequence');rows.append(dict(kind='abrupt-reversal',**w))
        if w['adaptive_reuse']['admitted']:raise ValueError('Opposite residual admitted as a similar timestep')
        w=check(policy,s,'new-trajectory');rows.append(dict(kind='context-change',**w))
        if w['adaptive_reuse']['admitted']or w['cache_in']:raise ValueError('Changed trajectory reused old state')
    write(args.output,dict(status='passed',rows=rows,source_sha256=sha(__file__),solver_sha256=sha(HERE/'solver.py'),recycling_sha256=sha(HERE/'recycling.py'),optimizer_updates=0))
    print('Independent dispatch, returned-field, adaptive/commit and abrupt-change checks passed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'policy-qualification-01.json');run(p.parse_args())
