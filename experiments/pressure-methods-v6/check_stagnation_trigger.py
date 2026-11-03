"""Exercise the dormant dispatch branch using a deliberately weakened global stage."""
from unittest.mock import patch
import os
from v6_support import ROOT,HERE,BASE,write,sha
from solver_v2 import load,Policy
def run():
    output=BASE/'stagnation-dispatch-stress-01.json'
    if output.exists():raise ValueError('Fresh dispatch stress result required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    import numerics,fast_multilevel
    n=16;coordinate=t.arange(n,dtype=t.float64)+.5
    r=.001*(t.cos(t.pi*coordinate/n)[None,None,None,None,:]+t.cos(4*t.pi*coordinate/n)[None,None,None,:,None]+t.cos(7*t.pi*coordinate/n)[None,None,:,None,None])
    h=t.tensor([.0625],dtype=t.float64);rho=t.tensor([1000.],dtype=t.float64);dt=t.tensor([1/60],dtype=t.float64)
    sample=dict(base=t.zeros_like(r),rhs=r,native_input_residual=r,cell_size_m=h,rho=rho,dt=dt)
    policy=Policy(state,network,optimized=True,variant='stagnation')
    with t.inference_mode(),patch.object(fast_multilevel,'correction',lambda residual,h,rho:residual*(rho/h).reshape(1,1,1,1,1)/8):
        p,w=policy.solve(sample,'weak-global-stage');policy.commit(False)
    if w['neural_decisions'][0]['neural']or w['network_calls']==0 or not any(d['neural']and d['stagnating']for d in w['neural_decisions'][1:]):raise ValueError('Stagnation did not dispatch a real CNN correction')
    residual=r-numerics.fine_action(p,h,rho);actual=float(t.linalg.vector_norm(residual))<=max(1e-12,float(t.linalg.vector_norm(r))*1e-6)and float(dt[0]*residual.abs().max()/h[0]**3)<=1e-7
    if w['qualified']!=actual:raise ValueError('Stress result misreports actual returned pressure')
    write(output,dict(status='passed_dispatch_and_returned_field_check',work=w,source_sha256=sha(__file__),solver_sha256=sha(HERE/'solver_v2.py'),optimizer_updates=0,
      scope='Synthetic compatible Neumann RHS and deliberate diagonal-only global preconditioner, isolated in this process. Exercises dispatch; does not establish speed or accuracy benefit on naturally stalled fluid solves.'))
    print('Stagnation branch invoked actual CNN corrections; returned-field qualification remains truthful',flush=True)
if __name__=='__main__':run()
