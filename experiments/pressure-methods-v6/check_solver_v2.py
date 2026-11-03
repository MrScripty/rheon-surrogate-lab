"""Show cost instrumentation preserves the existing accepted pressure arithmetic."""
import argparse,os
from pathlib import Path
from v6_support import ROOT,HERE,BASE,write,sha
from solver import load,Policy as Original
from solver_v2 import Policy as Instrumented
def run(output):
    if output.exists():raise ValueError('Fresh parity result required')
    os.sched_setaffinity(0,{22});state,network=load(ROOT/'runs/pressure-methods-v5/entries/v5-spectrum.json');t=state.torch;t.set_num_threads(1)
    import numerics
    t.manual_seed(2026101915);rows=[]
    def sample(shape):
        ref=t.randn((1,1,*shape),dtype=t.float64);ref-=ref[:,:,:1,:1,:1].clone()
        h=t.tensor([.0625],dtype=t.float64);rho=t.tensor([1000.],dtype=t.float64);rhs=numerics.fine_action(ref,h,rho)
        return dict(base=t.zeros_like(ref),reference=ref,rhs=rhs,native_input_residual=rhs,cell_size_m=h,rho=rho,dt=t.tensor([1/60],dtype=t.float64))
    with t.inference_mode():
        for shape in [(17,13,9),(64,64,64)]:
            s=sample(shape)
            for variant in ['all','first','first2','stagnation','coarse1']:
                a=Original(state,network,optimized=True,variant=variant,channels_last=True);b=Instrumented(state,network,optimized=True,variant=variant,channels_last=True)
                p,w=a.solve(s,'parity');q,v=b.solve(s,'parity');a.commit(w['qualified']);b.commit(v['qualified'])
                if not t.equal(p,q)or any(w[k]!=v[k]for k in ['qualified','network_calls','coarse_calls','iterations']):raise ValueError('Instrumentation changed full-policy arithmetic')
                rows.append(dict(shape=shape,variant=variant,bitwise_pressure_equal=True,qualified=v['qualified']))
        s=sample((17,13,9))
        for strategy in ['contribution','coefficient']:
            a=Original(state,network,optimized=True,reuse=4,retention=strategy);b=Instrumented(state,network,optimized=True,reuse=4,retention=strategy)
            for i in range(4):
                if i==2:s=sample((17,13,9))
                p,w=a.solve(s,'cached-parity');q,v=b.solve(s,'cached-parity');a.commit(w['qualified']);b.commit(v['qualified'])
                if not t.equal(p,q):raise ValueError('Instrumented cached policy changed pressure')
                rows.append(dict(strategy=strategy,step=i,bitwise_pressure_equal=True,qualified=v['qualified']))
    write(output,dict(status='passed',rows=rows,source_sha256=sha(__file__),
        dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver.py','solver_v2.py','recycling.py','recycling_v2.py','adaptive.py']},optimizer_updates=0))
    print('Instrumented solver preserves full-policy pressure bitwise on all18 cases',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'instrumented-solver-parity-02.json');run(p.parse_args().output)
