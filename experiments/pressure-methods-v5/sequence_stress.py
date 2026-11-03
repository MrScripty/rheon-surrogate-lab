"""Full-grid pressure-sequence stress: smooth changes, abrupt spectra, new geometry."""
import argparse,os,math,statistics
from pathlib import Path
from v5_support import ROOT,HERE,DOCS,read,write,sha
from solver_v3 import load,Policy
from spectrum import pattern
def run(args):
    if args.output.exists():raise ValueError('Fresh sequence result required')
    import torch,numerics
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    selected=read(DOCS/'model-selection.json')['selected']['arm'];state,wrapper=load(ROOT/f'runs/pressure-methods-v5/entries/{selected}.json')
    shapes=[[32,24,16],[32,24,16],[37,25,19]];bands=[[1,3],[8,12],[4,7]];rows=[]
    for capacity,strategy in [(0,'contribution'),(2,'contribution'),(4,'contribution'),(4,'tail')]:
        policy=Policy(state,wrapper,reuse=capacity,retention=strategy,optimized=True);previous=None
        with torch.inference_mode():
            for block in range(3):
                row=dict(seed=2026101990+block*10,counts_xyz=shapes[block],cells_per_metre=16 if block<2 else 8,
                    mode_band=bands[block],mode_count=6,pressure_rms_pa=1.)
                a,_=pattern(row,0);b,_=pattern(row,1);h=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
                for step in range(8):
                    reference=math.cos(step*.1)*a+math.sin(step*.1)*b
                    base=torch.zeros_like(reference)if previous is None or previous.shape!=reference.shape else previous
                    rhs=numerics.fine_action(reference,h,rho);initial=rhs-numerics.fine_action(base,h,rho)
                    sample=dict(base=base,rhs=rhs,native_input_residual=initial,reference=reference,cell_size_m=h,rho=rho,dt=torch.tensor([1/60],dtype=torch.float64))
                    pressure,work=policy.solve(sample,'same-sequence-with-geometry-change');r=rhs-numerics.fine_action(pressure,h,rho)
                    actual=float(torch.linalg.vector_norm(r))<=max(1e-12,float(torch.linalg.vector_norm(rhs))*1e-6)and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
                    if actual!=work['qualified']:raise ValueError('Sequence acceptance does not describe returned pressure')
                    policy.commit(actual)
                    error=pressure-reference;error-=error.mean()
                    rows.append(dict(capacity=capacity,retention=strategy,step=block*8+step,counts_xyz=shapes[block],event='geometry_reset'if block==2 and step==0 else'abrupt_spectral_change'if block==1 and step==0 else'smooth',
                        **work,pressure_relative_l2=float(torch.linalg.vector_norm(error))/float(torch.linalg.vector_norm(reference-reference.mean()))))
                    previous=pressure if actual else None
    write(args.output,dict(status='completed',rows=rows,entry=state.entry,source_sha256=sha(__file__),
        dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver_v3.py','recycling_v3.py','fast_multilevel.py','spectrum.py']},
        interpretation='Independent original-operator pressure equations, not advected fluid trajectories. No pressure reference enters a neural proposal.',optimizer_updates=0))
    print('Verified returned-field gates over96 full-grid sequence requests',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
