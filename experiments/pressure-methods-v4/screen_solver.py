"""Frozen mechanism comparison, retaining intermediate convergence and actual cost."""
import argparse
import os
import time
from pathlib import Path
from v4_support import ROOT, BASE, load, native, read, write, sha, summary, pressure_error

def run(args):
    if args.output.exists():raise ValueError('Fresh mechanism screen required')
    os.sched_setaffinity(0,{23})
    state,wrapper,parity=load()
    import torch,cases,numerics,multilevel
    from frozen_matrix import run_policy
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    population=cases.screen(cases.load(state))+native(state)
    methods=['learned','vcycle-neural','vcycle-only','spectral']
    if args.screen:population=[r for r in population if r[0]['frame']==0 or r[0]['population']=='consumed128']
    rows=[];setup=[];started=time.perf_counter()
    with torch.inference_mode():
        for identity,sample in population:
            tick=time.perf_counter();hierarchy=multilevel.geometry(tuple(sample['base'].shape[2:]));setup_s=time.perf_counter()-tick
            setup.append(dict(**identity,seconds=setup_s,levels=hierarchy.dimensions()))
            for name in methods:
                tick=time.perf_counter()
                if name=='learned':pressure,work=run_policy(wrapper,sample,state.normalization,branch,'learned',max_passes=8)
                elif name=='spectral':
                    pressure=sample['base']+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],None)
                    r=sample['native_input_residual']-numerics.fine_action(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
                    denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                    divergence=float(sample['dt'][0]*r.abs().max()/sample['cell_size_m'][0]**3)
                    work=dict(network_calls=0,coarse_calls=1,qualified=float(torch.linalg.vector_norm(r))<=max(1e-12,1e-6*denominator)and divergence<=1e-7,
                        relative_residual=float(torch.linalg.vector_norm(r))/denominator,predicted_divergence=divergence)
                else:pressure,work=multilevel.solve(wrapper,sample,state.normalization,max_passes=8,neural=name=='vcycle-neural')
                seconds=time.perf_counter()-tick
                rows.append(dict(**identity,method=name,**work,seconds=seconds,pressure_relative_l2=pressure_error(pressure,sample)))
            print('Screened',identity['counts_xyz'],identity['frame'],flush=True)
    summaries=[dict(method=name,**summary([r for r in rows if r['method']==name]))for name in methods]
    write(args.output,dict(status='completed',entry=state.entry,cpu=23,threads=1,optimizer_updates=0,
        source_sha256=sha(__file__),multilevel_source_sha256=sha(Path(__file__).with_name('multilevel.py')),
        rows=rows,summary=summaries,geometry_setup=setup,wrapper_parity=parity,elapsed_s=time.perf_counter()-started,
        scope='Consumed development screen only; shape-only setup recorded separately, one exploratory timing per method, eight-cycle ceiling. Not isolated repeated timing or blind qualification.'))
    print(summaries,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--screen',action='store_true')
    run(parser.parse_args())
