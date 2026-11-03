"""Repeated CPU pressure-policy benchmark on a declared identical population."""
import argparse
import os
from pathlib import Path
import statistics
import time
from v4_support import ROOT,DOCS,read,write,sha,summary,pressure_error

def benchmark(args):
    if args.output.exists():raise ValueError('Fresh quiet benchmark required')
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{23})
    import inference
    entry=read(args.entry);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    import torch,scalable,cases,numerics,multilevel,preconditioned
    from frozen_matrix import run_policy,learned_branch
    from evaluate_candidate import new_native
    wrapper,parity=scalable.qualify(state);inventory=state.data.helpers().state_inventory(state.network)
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    if args.policy in ['learned','gcr-learned','large-channels-last']and not branch:branch=learned_branch(state.np,torch)[0]
    if args.policy=='large-channels-last':
        from execution import LargeChannelsLast
        wrapper=LargeChannelsLast(wrapper).eval()
    if args.population=='historical128':
        population=[r for r in cases.load(state,include_old_large=False)if r[0]['population']=='consumed128']
        if len(population)!=128:raise ValueError('Exact original128-request population required')
    else:
        population=list(new_native(state,'development'))
        if len(population)!=10:raise ValueError('Exact v4 ten-request population required')
    def predict(sample):
        if args.policy in ['learned','numerical','large-channels-last']:
            return run_policy(wrapper,sample,state.normalization,branch,'numerical'if args.policy=='numerical'else'learned',max_passes=8)
        if args.policy.startswith('gcr-'):return preconditioned.solve(wrapper,sample,state.normalization,branch,kind=args.policy[4:],max_iterations=8)
        if args.policy in ['vcycle-neural','vcycle-only']:return multilevel.solve(wrapper,sample,state.normalization,max_passes=8,neural=args.policy=='vcycle-neural')
        pressure=sample['base'];h=sample['cell_size_m'];rho=sample['rho'];calls=0
        denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300);limit=max(1e-12,denominator*1e-6)
        def residual():return sample['native_input_residual']-numerics.fine_action(pressure-sample['base'],h,rho)
        def gates(r):return float(torch.linalg.vector_norm(r))<=limit and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
        if args.policy=='spectral':pressure=pressure+numerics.correction(sample['native_input_residual'],h,rho,None)
        else:
            for _ in range(8):
                if gates(residual()):break
                pressure=scalable.fine_step(wrapper,sample,state.normalization,pressure);calls+=1
        r=residual()
        return pressure,dict(network_calls=calls,coarse_calls=1 if args.policy=='spectral'else 0,qualified=gates(r),relative_residual=float(torch.linalg.vector_norm(r))/denominator)
    rows=[];started=time.perf_counter()
    with torch.inference_mode():
        for _,sample in population:predict(sample)
        for identity,sample in population:
            times=[];prior=None
            for _ in range(3):
                tick=time.perf_counter();pressure,work=predict(sample);times.append(time.perf_counter()-tick)
                if prior is not None and not torch.equal(prior,pressure):raise ValueError('Repeated inference changed pressure bytes')
                prior=pressure.clone()
            rows.append(dict(**identity,**work,seconds=statistics.median(times),repeat_seconds=times,pressure_relative_l2=pressure_error(pressure,sample)))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Quiet timing modified original model')
    write(args.output,dict(status='completed',identifier=args.identifier,entry=entry,policy=args.policy,population=args.population,
        parameters=0 if args.policy in ['spectral','vcycle-only','gcr-multilevel']else sum(p.numel()for p in state.network.parameters()),
        rows=rows,summary=summary(rows),cpu=23,threads=1,repeats=3,optimizer_updates=0,wrapper_parity=parity,
        source_sha256=sha(__file__),dependency_sources={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n)for n in ['multilevel.py','preconditioned.py','execution.py']},
        elapsed_s=time.perf_counter()-started,timing_scope='One whole-population warmup, three timed repeats per case. Median per case then median across identical requests. All method/coarse/conversion/stopping work counts; field loading, labels, compilation and model setup do not.'))
    print('Quiet benchmark',args.identifier,args.population,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--identifier',required=True);parser.add_argument('--policy',choices=['fine','learned','numerical','spectral','large-channels-last','vcycle-neural','vcycle-only','gcr-learned','gcr-multilevel','gcr-multilevel-neural'],required=True)
    parser.add_argument('--population',choices=['historical128','v4-development10'],required=True)
    benchmark(parser.parse_args())
