"""Shared-core, common-population timing for the historical 128-request frontier."""
import argparse
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'experiments/pressure-pass-sweep-v1'),
    str(ROOT/'experiments/pressure-methods-v1'),str(ROOT/'experiments/pressure-methods-v2')]
from common import read,write,sha
from frozen_matrix import run_policy,learned_branch

def benchmark(args):
    if args.output.exists():raise ValueError('Fresh quiet benchmark required')
    os.sched_setaffinity(0,{23})
    import inference,cases,scalable,numerics
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json');entry=read(args.entry)
    state=inference.load(dict(material,models=[entry]),entry['id']);torch=state.torch
    wrapper,parity=scalable.qualify(state);inventory=state.data.helpers().state_inventory(state.network)
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}if hasattr(state.network,'coarse_logits')else learned_branch(state.np,torch)[0]
    population=[(identity,sample)for identity,sample in cases.load(state,include_old_large=False)if identity['population']=='consumed128']
    if len(population)!=128:raise ValueError('Exact comparable original 128 requests required')
    def predict(sample):
        if args.kind in ['learned','numerical']:return run_policy(wrapper,sample,state.normalization,branch,args.kind,max_passes=8)
        base=sample['base'];h=sample['cell_size_m'];rho=sample['rho'];p=base;calls=0
        denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
        limit=max(1e-12,1e-6*denominator)
        def residual():return sample['native_input_residual']-numerics.fine_action(p-base,h,rho)
        def gates(r):return float(torch.linalg.vector_norm(r))<=limit and float(sample['dt'][0]*r.abs().max()/h[0]**3)<=1e-7
        if args.kind=='spectral':p=base+numerics.correction(sample['native_input_residual'],h,rho,None)
        else:
            for _ in range(8):
                if gates(residual()):break
                p=scalable.fine_step(wrapper,sample,state.normalization,p);calls+=1
        r=residual()
        return p,dict(network_calls=calls,coarse_calls=0,qualified=gates(r),relative_residual=float(torch.linalg.vector_norm(r))/denominator)
    rows=[]
    with torch.no_grad():
        # Warm shape-only geometry and allocator caches, without retaining predictions.
        for identity,sample in population:predict(sample)
        for identity,sample in population:
            times=[];prior=None
            for repeat in range(3):
                start=time.perf_counter();pressure,stats=predict(sample);times.append(time.perf_counter()-start)
                if prior is not None and not torch.equal(prior,pressure):raise ValueError('Timing repeats changed pressure bytes')
                prior=pressure.clone()
            error=pressure-sample['reference'];error-=error.mean();reference=sample['reference']-sample['reference'].mean()
            rows.append(dict(**identity,**stats,seconds=statistics.median(times),repeat_seconds=times,
                pressure_relative_l2=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Timing modified model state')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,dict(status='completed',identifier=args.identifier,entry=entry,kind=args.kind,cpu=23,threads=1,
        population='Exact historical128requests, grids5–16cells/axis; not128³ domains',
        timing='One full population warmup, three timed repeats per case, median per case then median across cases. Method, coarse work and stopping checks count; field loading and labels do not.',
        rows=rows,parameters=0 if args.kind=='spectral'else sum(p.numel()for p in state.network.parameters()),
        summary=dict(cases=128,qualified=sum(r['qualified']for r in rows),median_ms=statistics.median(r['seconds']for r in rows)*1000,
            worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in rows),worst_relative_residual=max(r['relative_residual']for r in rows)),
        source_sha256=sha(__file__),policy_source_sha256=sha(Path(__file__).with_name('frozen_matrix.py')),wrapper_parity=parity,optimizer_updates=0))
    print('Quiet common-population benchmark',args.identifier,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True);parser.add_argument('--identifier',required=True)
    parser.add_argument('--kind',choices=['learned','numerical','fine','spectral'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    benchmark(parser.parse_args())
