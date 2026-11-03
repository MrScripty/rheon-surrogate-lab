"""Evaluate completed managed fits and bounded coarse-policy comparisons."""
import argparse
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha
import inference

def evaluate(args):
    if args.output.exists():raise ValueError('Fresh evaluation required')
    os.sched_setaffinity(0,{args.cpu})
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    if args.study:
        receipt=read(args.study/'receipt.json')
        if receipt['status']!='completed' or not all(receipt['artifact_parity'].values()):
            raise ValueError('Completed CUDA fit and exact fresh CPU restore required')
        artifacts=args.study/'restore'
        entry=dict(id=args.name,project=str(args.project.relative_to(ROOT)),restore=str(artifacts.relative_to(ROOT)),
            weights_sha256=receipt['weights_sha256'],identities={str(p.relative_to(ROOT)):sha(p) for p in
            [args.project/'source-witness.json',args.project/'config.json',artifacts/'weights.npz',
             artifacts/'normalization.json',artifacts/'receipt.json',artifacts/'history.json']})
        material=dict(material,models=[entry])
    elif args.prior:
        entry=read(args.prior)['entry'];material=dict(material,models=[entry])
    else:entry=next(m for m in material['models'] if m['id']=='C53')
    state=inference.load(material,entry['id']);state.project=ROOT/entry['project']
    import cases
    samples=cases.load(state)
    if args.cost_screen:samples=cases.screen(samples)
    import policies,numerical
    torch=state.torch
    selected=([dict(kind='hybrid',nodes=n,placement=p,adaptive=a,passes=8)
        for n in [2,3,4,6] for p in ['once','each'] for a in [False,True]]+
        [dict(kind='pcg',nodes=n,max_iterations=256) for n in [2,3,4,6]]) if args.cost_screen else [dict(kind='retained',passes=n) for n in [2,4,6,8]]
    args.output.mkdir(parents=True)
    rows=[];started=time.perf_counter();calls=[]
    hook=state.network.register_forward_hook(lambda *_:calls.append(1))
    try:
        with torch.no_grad():
            for identity,sample in samples:
                h=sample['cell_size_m'];rho=sample['rho']
                action=lambda p:state.model.full_operator(p,h,rho)
                norm=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                closure=float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/norm
                if closure>1e-10:raise ValueError('Native pressure request closure differs')
                for policy in selected:
                    calls.clear();tick=time.perf_counter()
                    if policy['kind']=='retained':
                        pressure=inference.fixed(state,sample,policy['passes']);work={}
                    elif policy['kind']=='hybrid':
                        pressure,work=policies.run(state,sample,nodes=policy['nodes'],placement=policy['placement'],
                            max_passes=policy['passes'],adaptive=policy['adaptive'])
                    else:
                        pressure,iterations=numerical.pcg(sample,nodes=policy['nodes'],max_iterations=policy['max_iterations'])
                        work=dict(iterations=iterations)
                    remaining=sample['rhs']-action(pressure)
                    relative=float(torch.linalg.vector_norm(remaining))/norm
                    elapsed=time.perf_counter()-tick
                    if not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0])!=0:
                        raise ValueError('Finite cell-zero-gauge output required')
                    error=pressure-sample['reference'];centered=error-error.mean()
                    reference=sample['reference']-sample['reference'].mean()
                    pressure_error=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300)
                    rows.append(dict(**identity,policy=policy,relative_residual=relative,pressure_relative_l2=pressure_error,
                        qualified=float(torch.linalg.vector_norm(remaining))<=max(1e-12,1e-6*norm),
                        network_calls=len(calls),policy_seconds=elapsed,work=work))
    finally:hook.remove()
    import json
    summary=[]
    for population in sorted({r['population'] for r in rows}):
        for policy in selected:
            group=[r for r in rows if r['population']==population and r['policy']==policy]
            summary.append(dict(population=population,policy=policy,cases=len(group),qualified=sum(r['qualified'] for r in group),
                worst_relative_residual=max(r['relative_residual'] for r in group),
                worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(r['policy_seconds'] for r in group)))
    result=dict(status='completed',name=args.name,entry=entry,rows=rows,summary=summary,optimizer_updates=0,
        source_sha256=sha(__file__),helper_sources={p.name:sha(p) for p in [Path(__file__).with_name(n) for n in ['cases.py','policies.py','numerical.py']]},
        reservation_sha256=sha(ROOT/'docs/plans/pressure-methods-v2/reservation.json'),blind_scenes_consumed=0,
        elapsed_s=time.perf_counter()-started,peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing_scope='Single prepared request through final true fine residual; first invocation of each policy, geometry caches shared across cases. Finalists require isolated repeated timings.',
        measured_cases=len(samples),cost_screen=args.cost_screen)
    write(args.output/'results.json',result)
    print(json.dumps(dict(status='completed',name=args.name,cases=len(samples),measurements=len(rows))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--project',type=Path)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=22)
    parser.add_argument('--cost-screen',action='store_true')
    args=parser.parse_args()
    for name in ['project','study','prior','output']:
        if getattr(args,name):setattr(args,name,getattr(args,name).resolve())
    evaluate(args)
