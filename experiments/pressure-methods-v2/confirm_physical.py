"""Confirm selected fixed-weight policies on complete development populations."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
             str(ROOT/'experiments/pressure-pass-sweep-v1'),str(ROOT/'experiments/pressure-methods-v1')]
from common import read,write,sha

def confirm(args):
    if args.output.exists():raise ValueError('Fresh confirmation directory required')
    os.sched_setaffinity(0,{args.cpu})
    prior=read(args.prior)
    if prior['status']!='completed':raise ValueError('Completed restored/evaluated model required')
    original=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    entry=prior['entry'];material=dict(original,models=[entry])
    import inference,numerical,numerics,cases
    import policies_physical as policies
    state=inference.load(material,entry['id']);state.project=ROOT/entry['project']
    torch=state.torch;selected=read(args.policies)['policies']
    if not 1<=len(selected)<=12:raise ValueError('Finite selected policy list required')
    for policy in selected:
        if policy['kind'] not in ['hybrid','pcg','spectral','retained']:
            raise ValueError('Declared confirmation policy required')
        if policy['kind']=='hybrid' and (policy['nodes'] not in [2,3,4,6] or policy['placement'] not in ['once','each']
                or type(policy['adaptive']) is not bool or not 1<=policy['passes']<=16):
            raise ValueError('Bounded hybrid policy required')
    samples=cases.load(state)
    args.output.mkdir(parents=True)
    write(args.output/'intent.json',dict(prior_sha256=sha(args.prior),policies_sha256=sha(args.policies),
        sample_identities=[identity for identity,_ in samples],cpu=args.cpu,repetitions=args.repetitions))
    inventory=state.data.helpers().state_inventory(state.network)
    calls=[];hook=state.network.register_forward_hook(lambda *_:calls.append(1))
    rows=[];started=time.perf_counter()
    try:
        with torch.no_grad():
            for index,(identity,sample) in enumerate(samples):
                h=sample['cell_size_m'];rho=sample['rho']
                action=lambda pressure:state.model.full_operator(pressure,h,rho)
                denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                closure=float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/denominator
                if closure>1e-10:raise ValueError('Native request closure differs')
                for policy in selected:
                    def execute():
                        calls.clear();tick=time.perf_counter()
                        if policy['kind']=='hybrid':
                            pressure,work=policies.run(state,sample,nodes=policy['nodes'],placement=policy['placement'],
                                max_passes=policy['passes'],adaptive=policy['adaptive'])
                        elif policy['kind']=='pcg':
                            pressure,iterations=numerical.pcg(sample,nodes=policy['nodes'],max_iterations=256)
                            work=dict(iterations=iterations)
                        elif policy['kind']=='spectral':
                            pressure=sample['base']+numerics.correction(sample['native_input_residual'],h,rho,None);work={}
                        else:
                            pressure=inference.fixed(state,sample,policy['passes']);work={}
                        residual=sample['rhs']-action(pressure)
                        relative=float(torch.linalg.vector_norm(residual))/denominator
                        elapsed=time.perf_counter()-tick
                        return pressure,relative,elapsed,len(calls),work
                    cache_before=numerical.geometry.cache_info()._asdict()
                    pressure,relative,first_seconds,count,work=execute()
                    if not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0])!=0:
                        raise ValueError('Finite cell-zero-gauge pressure required')
                    times=[]
                    for _ in range(args.repetitions):
                        repeated,repeated_relative,seconds,repeated_calls,_=execute()
                        if not torch.equal(pressure,repeated) or relative!=repeated_relative or repeated_calls!=count:
                            raise ValueError('Repeated confirmation changed output or work')
                        times.append(seconds)
                    error=pressure-sample['reference'];centered=error-error.mean()
                    reference=sample['reference']-sample['reference'].mean()
                    gradient_error=sum(float(torch.diff(error,dim=d).square().sum()) for d in [2,3,4])
                    gradient_reference=sum(float(torch.diff(reference,dim=d).square().sum()) for d in [2,3,4])
                    rows.append(dict(**identity,policy=policy,relative_residual=relative,
                        qualified=relative<=1e-6 or float(torch.linalg.vector_norm(sample['rhs']-action(pressure)))<=1e-12,
                        pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300),
                        gradient_relative_l2=(gradient_error/max(gradient_reference,1e-300))**.5,
                        first_invocation_seconds=first_seconds,warm_seconds=times,
                        median_policy_seconds=statistics.median(times),network_calls=count,work=work,
                        geometry_cache_before=cache_before,geometry_cache_after=numerical.geometry.cache_info()._asdict()))
                if (index+1)%16==0:print('Confirmed',index+1,'requests',flush=True)
    finally:hook.remove()
    if state.data.helpers().state_inventory(state.network)!=inventory:raise ValueError('Inference changed network tensors')
    summary=[]
    for population in sorted({row['population'] for row in rows}):
        for policy in selected:
            group=[r for r in rows if r['population']==population and r['policy']==policy]
            summary.append(dict(population=population,policy=policy,cases=len(group),qualified=sum(r['qualified'] for r in group),
                worst_relative_residual=max(r['relative_residual'] for r in group),
                worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group),
                worst_gradient_relative_l2=max(r['gradient_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(r['median_policy_seconds'] for r in group)))
    write(args.output/'results.json',dict(status='completed',entry=entry,rows=rows,summary=summary,
        prior_evaluation_sha256=sha(args.prior),policies_sha256=sha(args.policies),measured_cases=len(samples),
        source_sha256=sha(__file__),helpers={name:sha(HERE/name) for name in ['cases.py','policies_physical.py','numerical.py']},
        peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_s=time.perf_counter()-started,
        timing_scope='Prepared physical request through final true fine residual; numerical work counted. First invocation and warm repeats separate; no loading, oracle metrics, IPC or simulation in policy timing.',
        concurrency_scope=args.concurrency,optimizer_updates=0,blind_scenes_consumed=0,physical_stopping='Relative residual and predicted divergence',cpu=args.cpu,threads=1,
        unchanged_tensor_inventory=True,repetitions=args.repetitions))
    print('Completed full confirmation',entry['id'],len(samples),'requests',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--policies',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=18)
    parser.add_argument('--repetitions',type=int,choices=[1,3],default=1)
    parser.add_argument('--concurrency',choices=['campaign-workers-active','owned-workers-quiet'],default='campaign-workers-active')
    args=parser.parse_args()
    for name in ['prior','policies','output']:setattr(args,name,getattr(args,name).resolve())
    confirm(args)
