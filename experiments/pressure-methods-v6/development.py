"""All consumed v5 inputs are development; no v6 reserved scene is read here."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
from solver import load,Policy
def cases(state):
    from evaluate_v3 import retained,native,spectral
    # These helpers retain their original v5 paths and decode immutable inputs.
    result=retained(state)
    for group in ['development','blind']:
        result+=list(native(state,group))+list(spectral(state,group))
    if len(result)!=254:raise ValueError('Exact192+10+18+10+24 consumed requests required')
    return result
def run(args):
    if args.output.exists():raise ValueError('Fresh development output required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source_sha=sha(__file__);dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver.py','recycling.py','native_local.py','local_native.cpp','local_native_stream.cpp']}
    os.sched_setaffinity(0,{args.cpu});state,network=load(args.entry);t=state.torch;t.set_num_threads(1)
    population=cases(state);rows=[];identities=[];inventory=state.data.helpers().state_inventory(network)
    policy=Policy(state,network,method=args.method,neural=not args.numerical,optimized=True,channels_last=args.channels_last,variant=args.variant,local_execution=args.local_execution)
    with t.inference_mode():
        for identity,sample in population:
            identities.append(dict(meta=identity,arrays={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()for k,v in sample.items()}))
            tick=time.perf_counter();pressure,work=policy.solve(sample,(identity['population'],identity['trajectory_id']));seconds=time.perf_counter()-tick;policy.commit(work['qualified'])
            error=pressure-sample['reference'];error-=error.mean();reference=sample['reference']-sample['reference'].mean()
            if not bool(t.isfinite(pressure).all()):raise ValueError('Nonfinite development pressure')
            rows.append(dict(**identity,**work,seconds=seconds,pressure_relative_l2=float(t.linalg.vector_norm(error))/max(float(t.linalg.vector_norm(reference)),1e-300)))
    if inventory!=state.data.helpers().state_inventory(network)or t.cuda.is_initialized():raise ValueError('Development changed model/device')
    if source_sha!=sha(__file__)or any(sha(ROOT/n)!=h for n,h in dependencies.items()):raise ValueError('Evaluation sources changed during execution; output excluded')
    write(args.output,dict(status='completed',identifier=args.identifier,entry=state.entry,variant=args.variant,local_execution=args.local_execution,
      neural=not args.numerical,method=args.method,channels_last=args.channels_last,rows=rows,
      population_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
      cases=len(rows),qualified=sum(r['qualified']for r in rows),parameters=sum(p.numel()for p in network.parameters()),
      median_coarse_calls=statistics.median(r['coarse_calls']for r in rows),median_network_calls=statistics.median(r['network_calls']for r in rows),
      worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in rows),source_sha256=source_sha,
      dependencies=dependencies,
      optimizer_updates=0,cpu=args.cpu,threads=1,timing_scope='Exploratory; not isolated repeated timing. No v6 reserved scenes used.'))
    print(args.identifier,'qualified',sum(r['qualified']for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--identifier',required=True)
    p.add_argument('--variant',choices=['all','first','first2','stagnation','coarse1'],default='all');p.add_argument('--local-execution',choices=['torch','native','streaming'],default='torch')
    p.add_argument('--channels-last',action='store_true');p.add_argument('--numerical',action='store_true');p.add_argument('--method',choices=['gcr','spectral'],default='gcr');p.add_argument('--cpu',type=int,default=21);run(p.parse_args())
