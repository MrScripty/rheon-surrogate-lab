"""Matched repeated total pressure-policy time on consumed native development."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
from v6_support import ROOT,HERE,BASE,read,write,sha
from solver import load,Policy
from fixed_control import FixedJacobi,work_kind
def run(a):
    if a.output.exists():raise ValueError('Fresh repeated pressure timing required')
    a.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{23})
    closure={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver.py','recycling.py','native_local.py','local_native.cpp','local_native_stream.cpp','fixed_control.py']};source=sha(__file__)
    point=read(a.points)['points'];point=next(p for p in point if p['identifier']==a.identifier)
    state,network=load(ROOT/point['entry']);t=state.torch;t.set_num_threads(1)
    from evaluate_v3 import native
    import numerics
    population=list(native(state,'development'))+list(native(state,'blind'))
    if len(population)!=20:raise ValueError('Exact20 consumed native requests required')
    fixed=point.get('fixed_jacobi',False)
    policy=Policy(state,FixedJacobi()if fixed else network,method=point.get('method','gcr'),neural=fixed or not point.get('numerical',False),optimized=True,
      variant=point.get('variant','all'),local_execution=point.get('local_execution','torch'),channels_last=point.get('channels_last',False))
    inventory=state.data.helpers().state_inventory(network);identities=[]
    def predict(meta,sample,key):
        p,w=policy.solve(sample,key);policy.commit(w['qualified'])
        residual=sample['rhs']-numerics.fine_action(p,sample['cell_size_m'],sample['rho'])
        denominator=max(float(t.linalg.vector_norm(sample['rhs'])),1e-300);norm=float(t.linalg.vector_norm(residual));div=float(sample['dt'][0]*residual.abs().max()/sample['cell_size_m'][0]**3)
        actual=norm<=max(1e-12,denominator*1e-6)and div<=1e-7
        if actual!=w['qualified']:raise ValueError('Returned-field gate disagrees with policy')
        return p,work_kind(dict(w,qualified=actual,relative_residual=norm/denominator,predicted_divergence=div),fixed)
    rows=[]
    with t.inference_mode():
        for i,(meta,s)in enumerate(population):predict(meta,s,('warmup',i))
        for i,(meta,s)in enumerate(population):
            identities.append(dict(meta=meta,arrays={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()for k,v in s.items()}));times=[];previous=None
            for repeat in range(3):
                tick=time.perf_counter();pressure,work=predict(meta,s,('timed',i,repeat));times.append(time.perf_counter()-tick)
                if previous is not None and not t.equal(previous,pressure):raise ValueError('Repeated pressure changed')
                previous=pressure.clone()
            error=pressure-s['reference'];error-=error.mean();reference=s['reference']-s['reference'].mean()
            rows.append(dict(**meta,**work,repeat_seconds=times,seconds=statistics.median(times),
              pressure_relative_l2=float(t.linalg.vector_norm(error))/max(float(t.linalg.vector_norm(reference)),1e-300)))
    if inventory!=state.data.helpers().state_inventory(network)or t.cuda.is_initialized():raise ValueError('Timing changed weights/device')
    if source!=sha(__file__)or any(sha(ROOT/n)!=h for n,h in closure.items()):raise ValueError('Timing source changed while running')
    write(a.output,dict(status='completed',identifier=a.identifier,point=point,entry=state.entry,rows=rows,summary=dict(cases=20,qualified=sum(r['qualified']for r in rows),
      median_ms=1000*statistics.median(r['seconds']for r in rows),worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in rows),
      median_coarse_calls=statistics.median(r['coarse_calls']for r in rows),median_network_calls=statistics.median(r['network_calls']for r in rows)),
      parameters=0 if point.get('numerical')else sum(p.numel()for p in network.parameters()),
      population_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),cpu=23,threads=1,repeats=3,optimizer_updates=0,
      source_sha256=source,dependencies=closure,points_sha256=sha(a.points),
      timing_scope='One complete warmup then3 repeats per request, all correction/operator/conversion/stopping work and one common independent returned-field gate. Loading, setup and labels excluded. No inter-request reuse.'))
    print('Timed20 common consumed native requests:',a.identifier,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--points',type=Path,required=True);p.add_argument('--identifier',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
