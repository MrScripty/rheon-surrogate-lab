"""Serial matched pressure timing, including an independent returned-field gate."""
import argparse,os,time,statistics
from pathlib import Path
from types import SimpleNamespace
from v5_support import ROOT,HERE,DOCS,read,write,sha
from solver_v3 import load,Policy
def run(args):
    if args.output.exists():raise ValueError('Fresh quiet timing required')
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{23})
    if args.initial_control:
        from evaluate_initial_v3 import initial_load
        state,wrapper=initial_load(args.entry)
    else:state,wrapper=load(args.entry)
    import torch,numerics,cases,multilevel
    from evaluate_v3 import native
    if args.population=='historical128':
        proxy=SimpleNamespace(**{**state.__dict__,'data':state.data.historical_data})
        population=[r for r in cases.load(proxy,include_old_large=False)if r[0]['population']=='consumed128']
        if len(population)!=128:raise ValueError('Exact128 historical requests required')
    else:
        population=list(native(state,'blind'))
        if len(population)!=10:raise ValueError('Exact10 fresh native requests required')
    policy=Policy(state,wrapper,method='vcycle'if args.method=='legacy'else args.method,neural=not args.numerical,
        optimized=args.optimized,channels_last=args.channels_last)
    inventory=state.data.helpers().state_inventory(state.network)
    def predict(sample,identity):
        if args.method=='legacy':pressure,work=multilevel.solve(wrapper,sample,state.normalization,max_passes=8,neural=True)
        else:
            pressure,work=policy.solve(sample,identity);policy.commit(work['qualified'])
        r=sample['rhs']-numerics.fine_action(pressure,sample['cell_size_m'],sample['rho'])
        den=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300);div=float(sample['dt'][0]*r.abs().max()/sample['cell_size_m'][0]**3)
        actual=float(torch.linalg.vector_norm(r))<=max(1e-12,den*1e-6)and div<=1e-7
        return pressure,dict(work,reported_qualified=work['qualified'],qualified=actual,relative_residual=float(torch.linalg.vector_norm(r))/den,predicted_divergence=div)
    rows=[]
    with torch.inference_mode():
        for i,(identity,sample)in enumerate(population):predict(sample,('warmup',i))
        for i,(identity,sample)in enumerate(population):
            times=[];prior=None
            for repeat in range(3):
                tick=time.perf_counter();pressure,work=predict(sample,('independent',i,repeat));times.append(time.perf_counter()-tick)
                if prior is not None and not torch.equal(prior,pressure):raise ValueError('Repeated prediction bytes changed')
                prior=pressure.clone()
            error=pressure-sample['reference'];error-=error.mean();reference=sample['reference']-sample['reference'].mean()
            rows.append(dict(**identity,**work,seconds=statistics.median(times),repeat_seconds=times,
                pressure_relative_l2=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('CPU timing changed weights/device')
    write(args.output,dict(status='completed',identifier=args.identifier,entry=state.entry,method=args.method,parameters=0 if args.numerical or args.method=='spectral'else sum(p.numel()for p in state.network.parameters()),
        population=args.population,rows=rows,summary=dict(cases=len(rows),qualified=sum(r['qualified']for r in rows),median_ms=1000*statistics.median(r['seconds']for r in rows),
            worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in rows),median_network_calls=statistics.median(r['network_calls']for r in rows)),
        optimized=args.optimized,channels_last=args.channels_last,initialization_control=args.initial_control,cpu=23,threads=1,repeats=3,optimizer_updates=0,
        source_sha256=sha(__file__),selection_sha256=sha(DOCS/'selection.json'),dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver_v3.py','recycling_v3.py','fast_multilevel.py']},
        timing_scope='Whole-population warmup, three timed repeats per request. All network/coarse/conversion/stopping work plus a common independent returned-pressure gate count. Loading, model setup and labels do not. Reuse is disabled for independent-request timing.'))
    print('Quiet pressure timing',args.identifier,args.population,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['entry','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--identifier',required=True);p.add_argument('--method',choices=['gcr','vcycle','spectral','legacy'],default='gcr');p.add_argument('--population',choices=['historical128','native-blind'],required=True)
    for flag in ['numerical','optimized','channels-last','initial-control']:p.add_argument('--'+flag,action='store_true')
    run(p.parse_args())

