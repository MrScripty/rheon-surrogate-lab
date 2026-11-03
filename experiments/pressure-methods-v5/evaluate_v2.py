"""Common retained/native/wavelength populations; no fitting or label inference."""
import argparse,importlib.util,os,time,statistics
from pathlib import Path
from types import SimpleNamespace
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha
from solver_v2 import load,Policy
from spectrum import pattern

def native(state,population):
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!=population:continue
        directory=BASE/'native-data'/(row['trajectory_id']+'-spectral');receipt=read(directory/'receipt.json')
        if receipt['status']!='completed'or receipt['row']!=row or not receipt['native_process']['reaped']:raise ValueError('Complete reserved native requests required')
        for frame in receipt['snapshots']:
            sample={}
            for field,spec in frame['arrays'].items():
                path=directory/spec['path']
                if sha(path)!=spec['sha256']:raise ValueError('Native input changed')
                sample[field]=state.torch.from_numpy(state.np.fromfile(path,dtype='<f8').reshape(spec['shape']))
            t=state.torch;sample.update(cell_size_m=t.tensor([1/row['scene']['cells_per_metre']],dtype=t.float64),rho=t.tensor([1000.],dtype=t.float64),dt=t.tensor([1/60],dtype=t.float64))
            yield dict(population='v5-native-'+population,trajectory_id=row['trajectory_id'],frame=frame['index'],counts_xyz=row['scene']['counts'],cells_per_metre=row['scene']['cells_per_metre']),sample

def spectral(state,population):
    import numerics
    t=state.torch
    for row in read(DOCS/'scene-reservation.json')['spectrum']:
        if row['population']!=population:continue
        h=t.tensor([1/row['cells_per_metre']],dtype=t.float64);rho=t.tensor([1000.],dtype=t.float64)
        for i in range(row['patterns']):
            reference,details=pattern(row,i);rhs=numerics.fine_action(reference,h,rho);base=t.zeros_like(rhs)
            yield dict(population='v5-spectrum-'+population,trajectory_id=row['trajectory_id'],frame=i,counts_xyz=row['counts_xyz'],cells_per_metre=row['cells_per_metre'],
                mode_band=row['mode_band'],modes=details),dict(base=base,rhs=rhs,native_input_residual=rhs,reference=reference,cell_size_m=h,rho=rho,dt=t.tensor([1/60],dtype=t.float64))

def retained(state):
    import cases
    # Retention inputs use the original validation reader. The fitting reader
    # deliberately accepts training-only banks and applies coarse conditioning.
    proxy=SimpleNamespace(**{**state.__dict__,'data':state.data.historical_data})
    result=cases.load(proxy)
    for version,filename in [(3,'evaluate.py'),(4,'evaluate_candidate.py')]:
        path=ROOT/f'experiments/pressure-methods-v{version}'/filename
        spec=importlib.util.spec_from_file_location('v5_retained_'+str(version),path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for group in ['development','blind']:
            source=module.native_cases(state,group)if version==3 else module.new_native(state,group)
            for identity,sample in source:result.append((dict(identity,population=f'retained-v{version}-'+group),sample))
    if len(result)!=192:raise ValueError('All192 retained prior requests required')
    return result

def summarize(rows):
    return dict(cases=len(rows),qualified=sum(r['qualified']for r in rows),median_ms=1000*statistics.median(r['seconds']for r in rows),
        median_network_calls=statistics.median(r['network_calls']for r in rows),median_coarse_calls=statistics.median(r['coarse_calls']for r in rows),
        worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in rows),worst_relative_residual=max(r['relative_residual']for r in rows))
def run(args):
    if args.output.exists():raise ValueError('Fresh evaluation required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.population=='blind'and not(DOCS/'selection.json').exists():raise ValueError('Freeze selection before blind evaluation')
    os.sched_setaffinity(0,{args.cpu});state,wrapper=load(args.entry);t=state.torch
    inventory=state.data.helpers().state_inventory(state.network)
    policy=Policy(state,wrapper,method=args.method,neural=not args.numerical,reuse=args.reuse,optimized=args.optimized,channels_last=args.channels_last)
    population=(retained(state)if args.old else[])+list(native(state,args.population))+list(spectral(state,args.population))
    rows=[];started=time.perf_counter()
    with t.inference_mode():
        for identity,sample in population:
            tick=time.perf_counter();pressure,work=policy.solve(sample,(identity['population'],identity['trajectory_id']));seconds=time.perf_counter()-tick
            policy.commit(work['qualified'])
            error=pressure-sample['reference'];error-=error.mean();reference=sample['reference']-sample['reference'].mean()
            if not bool(t.isfinite(pressure).all()):raise ValueError('Nonfinite predicted pressure')
            rows.append(dict(**identity,**work,seconds=seconds,pressure_relative_l2=float(t.linalg.vector_norm(error))/max(float(t.linalg.vector_norm(reference)),1e-300)))
    if inventory!=state.data.helpers().state_inventory(state.network)or t.cuda.is_initialized():raise ValueError('CPU evaluation changed weights/device')
    write(args.output,dict(status='completed',identifier=args.identifier,entry=state.entry,method=args.method,neural=not args.numerical,reuse_capacity=args.reuse,
        optimized=args.optimized,channels_last=args.channels_last,rows=rows,
        summary=[dict(population=name,**summarize([r for r in rows if r['population']==name]))for name in sorted({r['population']for r in rows})],
        parameters=sum(p.numel()for p in state.network.parameters())if not args.numerical and args.method!='spectral'else 0,
        cpu=args.cpu,threads=1,optimizer_updates=0,source_sha256=sha(__file__),reservation_sha256=sha(DOCS/'scene-reservation.json'),
        dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver_v2.py','recycling_v2.py','fast_multilevel.py','spectrum.py']},elapsed_s=time.perf_counter()-started,
        timing_scope='Exploratory single evaluation; final isolated repeated timing remains separate. Synthetic frame slots are independent pressure patterns, not fluid timesteps.'))
    print('Evaluated',args.identifier,'qualified',sum(r['qualified']for r in rows),'of',len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['entry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--identifier',required=True);p.add_argument('--method',choices=['spectral','gcr','vcycle'],default='gcr');p.add_argument('--numerical',action='store_true')
    p.add_argument('--reuse',type=int,choices=[0,2,4],default=0);p.add_argument('--optimized',action='store_true');p.add_argument('--channels-last',action='store_true')
    p.add_argument('--population',choices=['development','blind'],default='development');p.add_argument('--old',action='store_true');p.add_argument('--cpu',type=int,default=21);run(p.parse_args())
