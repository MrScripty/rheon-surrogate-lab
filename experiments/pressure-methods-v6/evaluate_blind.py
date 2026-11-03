"""Stream the same fourteen held-out native equations through each frozen point."""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha
from solver_v2 import load,Policy
from fixed_control import FixedJacobi,work_kind
def run(a):
    if a.output.exists():raise ValueError('Fresh blind evaluation required')
    a.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{a.cpu})
    selection=read(DOCS/'selection.json');point=next(p for p in selection['blind_points']if p['identifier']==a.identifier)
    for name,expected in selection['frozen_sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen predictor changed before blind evaluation')
    state,network=load(ROOT/point['entry']);t=state.torch;t.set_num_threads(1)
    fixed=point.get('fixed_jacobi',False)
    policy=Policy(state,FixedJacobi()if fixed else network,method=point['method'],neural=fixed or not point['numerical'],optimized=True,channels_last=point['channels_last'],variant=point['variant'],local_execution=point['local_execution'])
    import numerics
    inventory=state.data.helpers().state_inventory(network);rows=[];identities=[];started=time.perf_counter()
    with t.inference_mode():
        for row in read(DOCS/'scene-reservation.json')['rows']:
            if row['population']not in ['blind','large-blind']:continue
            directory=BASE/'native-data'/(row['trajectory_id']+'-spectral');receipt=read(directory/'receipt.json')
            if receipt['status']!='completed'or receipt['row']!=row or not receipt['native_process']['reaped']:raise ValueError('Complete reserved same-request native capture required')
            if [s['index']for s in receipt['snapshots']]!=row['snapshots']:raise ValueError('Requested cold/warm snapshots missing')
            for snap in receipt['snapshots']:
                sample={}
                for field,spec in snap['arrays'].items():
                    p=directory/spec['path']
                    if sha(p)!=spec['sha256']:raise ValueError('Native input bytes changed')
                    sample[field]=t.from_numpy(state.np.fromfile(p,dtype='<f8').reshape(spec['shape']))
                sample.update(cell_size_m=t.tensor([1/row['scene']['cells_per_metre']],dtype=t.float64),rho=t.tensor([1000.],dtype=t.float64),dt=t.tensor([1/60],dtype=t.float64))
                meta=dict(population=row['population'],trajectory_id=row['trajectory_id'],frame=snap['index'],counts_xyz=row['scene']['counts'],cells_per_metre=row['scene']['cells_per_metre'])
                identities.append(dict(meta=meta,fields=snap['arrays']))
                tick=time.perf_counter();p,work=policy.solve(sample,(row['trajectory_id'],snap['index']));seconds=time.perf_counter()-tick;policy.commit(work['qualified']);work=work_kind(work,fixed)
                residual=sample['rhs']-numerics.fine_action(p,sample['cell_size_m'],sample['rho']);den=max(float(t.linalg.vector_norm(sample['rhs'])),1e-300)
                div=float(sample['dt'][0]*residual.abs().max()/sample['cell_size_m'][0]**3);actual=float(t.linalg.vector_norm(residual))<=max(1e-12,den*1e-6)and div<=1e-7
                if actual!=work['qualified']:raise ValueError('Independent returned-pressure gate disagrees')
                error=p-sample['reference'];error-=error.mean();ref=sample['reference']-sample['reference'].mean()
                rows.append(dict(**meta,**work,seconds=seconds,pressure_relative_l2=float(t.linalg.vector_norm(error))/max(float(t.linalg.vector_norm(ref)),1e-300)))
    if len(rows)!=14 or inventory!=state.data.helpers().state_inventory(network)or t.cuda.is_initialized():raise ValueError('Complete14-request fixed-weight CPU matrix required')
    for name,expected in selection['frozen_sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen predictor changed during blind evaluation')
    write(a.output,dict(status='completed',identifier=a.identifier,point=point,entry=state.entry,rows=rows,cases=14,qualified=sum(r['qualified']for r in rows),
      population_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),selection_sha256=sha(DOCS/'selection.json'),
      source_sha256=sha(__file__),cpu=a.cpu,threads=1,optimizer_updates=0,elapsed_s=time.perf_counter()-started,
      timing_scope='Exploratory blind accuracy/work, not isolated repeated timing; inputs streamed one full equation at a time.'))
    print(a.identifier,'blind qualification',sum(r['qualified']for r in rows),'/14',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identifier',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,default=21);run(p.parse_args())
