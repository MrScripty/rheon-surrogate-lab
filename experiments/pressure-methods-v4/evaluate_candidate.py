"""Common-population evaluation of the declared v4 mechanisms and completed fits."""
import argparse
import os
from pathlib import Path
import time
from v4_support import ROOT,BASE,DOCS,read,write,sha,summary,pressure_error,native

def new_native(state,population):
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!=population:continue
        directory=BASE/'native-data'/(row['trajectory_id']+'-spectral');receipt=read(directory/'receipt.json')
        if receipt['status']!='completed'or receipt['row']!=row or not receipt['native_process']['reaped']:
            raise ValueError('Qualified reserved native evaluation scene required')
        for frame in receipt['snapshots']:
            sample={}
            for field,spec in frame['arrays'].items():
                path=directory/spec['path']
                if sha(path)!=spec['sha256']:raise ValueError('Reserved native field changed')
                sample[field]=state.torch.from_numpy(state.np.fromfile(path,dtype='<f8').reshape(spec['shape']))
            torch=state.torch
            sample.update(cell_size_m=torch.tensor([1/row['scene']['cells_per_metre']],dtype=torch.float64),
                rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
            yield dict(population='v4-'+population,trajectory_id=row['trajectory_id'],frame=frame['index'],
                counts_xyz=row['scene']['counts'],cells_per_metre=row['scene']['cells_per_metre']),sample

def run(args):
    if args.output.exists():raise ValueError('Fresh candidate evaluation required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.population=='blind'and not (DOCS/'selection.json').exists():raise ValueError('Frozen blind selection required')
    os.sched_setaffinity(0,{args.cpu})
    import inference
    entry=read(args.entry);material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(material,models=[entry]),entry['id'])
    import torch,scalable,cases,multilevel,preconditioned,numerics
    from frozen_matrix import run_policy
    inventory=state.data.helpers().state_inventory(state.network);wrapper,parity=scalable.qualify(state)
    if args.policy=='large-channels-last':
        from execution import LargeChannelsLast
        wrapper=LargeChannelsLast(wrapper).eval()
    branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    population=list(new_native(state,args.population))
    if args.old:population=cases.load(state)+native(state)+population
    rows=[];started=time.perf_counter()
    with torch.inference_mode():
        for identity,sample in population:
            tick=time.perf_counter()
            if args.policy in ['learned','large-channels-last','numerical']:
                pressure,work=run_policy(wrapper,sample,state.normalization,branch,'numerical'if args.policy=='numerical'else'learned',max_passes=8)
            elif args.policy=='spectral':
                pressure=sample['base']+numerics.correction(sample['native_input_residual'],sample['cell_size_m'],sample['rho'],None)
                r=sample['native_input_residual']-numerics.fine_action(pressure-sample['base'],sample['cell_size_m'],sample['rho'])
                denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                divergence=float(sample['dt'][0]*r.abs().max()/sample['cell_size_m'][0]**3)
                work=dict(network_calls=0,coarse_calls=1,qualified=float(torch.linalg.vector_norm(r))<=max(1e-12,denominator*1e-6)and divergence<=1e-7,
                    relative_residual=float(torch.linalg.vector_norm(r))/denominator,predicted_divergence=divergence)
            elif args.policy.startswith('gcr-'):pressure,work=preconditioned.solve(wrapper,sample,state.normalization,branch,kind=args.policy[4:],max_iterations=8)
            else:pressure,work=multilevel.solve(wrapper,sample,state.normalization,max_passes=8,neural=args.policy=='vcycle-neural')
            seconds=time.perf_counter()-tick
            rows.append(dict(**identity,**work,seconds=seconds,pressure_relative_l2=pressure_error(pressure,sample)))
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Evaluation changed original model state')
    groups=[]
    for population_name in sorted({r['population']for r in rows}):
        groups.append(dict(population=population_name,**summary([r for r in rows if r['population']==population_name])))
    write(args.output,dict(status='completed',entry=entry,policy=args.policy,rows=rows,summary=groups,
        parameters=0 if args.policy in ['spectral','vcycle-only','gcr-multilevel']else sum(p.numel()for p in state.network.parameters()),
        cpu=args.cpu,threads=1,optimizer_updates=0,wrapper_parity=parity,unchanged_inventory=True,
        source_sha256=sha(__file__),dependency_sources={str(p.relative_to(ROOT)):sha(p)for p in [
            Path(__file__).with_name('multilevel.py'),Path(__file__).with_name('preconditioned.py'),Path(__file__).with_name('execution.py')]},
        reservation_sha256=sha(DOCS/'scene-reservation.json'),selection_rule_sha256=sha(DOCS/'selection-rule.json'),elapsed_s=time.perf_counter()-started,
        timing_scope='Single exploratory timing with all policy work and stopping checks. Campaign work may run concurrently; final repeated timing is separate.'))
    print('Evaluated',entry['id'],args.policy,[(s['population'],s['qualified'],s['cases'])for s in groups],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['entry','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy',choices=['learned','numerical','large-channels-last','spectral','vcycle-neural','vcycle-only','gcr-learned','gcr-multilevel','gcr-multilevel-neural'],required=True)
    parser.add_argument('--population',choices=['development','blind'],default='development');parser.add_argument('--old',action='store_true');parser.add_argument('--cpu',type=int,default=21)
    run(parser.parse_args())
