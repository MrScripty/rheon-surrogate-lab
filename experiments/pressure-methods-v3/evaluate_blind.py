"""Evaluate frozen models on common consumed requests and reserved native scenes."""
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

def native_cases(state,population):
    reservation=ROOT/'docs/plans/pressure-methods-v3/reservation.json'
    if population=='blind':
        selection=read(ROOT/'docs/plans/pressure-methods-v3/selection.json')
        if selection['status']!='frozen_before_blind_capture':raise ValueError('Frozen selection required')
    for row in read(reservation)['rows']:
        if row['population']!=population:continue
        directory=ROOT/'runs/pressure-methods-v3/native-data'/row['trajectory_id'];receipt=read(directory/'receipt.json')
        if receipt['status']!='completed' or receipt['scene']!=row['scene'] or not receipt['native_process']['reaped']:
            raise ValueError('Complete independently reserved native scene required')
        for frame in receipt['snapshots']:
            sample={}
            for field,descriptor in frame['arrays'].items():
                path=directory/descriptor['path']
                if sha(path)!=descriptor['sha256']:raise ValueError('Native evaluation field changed')
                sample[field]=state.torch.from_numpy(state.np.fromfile(path,dtype='<f8').reshape(descriptor['shape']))
            scene=row['scene'];torch=state.torch
            sample.update(cell_size_m=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64),
                rho=torch.tensor([1000.],dtype=torch.float64),dt=torch.tensor([1/60],dtype=torch.float64))
            yield dict(population='native-'+population,trajectory_id=row['trajectory_id'],frame=frame['index'],
                context='cold'if frame['index']==0 else'warm',counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre']),sample

def evaluate(args):
    if args.output.exists():raise ValueError('Fresh evaluation required')
    os.sched_setaffinity(0,{args.cpu})
    import inference,cases,scalable,numerics
    material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    entry=read(args.entry)
    state=inference.load(dict(material,models=[entry]),entry['id']);torch=state.torch
    inventory=state.data.helpers().state_inventory(state.network)
    wrapper,parity=scalable.qualify(state)
    if hasattr(state.network,'coarse_logits'):
        branch={k:v.detach()for k,v in state.network.state_dict().items()if k=='coarse_logits'or k.startswith('shape_head.')}
    else:branch,_=learned_branch(state.np,torch)
    population=list(native_cases(state,args.population))
    if args.old:population=cases.load(state)+population
    rows=[];started=time.perf_counter()
    with torch.no_grad():
        for identity,sample in population:
            norm=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
            closure=float(torch.linalg.vector_norm(sample['rhs']-numerics.fine_action(sample['base'],sample['cell_size_m'],sample['rho'])-sample['native_input_residual']))/norm
            if closure>1e-10:raise ValueError('Native request closure differs from fine stencil')
            for passes in ([8]if identity['population']!='native-'+args.population else[16]):
                begin=time.perf_counter()
                pressure,work=run_policy(wrapper,sample,state.normalization,branch,args.kind,max_passes=passes)
                seconds=time.perf_counter()-begin
                error=pressure-sample['reference'];error-=error.mean()
                reference=sample['reference']-sample['reference'].mean()
                rows.append(dict(**identity,max_passes=passes,**work,seconds=seconds,
                    pressure_relative_l2=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
                if not bool(torch.isfinite(pressure).all()):raise ValueError('Nonfinite pressure')
    if inventory!=state.data.helpers().state_inventory(state.network)or torch.cuda.is_initialized():raise ValueError('Evaluation changed model state')
    summary=[]
    for population in sorted({r['population']for r in rows}):
        for passes in sorted({r['max_passes']for r in rows if r['population']==population}):
            group=[r for r in rows if r['population']==population and r['max_passes']==passes]
            summary.append(dict(population=population,max_passes=passes,cases=len(group),qualified=sum(r['qualified']for r in group),
                worst_relative_residual=max(r['relative_residual']for r in group),worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in group),
                median_network_calls=statistics.median(r['network_calls']for r in group),median_ms=1000*statistics.median(r['seconds']for r in group)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,dict(status='completed',entry=entry,kind=args.kind,parameters=sum(p.numel()for p in state.network.parameters()),
        raw_forward_parity=parity,unchanged_inventory=True,optimizer_updates=0,rows=rows,summary=summary,
        source_sha256=sha(__file__),policy_source_sha256=sha(Path(__file__).with_name('frozen_matrix.py')),
        reservation_sha256=sha(ROOT/'docs/plans/pressure-methods-v3/reservation.json'),elapsed_s=time.perf_counter()-started,
        timing_scope='One CPU thread pinned to declared core; concurrent exploratory evaluation. Coupled host acceptance and isolated timing remain separate.'))
    print(entry['id'],[(s['population'],s['max_passes'],s['qualified'],s['cases'])for s in summary],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population',choices=['development','blind'],default='development')
    parser.add_argument('--kind',choices=['numerical','learned'],required=True)
    parser.add_argument('--old',action='store_true')
    parser.add_argument('--cpu',type=int,default=21)
    evaluate(parser.parse_args())
