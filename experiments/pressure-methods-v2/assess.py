"""Frozen-model blind accuracy or isolated repeated development measurements."""
import argparse
import json
import math
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

def blind_cases(state,identifier):
    docs=ROOT/'docs/plans/pressure-methods-v2';base=ROOT/'runs/pressure-methods-v2'
    selection=read(docs/'selection.json');unlock=read(docs/'blind-unlock.json')
    if unlock['selection_sha256']!=sha(docs/'selection.json') or not unlock['no_training_overlap']:
        raise ValueError('Blind capture must follow the frozen development selection')
    allowed={'C53','v2-small16','v2-small24'}
    for row in selection['selected']:
        allowed.update([row['arm'],row['arm']+'-s25',row['arm']+'-s26'])
    if identifier not in allowed:raise ValueError('Only frozen finalists, repeats and declared controls may see blind labels')
    for row in selection['selected']:
        if identifier==row['arm']:
            path=ROOT/row['evaluation']
            if sha(path)!=row['evaluation_sha256'] or read(path)['entry']!=state.entry:
                raise ValueError('Selected checkpoint identity changed before blind evaluation')
        elif identifier in [row['arm']+'-s25',row['arm']+'-s26']:
            repeat=read(ROOT/state.entry['project']/'repeat.json')
            if (repeat['selection_sha256']!=sha(docs/'selection.json') or repeat['parent_arm']!=row['arm']
                    or repeat['identifier']!=identifier or repeat['seed'] not in [2026101825,2026101826]):
                raise ValueError('Blind repeat must retain its frozen selection and paired seed identity')
    reservation=read(docs/'reservation.json');results=[];evidence={}
    for row in reservation['scenes']:
        if row['population']!='blind':continue
        directory=base/'blind-native-data'/row['trajectory_id'];receipt=read(directory/'receipt.json')
        if receipt['status']!='qualified_pilot' or not receipt['native_process']['reaped'] or receipt['row']!=row:
            raise ValueError('Qualified exact blind native capture required')
        evidence[str((directory/'receipt.json').relative_to(ROOT))]=sha(directory/'receipt.json')
        for frame in receipt['frames']:
            sample={}
            for field in ['base','rhs','reference','native_input_residual']:
                descriptor=frame['arrays']['pressure' if field=='reference' else field];path=directory/descriptor['path']
                if sha(path)!=descriptor['sha256']:raise ValueError('Blind field changed')
                array=state.np.fromfile(path,dtype='<f8').reshape(1,1,*row['counts_xyz'][::-1])
                sample[field]=state.torch.from_numpy(array)
            sample.update(cell_size_m=state.torch.tensor([1/row['cells_per_metre']],dtype=state.torch.float64),
                rho=state.torch.tensor([1000.],dtype=state.torch.float64),dt=state.torch.tensor([1/60],dtype=state.torch.float64))
            identity=dict(population='blind-small' if max(row['counts_xyz'])<=16 else 'blind-rectangular',
                trajectory_id=row['trajectory_id'],frame=frame['index'],counts_xyz=row['counts_xyz'],cells_per_metre=row['cells_per_metre'])
            results.append((identity,sample))
    if len(evidence)!=8 or len(results)!=16:raise ValueError('All eight whole blind scenes and both snapshots required')
    return results,evidence

def assess(args):
    if args.output.exists():raise ValueError('Fresh assessment required')
    os.sched_setaffinity(0,{args.cpu})
    prior=read(args.prior)
    if prior['status']!='completed':raise ValueError('Completed evaluated checkpoint required')
    entry=prior['entry'];identifier=entry['id']
    import inference,numerics
    original=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    state=inference.load(dict(original,models=[entry]),identifier);torch=state.torch
    wrapper=None;wrapper_qualification=[]
    if identifier=='C53':
        import scalable
        wrapper,wrapper_qualification=scalable.qualify(state)
    if args.population=='blind':samples,evidence=blind_cases(state,identifier)
    else:
        import cases
        samples=cases.load(state);evidence={}
    args.output.mkdir(parents=True);(args.output/'executed-source.py').write_bytes(Path(__file__).read_bytes())
    write(args.output/'intent.json',dict(entry=entry,passes=args.passes,population=args.population,
        prior_sha256=sha(args.prior),sample_identities=[i for i,_ in samples],repetitions=args.repetitions,cpu=args.cpu))
    calls=[];hook=(wrapper if wrapper else state.network).register_forward_hook(lambda *_:calls.append(1))
    inventory=state.data.helpers().state_inventory(state.network)
    rows=[];started=time.perf_counter()
    try:
        with torch.no_grad():
            for index,(identity,sample) in enumerate(samples):
                action=lambda pressure:numerics.fine_action(pressure,sample['cell_size_m'],sample['rho'])
                norm=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                closure=float(torch.linalg.vector_norm(sample['rhs']-action(sample['base'])-sample['native_input_residual']))/norm
                if closure>1e-10:raise ValueError('Actual native request closure failed independent fine stencil')
                for passes in args.passes:
                    def execute():
                        calls.clear();tick=time.perf_counter()
                        if wrapper:
                            pressure=sample['base']
                            for _ in range(passes):pressure=scalable.fine_step(wrapper,sample,state.normalization,pressure)
                        else:pressure=inference.fixed(state,sample,passes)
                        remaining=sample['rhs']-action(pressure)
                        relative=float(torch.linalg.vector_norm(remaining))/norm
                        elapsed=time.perf_counter()-tick
                        return pressure,relative,elapsed,len(calls)
                    pressure,relative,first,count=execute();times=[]
                    for _ in range(args.repetitions):
                        repeated,repeated_relative,seconds,repeated_count=execute()
                        if not torch.equal(repeated,pressure) or repeated_relative!=relative or count!=repeated_count:
                            raise ValueError('Assessment repeats changed output or work')
                        times.append(seconds)
                    if count>passes or not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0])!=0:
                        raise ValueError('Finite gauge-fixed bounded neural output required')
                    error=pressure-sample['reference'];centered=error-error.mean()
                    reference=sample['reference']-sample['reference'].mean()
                    gradient_error=sum(float(torch.diff(error,dim=d).square().sum()) for d in [2,3,4])
                    gradient_reference=sum(float(torch.diff(reference,dim=d).square().sum()) for d in [2,3,4])
                    rows.append(dict(**identity,passes=passes,network_calls=count,relative_residual=relative,
                        qualified=relative<=1e-6 or float(torch.linalg.vector_norm(sample['rhs']-action(pressure)))<=1e-12,
                        pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300),
                        gauged_pressure_relative_l2=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(sample['reference'])),1e-300),
                        gradient_relative_l2=math.sqrt(gradient_error/max(gradient_reference,1e-300)),
                        first_invocation_seconds=first,warm_seconds=times,median_policy_seconds=statistics.median(times) if times else first))
                if (index+1)%32==0:print('Assessed',identifier,index+1,'requests',flush=True)
    finally:hook.remove()
    if state.data.helpers().state_inventory(state.network)!=inventory:raise ValueError('Assessment changed checkpoint tensors')
    summary=[]
    for population in sorted({r['population'] for r in rows}):
        for passes in args.passes:
            group=[r for r in rows if r['population']==population and r['passes']==passes]
            summary.append(dict(population=population,passes=passes,cases=len(group),qualified=sum(r['qualified'] for r in group),
                worst_relative_residual=max(r['relative_residual'] for r in group),
                worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group),
                worst_gradient_relative_l2=max(r['gradient_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(r['median_policy_seconds'] for r in group)))
    result=dict(status='completed',entry=entry,rows=rows,summary=summary,source_evidence=evidence,
        source_sha256=sha(__file__),prior_sha256=sha(args.prior),population=args.population,
        measured_cases=len(samples),optimizer_updates=0,unchanged_tensor_inventory=True,elapsed_s=time.perf_counter()-started,
        peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wrapper_qualification=wrapper_qualification,
        inference_route='Qualified extended-shape wrapper with original C53 tensors' if wrapper else 'Exact captured model recurrence',
        helpers={name:sha(HERE/name) for name in ['cases.py','scalable.py']},cpu=args.cpu,threads=1,repetitions=args.repetitions,
        timing_scope='Prepared physical tensors through final independent fine residual; no loading, oracle metrics, IPC or simulation. First invocation and warm samples separate.',
        concurrency_scope=args.concurrency,blind_scenes_consumed=8 if args.population=='blind' else 0)
    if args.population=='blind':result['selection_sha256']=sha(ROOT/'docs/plans/pressure-methods-v2/selection.json')
    write(args.output/'results.json',result)
    print('Completed assessment',identifier,args.population,len(samples),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--population',choices=['development','blind'],required=True)
    parser.add_argument('--passes',type=int,nargs='+',choices=range(1,17),default=[8])
    parser.add_argument('--repetitions',type=int,choices=[0,1,3],default=0)
    parser.add_argument('--cpu',type=int,default=23)
    parser.add_argument('--concurrency',choices=['campaign-workers-active','owned-workers-quiet'],default='campaign-workers-active')
    args=parser.parse_args();args.prior=args.prior.resolve();args.output=args.output.resolve();assess(args)
