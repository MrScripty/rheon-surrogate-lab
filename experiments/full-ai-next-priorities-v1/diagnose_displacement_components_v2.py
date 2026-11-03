"""Train-only transport/viscosity controls; numerical primitives stay offline."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from displacement_evaluation_support import load_model
from temporal_train_displacement import inventory
from training_displacement_teacher import displacement_labels
from temporal_displacement_objective import point_mask
from viscosity_teacher_support_v2 import native,from_carrier,packed_output
from temporal_metrics import metrics,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def relative(a,b,floor=1e-30):
    return float((a.double()-b.double()).norm())/max(float(b.double().norm()),floor)


def run(args):
    if args.output.exists():raise ValueError('Fresh component diagnosis required')
    protocol_path=DOCS/'displacement-component-diagnostic-protocol-02.json';reg=read(protocol_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered component source changed')
    qualification_path=BASE/'viscosity-teacher-qualification-02.json';qualification=read(qualification_path)
    if qualification['status']!='passed'or qualification['protocol_sha256']!=sha(protocol_path):raise ValueError('Qualified viscosity primitive required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    network,norm,identity=load_model(args.pair.resolve());before=inventory(network)
    original_viscosity=network.viscous_head.weight.detach().clone()
    previous=read(BASE/'temporal-displacement-fitting-diagnosis-01.json')
    expected={(r['identifier'],r['start'],r['horizon']):r['learned']for r in previous['rows']if r['split']=='train'}
    rows=[];intrinsic=[];replays=[];sources={}

    def predict(mode,fields,scene,start,observe=False):
        counts=scene['counts'];hooks=[];pending=[];local=[]
        def shifts(module,arguments,output):
            if observe or mode in ('offline_native_displacement','offline_both'):
                velocity,hidden,alpha=arguments
                target,_=displacement_labels((velocity*norm['scales']['velocity']).float(),counts,alpha.double()/norm['scales']['velocity'])
                if observe:
                    mask=point_mask(counts,'cpu')
                    difference=(output.double()-target)*mask
                    local.append(dict(displacement_absolute_RMS_cells=float((difference.square().sum()/(3*mask.sum())).sqrt()),
                        displacement_error_in_alpha_squared_units=float(((difference/alpha.double().square().unsqueeze(1)).square().sum()/(3*mask.sum())).sqrt())))
                if mode in ('offline_native_displacement','offline_both'):return target
            return output
        def viscosity(module,arguments,output):
            if observe or mode in ('offline_native_viscosity','offline_both'):
                entering=(arguments[0]*norm['scales']['velocity']).float()
                case=from_carrier(entering,counts,scene['cells_per_metre'])
                reference=native([case])[0];truth=packed_output(reference,counts)
                closed=packed_output(dict(updated_velocity=case['velocity']),counts)
                pending.append((truth,closed))
        def provisional(module,arguments):
            if observe or mode in ('offline_native_viscosity','offline_both'):
                truth,entering=pending.pop(0)
                if observe:
                    difference=arguments[0]-truth
                    increment=truth-entering
                    local[len(local)-1].update(provisional_relative_error_vs_native_viscosity=relative(arguments[0],truth),
                        viscosity_error_in_native_increment_units=float(difference.double().norm())/max(float(increment.double().norm()),1e-8*float(truth.double().norm()),1e-30))
                if mode in ('offline_native_viscosity','offline_both'):arguments[0].copy_(truth)
        hooks.append(network.displacement.register_forward_hook(shifts))
        hooks.append(network.viscous_head.register_forward_hook(viscosity))
        hooks.append(network.flux.register_forward_pre_hook(provisional))
        try:
            if mode=='zero_viscosity':network.viscous_head.weight.zero_()
            outputs=network.predict(fields,scene,start,norm)
            if len(outputs)!=2 or pending:raise ValueError('Every causal horizon and native control must be consumed')
            return outputs,local
        finally:
            for hook in hooks:hook.remove()
            network.viscous_head.weight.copy_(original_viscosity)

    with torch.inference_mode():
        for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
            if row['split']!='train':continue
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(receipt_path);scene=receipt['scene'];sources[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
            cache={}
            def state(step):
                if step not in cache:
                    spec=receipt['states'][step];path=receipt_path.parent/spec['path']
                    if sha(path)!=spec['sha256']:raise ValueError('Training native field changed')
                    sources[str(path.relative_to(ROOT))]=sha(path)
                    with np.load(path,allow_pickle=False)as a:cache[step]={n:a[n].copy()for n in (*FIELDS,'rhs')}
                return cache[step]
            for start in reg['starts']:
                current=state(start);initial={n:current[n]for n in FIELDS}
                for mode in reg['modes']:
                    outputs,parts=predict(mode,initial,scene,start,observe=mode=='normal')
                    for horizon,output in enumerate(outputs,1):
                        target=state(start+horizon);values=metrics(output,target,target['rhs'],scene,norm,0.)['relative']
                        if mode=='normal'and values!=expected[(row['identifier'],start,horizon)]:raise ValueError('Normal metrics differ from fixed fitting diagnosis')
                        rows.append(dict(identifier=row['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                            start=start,horizon=horizon,mode=mode,relative=values))
                    intrinsic.extend(dict(identifier=row['identifier'],start=start,horizon=i+1,**p)for i,p in enumerate(parts))
            if row['identifier']in reg['replay_training_identifiers']:
                for mode in reg['replay_modes']:
                    current={n:state(0)[n]for n in FIELDS};trace=[]
                    for start in range(0,128,2):
                        outputs,_=predict(mode,current,scene,start)
                        for horizon,output in enumerate(outputs,1):
                            target=state(start+horizon)
                            trace.append(dict(step=start+horizon,relative=metrics(output,target,target['rhs'],scene,norm,0.)['relative']))
                        current=outputs[-1]
                    replays.append(dict(identifier=row['identifier'],mode=mode,steps=128,rows=trace,
                        worst={f:max(r['relative'][f]for r in trace)for f in ('pressure','equation','velocity','tracer')}))
    if inventory(network)!=before:raise ValueError('Offline controls changed candidate coefficients')
    summaries=[]
    for phase in ('cold','warm'):
        for mode in reg['modes']:
            selected=[r for r in rows if(r['start']==0)==(phase=='cold')and r['mode']==mode]
            for field in ('pressure','equation','velocity','tracer'):
                values=[r['relative'][field]for r in selected]
                summaries.append(dict(phase=phase,mode=mode,field=field,median=statistics.median(values),worst=max(values),outputs=len(values)))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,intrinsic=intrinsic,replays=replays,
        source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=sha(protocol_path),
        native_qualification_sha256=sha(qualification_path),normal_metrics_bitwise_equal_to_prior=True,candidate_coefficients_unchanged=True,
        optimizer_updates=0,training_queries_only=True,offline_128step_controls_not_240step_goal=True,
        numerical_primitives_excluded_from_candidate_inference=True,candidate_frontier_point=False,CPU_latency_claim=False,reserved_scenes_consumed=False))
    print('Completed offline displacement/viscosity separation:',len(rows),'fixed outputs;',len(replays),'training-only128-step replays',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
