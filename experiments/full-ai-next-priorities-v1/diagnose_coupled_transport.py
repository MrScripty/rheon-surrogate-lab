"""Train-only ablations and qualified provisional-label ceilings; no new fit."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from coupled_evaluation_support import load_model
from temporal_train_coupled import inventory,load_weights
from temporal_features_v2 import force_features
from temporal_metrics import metrics,FIELDS
from provisional_teacher import reconstructed_provisional,padded_velocity,vectors
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def relative(a,b):return float(torch.linalg.vector_norm((a-b).double()))/max(float(torch.linalg.vector_norm(b.double())),1e-30)


def run(args):
    if args.output.exists():raise ValueError('Fresh coupled transport diagnosis required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    protocol=DOCS/'temporal-coupled-transport-diagnostic-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered diagnostic source changed')
    qualification_path=BASE/'provisional-teacher-qualification-01.json';qualification=read(qualification_path)
    if qualification['status']!='passed'or qualification['protocol_sha256']!=sha(protocol):raise ValueError('Qualified offline provisional labels required')
    network,norm,identity=load_model(args.pair.resolve());before=inventory(network)
    if network.horizons!=2:raise ValueError('Registered K2 comparison required')
    original={name:p.detach().clone()for name,p in network.named_parameters()}
    restored=read(args.pair/'restore/receipt.json');initial_path=args.pair/'restore/initial-weights.npz'
    if sha(initial_path)!=restored['outputs']['initial-weights.npz']['sha256']:raise ValueError('Managed initial artifact changed')
    load_weights(network,initial_path,restored['initial_inventory'])
    initial={name:p.detach().clone()for name,p in network.named_parameters()}
    def restore(values):
        for name,p in network.named_parameters():p.copy_(values[name])
    previous=read(BASE/'temporal-coupled-fitting-diagnosis-01.json')
    expected={(r['identifier'],r['start'],r['horizon']):r['learned']for r in previous['rows']if r['split']=='train'}
    rows=[];transport=[];sources={}
    with torch.inference_mode():
        restore(original)
        for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
            if row['split']!='train':continue
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(receipt_path);scene=receipt['scene'];counts=scene['counts'];nx,ny,nz=counts
            sources[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
            def state(step):
                spec=receipt['states'][step];path=receipt_path.parent/spec['path']
                if sha(path)!=spec['sha256']:raise ValueError('Native reference changed')
                sources[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path,allow_pickle=False)as arrays:return {n:arrays[n].copy()for n in (*FIELDS,'rhs')}
            for start in reg['starts']:
                current=state(start);physical={n:current[n]for n in FIELDS};targets=[state(start+h)for h in (1,2)]
                teacher=[reconstructed_provisional(t,scene,network.flux.weight).float()for t in targets]
                observed=[];requests=[]
                hook=network.flux.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
                pressure_hook=network.pressure_core.register_forward_pre_hook(lambda module,args:requests.append(args[0].detach().clone()))
                try:normal=network.predict(physical,scene,start,norm)
                finally:hook.remove();pressure_hook.remove()
                modes={'normal':normal}
                try:
                    restore(initial);modes['saved_initialization']=network.predict(physical,scene,start,norm)
                finally:restore(original)
                for mode,names in [('zero_transport',('transport_head',)),('zero_viscosity',('viscous_head',)),('zero_both',('transport_head','viscous_head'))]:
                    try:
                        for name in names:
                            for p in getattr(network,name).parameters():p.zero_()
                        modes[mode]=network.predict(physical,scene,start,norm)
                    finally:restore(original)
                call=[0]
                def oracle_pressure(module,args,output):
                    p=torch.as_tensor(targets[call[0]]['pressure']).reshape(1,1,nz,ny,nx).to(output)
                    call[0]+=1;return p-p.flatten()[0]
                hook=network.pressure_core.register_forward_hook(oracle_pressure)
                try:modes['offline_native_pressure']=network.predict(physical,scene,start,norm)
                finally:hook.remove()
                if call[0]!=2:raise ValueError('Every pressure horizon must be counted')
                call=[0]
                def oracle_provisional(module,args):
                    args[0].copy_(teacher[call[0]].to(args[0]));call[0]+=1
                hook=network.flux.register_forward_pre_hook(oracle_provisional)
                try:modes['offline_native_provisional']=network.predict(physical,scene,start,norm)
                finally:hook.remove()
                if call[0]!=2:raise ValueError('Every provisional horizon must be counted')
                forces=force_features(scene,start,2,1.,'cpu').reshape(2,3,*observed[0].shape[2:])
                for horizon,target in enumerate(targets,1):
                    for mode,predictions in modes.items():
                        values=metrics(predictions[horizon-1],target,target['rhs'],scene,norm,0.)['relative']
                        if mode=='normal'and values!=expected[(row['identifier'],start,horizon)]:raise ValueError('Unchanged candidate differs from prior fitting diagnostic')
                        rows.append(dict(identifier=row['identifier'],counts_xyz=counts,start=start,horizon=horizon,mode=mode,relative=values))
                    native_entering=padded_velocity(current if horizon==1 else targets[0],counts)
                    desired=teacher[horizon-1].double()-native_entering-forces[horizon-1:horizon].double()
                    difference=observed[horizon-1].double()-teacher[horizon-1].double()
                    q=torch.as_tensor(target['rhs']).double().reshape(1,1,nz,ny,nx)*(1000*scene['cells_per_metre'])
                    q=q-q.mean()
                    transport.append(dict(identifier=row['identifier'],start=start,horizon=horizon,
                        provisional_relative_error=relative(observed[horizon-1],teacher[horizon-1]),
                        provisional_error_in_native_transport_units=float(difference.norm())/max(float(desired.norm()),1e-8*float(teacher[horizon-1].norm()),1e-30),
                        native_pressure_request_relative_error=relative(requests[horizon-1],q)))
        restore(original)
    if inventory(network)!=before:raise ValueError('Counterfactual diagnosis modified candidate coefficients')
    summaries=[];transport_summaries=[]
    for phase in ('cold','warm'):
        for mode in reg['modes']:
            selected=[r for r in rows if(r['start']==0)==(phase=='cold')and r['mode']==mode]
            for field in ('pressure','equation','velocity','tracer'):
                values=[r['relative'][field]for r in selected]
                summaries.append(dict(phase=phase,mode=mode,field=field,outputs=len(values),median=statistics.median(values),worst=max(values)))
        selected=[r for r in transport if(r['start']==0)==(phase=='cold')]
        for field in ('provisional_relative_error','provisional_error_in_native_transport_units','native_pressure_request_relative_error'):
            values=[r[field]for r in selected]
            transport_summaries.append(dict(phase=phase,field=field,median=statistics.median(values),worst=max(values)))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,transport=transport,
        transport_summaries=transport_summaries,normal_metrics_bitwise_equal_to_prior=True,candidate_coefficients_unchanged=True,
        protocol_sha256=sha(protocol),teacher_qualification_sha256=sha(qualification_path),source_sha256=reg['source_sha256'],
        source_evidence=sources,training_queries_only=True,optimizer_updates=0,
        native_future_data_only_in_offline_controls=True,candidate_frontier_point=False,CPU_latency_claim=False,
        reserved_scenes_consumed=False,original_goal_complete=False))
    print('Completed train-only transport diagnosis:',len(rows),'outputs; fitted candidate unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
