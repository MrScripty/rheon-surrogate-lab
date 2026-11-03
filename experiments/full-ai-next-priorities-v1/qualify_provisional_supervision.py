"""Qualify actual continuation constructors, deployed feedback and target labels."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from temporal_train_provisional import construct_network,construct_objective,inventory
from temporal_provisional_objective import provisional_labels
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from provisional_teacher import native,packet
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh actual-constructor qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=DOCS/'temporal-provisional-registration-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered continuation source changed')
    inputs=BASE/'temporal-provisional-qualification-inputs-01';view=BASE/'temporal-training-view-02'
    manifest=read(view/'manifest.json');shards=[view/r['path']for r in manifest['shards']]
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',shards,'cpu')
    parent=BASE/'b2-coupled-pair-01/restore/weights.npz';assets=BASE/'temporal-coupled-assets-01';stats=BASE/'temporal-step-scale-statistics-01.json'
    if sha(parent)!=reg['continuation_weights_sha256']:raise ValueError('Registered parent changed')
    configs={mode:read(inputs/(mode+'-config.json'))for mode in ('control','aux')}
    for mode,config in configs.items():
        if sha(inputs/(mode+'-config.json'))!=reg['config_sha256'][mode]:raise ValueError('Registered config changed')
    networks={mode:construct_network(config,args.device,stats,assets,parent)for mode,config in configs.items()}
    if inventory(networks['control'])!=inventory(networks['aux']):raise ValueError('Matched continuation tensors differ')
    before={mode:inventory(n)for mode,n in networks.items()};objectives={m:construct_objective(configs[m],n)for m,n in networks.items()}
    caches=[make_features(p,norm['scales'],2,'cpu')for p in parents]
    rows=[];label_rows=[];source_evidence={};pending=[]
    bank={r['identifier']:r for r in read(BASE/'temporal-data-01/bank.json')['trajectories']if r['split']=='train'}
    with torch.no_grad():
        for parent_index,p in enumerate(parents):
            record=bank[p['identifier']];path=ROOT/record['receipt']
            if sha(path)!=record['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(path);source_evidence[str(path.relative_to(ROOT))]=sha(path)
            for start in (0,8,63,120):
                target=p['states'][start+1:start+3].unsqueeze(0).to(args.device)
                features=feature_at(caches[parent_index],start,2).to(args.device)
                labels=provisional_labels(target,features,p['scene']['counts'],norm['scales'],networks['aux'].flux.weight)
                cpu=provisional_labels(target.cpu(),features.cpu(),p['scene']['counts'],norm['scales'],networks['aux'].flux.weight.cpu())
                parity=float((labels.cpu()-cpu).norm())/max(float(cpu.norm()),1e-30)
                if parity>reg['label_CPU_device_relative_tolerance']:raise ValueError('Training label CPU/device parity failed')
                for horizon in (1,2):
                    spec=receipt['states'][start+horizon];state_path=path.parent/spec['path']
                    if sha(state_path)!=spec['sha256']:raise ValueError('Native training target changed')
                    source_evidence[str(state_path.relative_to(ROOT))]=sha(state_path)
                    with np.load(state_path,allow_pickle=False)as arrays:fields={n:arrays[n].copy()for n in ('pressure','vx','vy','vz','rhs')}
                    pending.append((packet(labels[horizon-1:horizon].cpu(),fields,p['scene']),fields,
                        dict(identifier=p['identifier'],start=start,horizon=horizon,CPU_device_label_relative_difference=parity)))
        for start in range(0,len(pending),16):
            cases=pending[start:start+16];actual=native([c[0]for c in cases])
            for (_,fields,identity),observed in zip(cases,actual):
                velocity=np.concatenate([fields[n].astype('f8')for n in ('vx','vy','vz')])
                out=np.concatenate([np.asarray(v,dtype='f8')for v in observed['corrected_velocity']])
                error=float(np.linalg.norm(out-velocity))/max(float(np.linalg.norm(velocity)),1e-30)
                rhs=np.asarray(observed['rhs'],dtype='f8');rhs_error=float(np.linalg.norm(rhs-fields['rhs']))/max(float(np.linalg.norm(fields['rhs'])),1e-30)
                if error>reg['training_label_velocity_relative_tolerance']or rhs_error>reg['training_label_RHS_relative_tolerance']:
                    raise ValueError(f'Actual normalized float32 training label differs from native: velocity={error} RHS={rhs_error}')
                label_rows.append(dict(**identity,velocity_roundtrip_relative_error=error,native_RHS_relative_error=rhs_error))
    # Exercise the exact future trainer/objective constructors on all geometries
    # and densities, including two deployed own-history blocks and repeat grads.
    selected=[];seen=set()
    for index,p in enumerate(parents):
        key=tuple(p['scene']['counts']),p['scene']['cells_per_metre']
        if key not in seen:selected.append(index);seen.add(key)
    for mode,network in networks.items():
        objective=objectives[mode]
        for index in selected:
            p=parents[index];counts=p['scene']['counts'];state=p['states'].to(args.device)
            for start in (0,8):
                gradients=[];losses=[];before_labels=objective.labels_consumed
                for repeat in range(2):
                    network.zero_grad(set_to_none=True)
                    first=network(state[start:start+1],feature_at(caches[index],start,2).to(args.device))
                    loss1=objective(first,state[start+1:start+3].unsqueeze(0),counts,norm['scales']['tracer'])
                    if mode=='control'and float(loss1)!=float(objective.last_terms['full_state']):raise ValueError('Control changed original full-state scalar')
                    second=network(next_input(first,counts,norm['scales']),feature_at(caches[index],start+2,2).to(args.device))
                    loss2=objective(second,state[start+3:start+5].unsqueeze(0),counts,norm['scales']['tracer'])
                    total=(loss1+loss2)/2
                    if not bool(torch.isfinite(total)):raise ValueError('Nonfinite qualified objective')
                    total.backward();g={}
                    for name,param in network.named_parameters():
                        if param.requires_grad:
                            if param.grad is None or not bool(torch.isfinite(param.grad).all()):raise ValueError('Missing finite actual gradient')
                            g[name]=param.grad.detach().cpu().clone()
                        elif param.grad is not None:raise ValueError('Frozen expert acquired a gradient')
                    gradients.append(g);losses.append(float(total))
                if losses[0]!=losses[1]or any(not torch.equal(gradients[0][n],gradients[1][n])for n in gradients[0]):
                    raise ValueError('Actual two-block gradients are not repeatable')
                if objective.labels_consumed-before_labels!=8:raise ValueError('Every physical label must be counted in both arms')
                rows.append(dict(mode=mode,architecture='coupled',horizons=2,counts_xyz=counts,cells_per_metre=p['scene']['cells_per_metre'],
                    start=start,objective=losses[0],deterministic_two_block_gradients=True,
                    parameter_gradient_norms={n:float(g.norm())for n,g in gradients[0].items()},labels_consumed=8,optimizer_updates=0))
        objective.close()
        if inventory(network)!=before[mode]:raise ValueError('Qualification changed continuation coefficients')
    active={name:any(r['parameter_gradient_norms'][name]>0 for r in rows)for name in rows[0]['parameter_gradient_norms']}
    if not all(active.values()):raise ValueError('Some trainable tensors have no active qualified gradient')
    control=[r for r in rows if r['mode']=='control'];aux=[r for r in rows if r['mode']=='aux']
    if not any(a['parameter_gradient_norms']!=b['parameter_gradient_norms']for a,b in zip(control,aux)):
        raise ValueError('Auxiliary objective did not affect model gradients')
    write(args.output,dict(status='passed',device=args.device,rows=rows,training_label_rows=label_rows,
        maximum_training_label_native_RHS_relative_error=max(r['native_RHS_relative_error']for r in label_rows),
        maximum_training_label_velocity_roundtrip_relative_error=max(r['velocity_roundtrip_relative_error']for r in label_rows),
        actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,
        matched_continuation_initial_inventory=before['control'],gradient_active_tensors=active,
        feedback_protocol=2,step_statistics_sha256=sha(stats),expert_asset_sha256=configs['control']['expert_asset_sha256'],
        config_sha256=reg['config_sha256'],registration_sha256=sha(protocol),source_sha256=reg['source_sha256'],source_evidence=source_evidence,
        optimizer_updates=0,coupled_learned_flux_pressure_and_momentum=True,numerical_pressure_solver_called=False,candidate_post_projection=False,
        inference_model_source_unchanged=True,training_labels_never_predictor_inputs=True,
        candidate_frontier_point=False,CPU_latency_claim=False,reserved_scenes_consumed=False))
    print('Qualified actual continuation trainer/objective:',args.device,len(rows),'gradient cases and',len(label_rows),'native label cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
