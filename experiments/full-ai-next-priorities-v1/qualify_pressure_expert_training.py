"""Qualify actual full-state expert constructors and two deployed feedback blocks."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from temporal_train_pressure_expert import construct_network,construct_objective,inventory
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from study_support import HERE,BASE,DOCS,read,write,sha


def config_for(horizons,architecture):
    view=BASE/'temporal-training-view-02';assets=BASE/'temporal-pressure-expert-assets-01'
    return dict(horizons=horizons,width=16,architecture=architecture,supervision_mode='increments',
        step_scale_statistics_sha256=sha(BASE/'temporal-step-scale-statistics-01.json'),
        original_normalization_sha256=sha(view/'normalization.json'),physical_scales=read(view/'normalization.json')['scales'],
        expert_asset_sha256={str(p.relative_to(assets)):sha(p)for p in sorted(assets.rglob('*'))if p.is_file()})


def run(args):
    if args.output.exists():raise ValueError('Fresh actual-constructor qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registration=read(DOCS/'temporal-pressure-expert-registration-01.json')
    for name,digest in registration['model_source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Model/trainer changed after registration')
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',[view/row['path']for row in manifest['shards']],args.device)
    rows=[];aggregate={}
    for architecture in ('endpoint','reuse'):
        for horizons in (2,3,4):
            torch.manual_seed(2026101921);config=config_for(horizons,architecture)
            network=construct_network(config,'cpu',BASE/'temporal-step-scale-statistics-01.json',BASE/'temporal-pressure-expert-assets-01')
            initial=inventory(network)
            # Diagnostic copies activate new heads; seeded training initialization
            # remains zero for those heads and retains managed request weights.
            with torch.no_grad():
                for name,module in network.named_modules():
                    if name.startswith(('pressure_core','request.')):continue
                    if isinstance(module,torch.nn.Conv3d)and bool((module.weight==0).all()):module.weight.normal_(std=.001)
            cpu=copy.deepcopy(network).eval();network=network.to(args.device)
            objective_fn=construct_objective(config,network)
            before=inventory(network);trainable={name:p for name,p in network.named_parameters()if p.requires_grad}
            total_active={name:False for name in trainable}
            for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
                parent=next(p for p in parents if p['scene']['counts']==counts)
                cache=make_features(parent,norm['scales'],horizons,args.device)
                for start in (0,8):
                    current=parent['states'][start:start+1];feature=feature_at(cache,start,horizons)
                    target1=parent['states'][start+1:start+horizons+1].unsqueeze(0)
                    target2=parent['states'][start+horizons+1:start+2*horizons+1].unsqueeze(0)
                    observed=dict(encoder=0,expert=0)
                    def count_encoder(*_):observed['encoder']+=1
                    def count_expert(*_):observed['expert']+=1
                    hooks=[network.fluid.register_forward_hook(count_encoder),network.pressure_core.register_forward_hook(count_expert)]
                    previous=None
                    for repeat in range(2):
                        first=network(current,feature);feedback=next_input(first,counts,norm['scales'])
                        if not feedback.requires_grad:raise ValueError('Deployed own-history feedback detached')
                        second=network(feedback,feature_at(cache,start+horizons,horizons))
                        objective=(objective_fn(first,target1,counts,norm['scales']['tracer'])+
                                   objective_fn(second,target2,counts,norm['scales']['tracer']))/2
                        if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite expert objective')
                        network.zero_grad(set_to_none=True);objective.backward()
                        norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in trainable.items()}
                        if any(value is None or not np.isfinite(value)for value in norms.values()):raise ValueError('Finite trainable gradients required')
                        for name,value in norms.items():total_active[name]|=value>0
                        actual={name:p.grad.detach().cpu().numpy().tobytes()for name,p in trainable.items()}
                        if previous is not None and previous!=actual:raise ValueError('Repeated two-block gradients differ')
                        previous=actual
                        if any(p.grad is not None for p in network.parameters()if not p.requires_grad):
                            raise ValueError('Frozen expert received parameter gradients')
                    for hook in hooks:hook.remove()
                    if observed!={'encoder':4,'expert':4}:raise ValueError('Exactly one fluid encoder and pressure expert per block required')
                    with torch.no_grad():
                        reference=cpu(current.cpu(),feature.cpu());difference=float((first.cpu()-reference).abs().max())
                        if difference>1e-5:raise ValueError(f'CPU/device full-state parity exceeded tolerance: {difference}')
                        changed=current.clone();changed[:,0]=123.;changed[:,4]=current[:,4]*-.25+17.
                        modified=network(changed,feature)
                        if not torch.equal(first[:,:,:4],modified[:,:,:4]):raise ValueError('Tracer/previous pressure influences fluid outputs')
                        if torch.equal(first[:,:,4],modified[:,:,4]):raise ValueError('Tracer output ignores current tracer')
                        altered=feature.clone();altered[:,7:]=0
                        if torch.equal(first,network(current,altered)):raise ValueError('Known forcing ignored')
                    if inventory(network)!=before:raise ValueError('Zero-update qualification changed coefficients')
                    rows.append(dict(architecture=architecture,horizons=horizons,counts_xyz=counts,start=start,
                        loss=float(objective.detach()),parameter_gradient_norms=norms,CPU_device_max_abs=difference,
                        encoder_calls_per_block=1,pressure_expert_calls_per_block=1,deterministic_two_block_gradients=True,
                        pressure_and_tracer_independent_fluid=True,optimizer_updates=0))
            if not all(total_active.values()):raise ValueError('Some trainable tensors never receive an active gradient')
            aggregate[f'{horizons}-{architecture}']=dict(all_trainable_tensors_active=True,parameters=sum(p.numel()for p in network.parameters()),
                trainable_parameters=sum(p.numel()for p in trainable.values()),initial_inventory=initial)
            print(f'Qualified K{horizons} {architecture}, four grids, cold/warm and two-block gradients',flush=True)
    files=('bundle_model_pressure_expert.py','frozen_temporal_pressure_core.py','factored_request_model.py',
        'temporal_train_pressure_expert.py','temporal_objective_increment.py','temporal_objective.py','temporal_objective_mse.py',
        'temporal_features_v2.py','temporal_feedback.py','temporal_data.py','temporal_codec.py')
    write(args.output,dict(status='passed',device=args.device,rows=rows,aggregate=aggregate,optimizer_updates=0,feedback_protocol=2,
        actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,
        source_sha256={name:sha(HERE/name)for name in files},qualification_source_sha256=sha(__file__),
        registration_sha256=sha(DOCS/'temporal-pressure-expert-registration-01.json'),
        step_statistics_sha256=sha(BASE/'temporal-step-scale-statistics-01.json'),expert_asset_sha256=config_for(2,'endpoint')['expert_asset_sha256'],
        reserved_scenes_consumed=False,candidate_numerical_pressure_solve=False,candidate_velocity_post_projection=False,
        learned_pressure_conditioned_velocity_readout=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if not args.output.exists():write(args.output,dict(status='failed',device=args.device,error=type(error).__name__+': '+str(error),
            source_sha256=sha(__file__),optimizer_updates=0))
        raise
