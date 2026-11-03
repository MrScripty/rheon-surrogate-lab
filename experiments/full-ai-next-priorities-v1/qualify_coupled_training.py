"""Actual coupled model constructors, temporal staging and own-history gradients."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from temporal_train_coupled import construct_network,construct_objective,inventory
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from temporal_objective import errors
from study_support import HERE,BASE,DOCS,read,write,sha


def config_for(horizons):
    view=BASE/'temporal-training-view-02';assets=BASE/'temporal-coupled-assets-01'
    return dict(horizons=horizons,width=12,architecture='coupled',supervision_mode='increments',
        step_scale_statistics_sha256=sha(BASE/'temporal-step-scale-statistics-01.json'),
        original_normalization_sha256=sha(view/'normalization.json'),physical_scales=read(view/'normalization.json')['scales'],
        expert_asset_sha256={str(p.relative_to(assets)):sha(p)for p in sorted(assets.rglob('*'))if p.is_file()})


def run(args):
    if args.output.exists():raise ValueError('Fresh coupled qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reg=read(DOCS/'temporal-coupled-registration-01.json')
    for name,digest in reg['model_source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered coupled model/trainer changed')
    operator=read(BASE/'full-learned-pressure-map-operator-audit-01.json')
    if operator['status']!='completed'or operator['observed_growing_residual_modes']:
        raise ValueError('Complete learned residual stability diagnostics required')
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',[view/row['path']for row in manifest['shards']],args.device)
    rows=[];aggregate={};initial_cold=[];common_initial=None
    for horizons in (1,2,3,4):
        torch.manual_seed(2026101921);config=config_for(horizons)
        network=construct_network(config,'cpu',BASE/'temporal-step-scale-statistics-01.json',BASE/'temporal-coupled-assets-01')
        initial=inventory(network)
        if common_initial is not None and initial!=common_initial:raise ValueError('Shared cell initialization differs across horizons')
        common_initial=initial
        with torch.no_grad():
            for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
                for cpm in (8,16):
                    parent=next(p for p in parents if p['scene']['counts']==counts and p['scene']['cells_per_metre']==cpm)
                    current=parent['states'][0:1].cpu();cache=make_features(parent,norm['scales'],horizons,args.device)
                    prediction=network(current,feature_at(cache,0,horizons).cpu())
                    target=parent['states'][1:horizons+1].unsqueeze(0).cpu()
                    metric=errors(prediction,target,counts)
                    initial_cold.append(dict(horizons=horizons,counts_xyz=counts,cells_per_metre=cpm,
                        field_errors={name:metric[name].sqrt().flatten().tolist()for name in ('pressure','equation','velocity','tracer')},
                        diagnostic_only=True,optimizer_updates=0))
            # Activate diagnostic copies; original fitting heads remain zero.
            for name,module in network.named_modules():
                if name.startswith(('pressure_core','flux')):continue
                if isinstance(module,torch.nn.Conv3d)and bool((module.weight==0).all()):module.weight.normal_(std=.001)
        cpu=copy.deepcopy(network).eval();network=network.to(args.device)
        objective_fn=construct_objective(config,network);before=inventory(network)
        trainable={name:p for name,p in network.named_parameters()if p.requires_grad};active={name:False for name in trainable}
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            for cpm in (8,16):
                parent=next(p for p in parents if p['scene']['counts']==counts and p['scene']['cells_per_metre']==cpm)
                cache=make_features(parent,norm['scales'],horizons,args.device)
                for start in (0,8):
                    current=parent['states'][start:start+1];feature=feature_at(cache,start,horizons)
                    target1=parent['states'][start+1:start+horizons+1].unsqueeze(0)
                    target2=parent['states'][start+horizons+1:start+2*horizons+1].unsqueeze(0)
                    observed={name:0 for name in ('encoder','local','pressure','flux','momentum','tracer','pressure_CNN')}
                    hooks=[]
                    for label,module in (('encoder',network.fluid),('local',network.local_update),('pressure',network.pressure_core),
                        ('flux',network.flux),('momentum',network.momentum),('tracer',network.tracer_readout),('pressure_CNN',network.pressure_core.source.fine)):
                        def count(*_,key=label):observed[key]+=1
                        hooks.append(module.register_forward_hook(count))
                    previous=None
                    for repeat in range(2):
                        first=network(current,feature);feedback=next_input(first,counts,norm['scales'])
                        if not feedback.requires_grad:raise ValueError('Deployed feedback detached')
                        second=network(feedback,feature_at(cache,start+horizons,horizons))
                        objective=(objective_fn(first,target1,counts,norm['scales']['tracer'])+
                            objective_fn(second,target2,counts,norm['scales']['tracer']))/2
                        if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite coupled objective')
                        network.zero_grad(set_to_none=True);objective.backward()
                        norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in trainable.items()}
                        if any(v is None or not np.isfinite(v)for v in norms.values()):raise ValueError('Finite trainable gradients required')
                        for name,value in norms.items():active[name]|=value>0
                        actual={name:p.grad.detach().cpu().numpy().tobytes()for name,p in trainable.items()}
                        if previous is not None and previous!=actual:raise ValueError('Repeated coupled gradients differ')
                        previous=actual
                        if any(p.grad is not None for p in network.parameters()if not p.requires_grad):raise ValueError('Frozen learned coefficients received parameter gradients')
                    for hook in hooks:hook.remove()
                    expected={name:4*horizons for name in ('local','pressure','flux','momentum','tracer')}
                    expected.update(encoder=4,pressure_CNN=8*horizons if max(counts)>16 else 0)
                    if observed!=expected:raise ValueError('Shared encoder or internal readout counts differ')
                    with torch.no_grad():
                        reference=cpu(current.cpu(),feature.cpu());difference=float((first.cpu()-reference).abs().max())
                        if difference>reg['CPU_device_max_abs_tolerance']:raise ValueError(f'CPU/device difference exceeded: {difference}')
                        changed=current.clone();changed[:,0]=123.;changed[:,4]=current[:,4]*-.25+17.
                        modified=network(changed,feature)
                        if not torch.equal(first[:,:,:4],modified[:,:,:4]):raise ValueError('Tracer/previous pressure influences fluid')
                        altered=feature.clone();altered[:,7:]=0.;unforced=network(current,altered)
                        if torch.equal(first[:,:,:4],unforced[:,:,:4]):raise ValueError('Forcing ignored by fluid')
                        if not torch.equal(first[:,0,4],unforced[:,0,4]):raise ValueError('First tracer uses current or future forcing')
                        resting=current.clone();resting[:,1:4]=0.;rest=network(resting,feature)
                        if not torch.equal(rest[:,0,4],resting[:,4]):raise ValueError('Zero entering velocity changes first tracer')
                        zero=current.clone();zero[:,4]=0.;zero_first=network(zero,feature)
                        zero_second=network(next_input(zero_first,counts,norm['scales']),feature_at(cache,start+horizons,horizons))
                        if torch.count_nonzero(zero_first[:,:,4])or torch.count_nonzero(zero_second[:,:,4]):raise ValueError('Zero tracer must remain exact across two blocks')
                        constant=current.clone();constant[:,4]=3.25;constant_out=network(constant,feature)
                        constant_error=float((constant_out[:,:,4]-3.25).abs().max())/3.25
                        if constant_error>reg['constant_tracer_relative_tolerance']:raise ValueError('Constant tracer exceeds float32 interchange tolerance')
                        other=current.clone();other[:,4]=current[:,4].flip(dims=(-1,));other_out=network(other,feature)
                        mixed=current.clone();mixed[:,4]=.3*current[:,4]+.7*other[:,4];mixed_out=network(mixed,feature)
                        scale=max(1.,float(first[:,:,4].abs().max()),float(other_out[:,:,4].abs().max()))
                        linear_error=float((mixed_out[:,:,4]-(.3*first[:,:,4]+.7*other_out[:,:,4])).abs().max())/scale
                        if linear_error>reg['tracer_linearity_relative_tolerance']:raise ValueError('Tracer linearity exceeds interchange tolerance')
                        lower=float(current[:,4].min());upper=float(current[:,4].max());tol=reg['tracer_bounds_relative_tolerance']*max(1.,abs(lower),abs(upper))
                        if float(first[:,:,4].min())<lower-tol or float(first[:,:,4].max())>upper+tol:raise ValueError('Tracer left donor range')
                    if inventory(network)!=before:raise ValueError('Zero-update qualification changed coefficients')
                    rows.append(dict(architecture='coupled',horizons=horizons,counts_xyz=counts,cells_per_metre=cpm,start=start,
                        objective=float(objective.detach()),parameter_gradient_norms=norms,CPU_device_max_abs=difference,
                        counts_per_block={name:value//4 for name,value in observed.items()},deterministic_two_block_gradients=True,
                        first_tracer_force_independent=True,zero_entering_velocity_first_tracer_exact=True,zero_tracer_two_blocks_exact=True,
                        constant_tracer_relative_difference=constant_error,tracer_linearity_relative_difference=linear_error,
                        tracer_within_donor_range=True,optimizer_updates=0))
        if not all(active.values()):raise ValueError('Some trainable tensors never receive active gradients')
        aggregate[str(horizons)]=dict(all_trainable_tensors_active=True,initial_inventory=initial,
            parameters=sum(p.numel()for p in network.parameters()),trainable_parameters=sum(p.numel()for p in trainable.values()))
        print('Qualified coupled K',horizons,'both densities, four grids, staging and two-block gradients',flush=True)
    files=('bundle_model_coupled.py','full_learned_pressure_map.py','frozen_temporal_pressure_core.py','tied_momentum_readout.py',
        'bundle_model_decoder.py','bundle_model_pressure_expert.py','factored_request_model.py','temporal_train_coupled.py',
        'temporal_objective_increment.py','temporal_objective.py','temporal_objective_mse.py','temporal_features_v2.py',
        'temporal_feedback.py','temporal_data.py','temporal_codec.py')
    write(args.output,dict(status='passed',device=args.device,rows=rows,aggregate=aggregate,initial_cold_diagnostics=initial_cold,
        optimizer_updates=0,feedback_protocol=2,actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,
        source_sha256={n:sha(HERE/n)for n in files},qualification_source_sha256=sha(__file__),
        registration_sha256=sha(DOCS/'temporal-coupled-registration-01.json'),
        step_statistics_sha256=sha(BASE/'temporal-step-scale-statistics-01.json'),expert_asset_sha256=config_for(2)['expert_asset_sha256'],
        shared_initial_parameter_tensors_across_horizons=True,numerical_pressure_solver_called=False,candidate_post_projection=False,
        coupled_learned_flux_pressure_and_momentum=True,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if not args.output.exists():write(args.output,dict(status='failed',device=args.device,error=type(error).__name__+': '+str(error),source_sha256=sha(__file__),optimizer_updates=0))
        raise
