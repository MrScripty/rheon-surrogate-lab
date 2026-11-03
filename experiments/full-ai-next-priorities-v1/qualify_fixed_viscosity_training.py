"""Actual full-state displacement constructors, gradients and causal staging."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import temporal_displacement_objective as teaching
from temporal_train_fixed_viscosity import construct_network,construct_objective,inventory
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from temporal_objective import errors
from study_support import HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh displacement constructor qualification required')
    reg_path=DOCS/'fixed-viscosity-qualification-protocol-01.json';reg=read(reg_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered displacement source changed')
    inputs=BASE/'fixed-viscosity-inputs-01'
    if sha(inputs/'receipt.json')!=reg['inputs_receipt_sha256']:raise ValueError('Qualified component inputs changed')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',[view/r['path']for r in manifest['shards']],args.device)
    rows=[];initial_rows=[];aggregate={};common_initial=None
    for k in (1,2,3,4):
        config=read(inputs/f'k{k}-config.json');torch.manual_seed(config['seed'])
        network=construct_network(config,'cpu',BASE/'temporal-step-scale-statistics-01.json',inputs/'expert-assets')
        initial=inventory(network)
        if network.viscous_head.weight.requires_grad:raise ValueError('Calibrated viscosity must be frozen from initialization')
        if common_initial is not None and initial!=common_initial:raise ValueError('Shared-cell initialization differs across K')
        common_initial=initial
        with torch.no_grad():
            for counts in reg['counts_xyz']:
                for cpm in (8,16):
                    parent=next(p for p in parents if p['scene']['counts']==counts and p['scene']['cells_per_metre']==cpm)
                    cache=make_features(parent,norm['scales'],k,'cpu')
                    for start in (0,8):
                        pred=network(parent['states'][start:start+1].cpu(),feature_at(cache,start,k))
                        truth=parent['states'][start+1:start+k+1].unsqueeze(0).cpu()
                        metric=errors(pred,truth,counts)
                        initial_rows.append(dict(horizons=k,counts_xyz=counts,cells_per_metre=cpm,start=start,
                            field_errors={name:metric[name].sqrt().flatten().tolist()for name in ('pressure','equation','velocity','tracer')},
                            diagnostic_only=True,optimizer_updates=0))
            # Activate diagnostic copies only, so every trainable path is exercised.
            for name,module in network.named_modules():
                if name.startswith(('pressure_core','flux')):continue
                if isinstance(module,torch.nn.Conv3d)and bool((module.weight==0).all()):module.weight.normal_(std=.001)
        cpu=copy.deepcopy(network).eval();network=network.to(args.device);before=inventory(network)
        trainable={name:p for name,p in network.named_parameters()if p.requires_grad};active={n:False for n in trainable}
        for counts in reg['counts_xyz']:
            for cpm in (8,16):
                parent=next(p for p in parents if p['scene']['counts']==counts and p['scene']['cells_per_metre']==cpm)
                cache=make_features(parent,norm['scales'],k,args.device)
                for start in (0,8):
                    current=parent['states'][start:start+1];features=feature_at(cache,start,k)
                    target1=parent['states'][start+1:start+k+1].unsqueeze(0)
                    target2=parent['states'][start+k+1:start+2*k+1].unsqueeze(0)
                    observed={name:0 for name in ('encoder','local','displacement','pressure','flux','momentum')};hooks=[]
                    for label,module in (('encoder',network.fluid),('local',network.local_update),('displacement',network.displacement),
                        ('pressure',network.pressure_core),('flux',network.flux),('momentum',network.momentum)):
                        def count(*_,key=label):observed[key]+=1
                        hooks.append(module.register_forward_hook(count))
                    previous=None;objective_fn=construct_objective(config,network)
                    for repeat in range(2):
                        network.zero_grad(set_to_none=True)
                        first=network(current,features);loss1=objective_fn(first,target1,counts,norm['scales']['tracer'])
                        feedback=next_input(first,counts,norm['scales'])
                        if not feedback.requires_grad:raise ValueError('Physical feedback detached')
                        second=network(feedback,feature_at(cache,start+k,k));loss2=objective_fn(second,target2,counts,norm['scales']['tracer'])
                        objective=(loss1+loss2)/2
                        if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite actual displacement objective')
                        objective.backward()
                        norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in trainable.items()}
                        if any(v is None or not np.isfinite(v)for v in norms.values()):raise ValueError('Every trainable path needs finite gradients')
                        for name,value in norms.items():active[name]|=value>0
                        gradients={n:p.grad.detach().cpu().numpy().tobytes()for n,p in trainable.items()}
                        if previous is not None and gradients!=previous:raise ValueError('Repeated two-block gradients differ')
                        previous=gradients
                        if any(p.grad is not None for p in network.parameters()if not p.requires_grad):raise ValueError('Frozen neural coefficients acquired gradients')
                    objective_fn.close()
                    for hook in hooks:hook.remove()
                    expected={name:4*k for name in observed};expected['encoder']=4
                    if observed!=expected:raise ValueError('Shared encoder/readout counts differ')
                    original_teacher=teaching.displacement_labels
                    def forbidden(*args,**kwargs):raise ValueError('Training label function reached from inference')
                    teaching.displacement_labels=forbidden
                    try:
                        with torch.no_grad():
                            reference=cpu(current.cpu(),features.cpu());difference=float((first.cpu()-reference).abs().max())
                            if difference>reg['CPU_device_max_abs_tolerance']:raise ValueError('CPU/device predictions differ')
                            changed=current.clone();changed[:,0]=123.;changed[:,4]=17.-.25*current[:,4]
                            altered=network(changed,features)
                            if not torch.equal(first[:,:,:4],altered[:,:,:4]):raise ValueError('Tracer/previous pressure affects fluid')
                            unforced=features.clone();unforced[:,7:]=0;other=network(current,unforced)
                            if torch.equal(first[:,:,:4],other[:,:,:4]):raise ValueError('Fluid ignores forcing')
                            if not torch.equal(first[:,0,4],other[:,0,4]):raise ValueError('First transport uses current-step force')
                            resting=current.clone();resting[:,1:4]=0;rest=network(resting,features)
                            if not torch.equal(rest[:,0,4],resting[:,4]):raise ValueError('Zero entering velocity changes tracer')
                            zero=current.clone();zero[:,4]=0;zero_out=network(zero,features)
                            zero_next=network(next_input(zero_out,counts,norm['scales']),feature_at(cache,start+k,k))
                            if torch.count_nonzero(zero_out[:,:,4])or torch.count_nonzero(zero_next[:,:,4]):raise ValueError('Zero tracer changed')
                            constant=current.clone();constant[:,4]=3.25;constant_out=network(constant,features)
                            constant_error=float((constant_out[:,:,4]-3.25).abs().max())/3.25
                            if constant_error>reg['constant_tracer_relative_tolerance']:raise ValueError('Constant tracer changed')
                            lower=float(current[:,4].min());upper=float(current[:,4].max())
                            tolerance=reg['tracer_bounds_relative_tolerance']*max(1.,abs(lower),abs(upper))
                            if float(first[:,:,4].min())<lower-tolerance or float(first[:,:,4].max())>upper+tolerance:
                                raise ValueError('Tracer left donor range')
                    finally:teaching.displacement_labels=original_teacher
                    if inventory(network)!=before:raise ValueError('Zero-update qualification changed weights')
                    rows.append(dict(horizons=k,counts_xyz=counts,cells_per_metre=cpm,start=start,
                        objective=float(objective.detach()),parameter_gradient_norms=norms,CPU_device_max_abs=difference,
                        counts_per_block={name:value//4 for name,value in observed.items()},deterministic_two_block_gradients=True,
                        training_teacher_disabled_during_inference=True,first_transport_force_independent=True,
                        zero_entering_velocity_first_tracer_exact=True,zero_tracer_two_blocks_exact=True,
                        constant_tracer_relative_difference=constant_error,tracer_within_donor_range=True,optimizer_updates=0))
        if not all(active.values()):raise ValueError('Some trainable tensors never have active gradients')
        aggregate[str(k)]=dict(all_trainable_tensors_active=True,initial_inventory=initial,
            parameters=sum(p.numel()for p in network.parameters()),trainable_parameters=sum(p.numel()for p in trainable.values()))
        print('Qualified fixed viscosity',args.device,'K',k,'four grids, two densities, cold/warm and own-history gradients',flush=True)
    write(args.output,dict(status='passed',device=args.device,rows=rows,aggregate=aggregate,initial_diagnostics=initial_rows,
        actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,optimizer_updates=0,feedback_protocol=2,
        source_sha256=reg['source_sha256'],protocol_sha256=sha(reg_path),inputs_receipt_sha256=sha(inputs/'receipt.json'),
        full_state_accuracy_claim=False,CPU_latency_claim=False,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if not args.output.exists():write(args.output,dict(status='failed',device=args.device,error=type(error).__name__+': '+str(error),qualifier_sha256=sha(__file__),optimizer_updates=0))
        raise
