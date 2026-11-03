"""Qualify real continuation loss, complete gradients, sampling and CPU publication."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
from study_support import ROOT,BASE,read,write,sha

def run(args):
    if args.output.exists():raise ValueError('Fresh fit qualification required')
    started=time.perf_counter();os.sched_setaffinity(0,{22})
    project=args.project.resolve();witness=read(project/'source-witness.json')
    for name,digest in witness['sources'].items():
        if sha(project/name)!=digest:raise ValueError('Captured Project source changed')
    slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    for name,spec in witness['inputs'].items():
        if sha(slots[name])!=spec['sha256']:raise ValueError('Fitting input changed')
    sys.path.insert(0,str(project))
    import torch,model,data,train,fine_tuning,mixed_order,head_optimizer
    model.configure(args.device);torch.set_num_threads(1)
    config=train.experiment(read(project/'config.json'));model.select(config)
    _,parents=data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    norm=model.fit_normalization(parents,data.fit_normalization)
    torch.manual_seed(config['training']['seed']);network=model.PressureCorrectionNet().to(args.device)
    fine_tuning.initialize(network,slots['initial_weights'],data.helpers(),slots['teacher_weights'])
    inventory=data.helpers().state_inventory(network)
    if inventory!=fine_tuning.SPEC['initial_inventory']:raise ValueError('Architecture fit must start at the registered mapped C107 tensors')
    teacher_before=data.helpers().state_inventory(fine_tuning.FULL_TEACHER)
    order=mixed_order.order(parents,config);sampling=dict(mixed_order.STATS)
    updates=config['training']['updates']
    if order.shape!=(updates,1,2)or sampling['tier_exposures']['16']!=updates*2//3:
        raise ValueError('Registered spatial exposure proportions failed')
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Fitting parents must be train-only')
    groups=head_optimizer.groups(network,config['training']['learning_rate'])
    if len({id(p)for g in groups for p in g['params']})!=len(list(network.parameters())):
        raise ValueError('Incomplete optimizer ownership')
    largest=max(parents,key=lambda p:len(p['arrays']['base'][0].reshape(-1)))
    losses=[]
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    for frame in range(len(largest['frames'])):
        network.zero_grad(set_to_none=True)
        sample=data.sample(largest,frame,device=args.device)
        loss=model.training_loss(network,sample,norm).mean()
        if not loss.requires_grad or not bool(torch.isfinite(loss)):raise ValueError('Actual loss not finite and differentiable')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
            raise ValueError('Real loss must reach all model parameters with finite gradients')
        losses.append(float(loss.detach()))
    if fine_tuning.STATS['distillation_teacher_forwards']!=0:raise ValueError('Continuation must not use distillation')
    if inventory!=data.helpers().state_inventory(network)or teacher_before!=data.helpers().state_inventory(fine_tuning.FULL_TEACHER):
        raise ValueError('Zero-update check changed checkpoint weights')
    if any(p.requires_grad or p.grad is not None for p in fine_tuning.FULL_TEACHER.parameters()):
        raise ValueError('Frozen teacher ownership violated')
    publication=None
    if args.device=='cpu':
        predicted={}
        with torch.no_grad():
            sample=data.sample(parents[0],0)
            pressure=model.predict_pressure(network,sample,norm)
            changed=dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99)
            if not torch.equal(pressure,model.predict_pressure(network,changed,norm)):
                raise ValueError('Inference accessed labels')
            for i,case in enumerate(train.cases(parents)):
                sample=data.sample(parents[case['parent']],case['frame'])
                value=model.predict_pressure(network,sample,norm)
                if not bool(torch.isfinite(value).all()):raise ValueError('Prediction publication nonfinite')
                predicted[f'case_{i:03d}']=value[0,0].numpy().copy()
        stream=io.BytesIO();data.np.savez(stream,**train.publication_arrays(predicted))
        if len(stream.getvalue())>8*1024**2:raise ValueError('Publication exceeds declared output budget')
        publication=dict(bytes=len(stream.getvalue()),full_inventory=data.inventory(predicted),label_independent=True)
        if torch.cuda.is_initialized():raise ValueError('CPU qualification initialized CUDA')
    write(args.output,dict(status='passed',optimizer_updates=0,project_sha256=sha(project/'project.json'),
        source_witness_sha256=sha(project/'source-witness.json'),exact_initial_inventory=True,mapped_C107_initialization=read(project/'architecture-initialization.json'),
        frozen_teacher_unchanged=True,parameters=sum(p.numel()for p in network.parameters()),
        complete_finite_gradient_ownership=True,largest_counts_xyz=largest['counts_xyz'],largest_losses=losses,
        sampling=sampling,head_multiplier=groups[0]['lr_multiplier'],publication=publication,
        device=args.device,cuda_initialized=torch.cuda.is_initialized(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()if args.device=='cuda'else 0,
        source_sha256=sha(__file__),elapsed_s=time.perf_counter()-started,scientific_acceptance=False))
    print('Qualified actual mapped C107 architecture',project.name,args.device,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args())
