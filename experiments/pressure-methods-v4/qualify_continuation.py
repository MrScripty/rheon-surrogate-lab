"""Qualify the full-width replay/solver-aware fits without optimizer updates."""
import argparse
import io
import os
from pathlib import Path
import sys
import time
from v4_support import ROOT,read,write,sha

def run(args):
    if args.output.exists():raise ValueError('Fresh objective qualification required')
    os.sched_setaffinity(0,{22});project=args.project.resolve();started=time.perf_counter()
    witness=read(project/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(project/name)!=expected:raise ValueError('Captured objective source changed')
    slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    for name,spec in witness['inputs'].items():
        if sha(slots[name])!=spec['sha256']:raise ValueError('Fitting input changed')
    sys.path.insert(0,str(project))
    import torch,model,data,train,fine_tuning,mixed_order
    model.configure('cuda'if args.cuda else'cpu');torch.set_num_threads(1)
    config=train.experiment(read(project/'config.json'));model.select(config)
    _,parents=data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    norm=model.fit_normalization(parents,data.fit_normalization);torch.manual_seed(config['training']['seed'])
    network=model.PressureCorrectionNet().to('cuda'if args.cuda else'cpu')
    fine_tuning.initialize(network,slots['initial_weights'],data.helpers())
    inventory=data.helpers().state_inventory(network);teacher=data.helpers().state_inventory(fine_tuning.TEACHER)
    if inventory!=fine_tuning.SPEC['initial_inventory']:raise ValueError('Exact continuation required')
    order=mixed_order.order(parents,config);sampling=dict(mixed_order.STATS)
    small=[p for p in parents if max(p['counts_xyz'])<=16]
    expected=mixed_order.order(small,dict(config,training=dict(config['training'],updates=1600)))
    retained=order[[max(parents[int(v[0,0])]['counts_xyz'])<=16 for v in order]]
    if retained.shape!=expected.shape or not bool((retained==expected).all()):raise ValueError('Original small exposure changed')
    largest=max(parents,key=lambda p:p['arrays']['base'][0].size);losses=[]
    if args.cuda:torch.cuda.reset_peak_memory_stats()
    for frame in range(len(largest['frames'])):
        network.zero_grad(set_to_none=True);sample=data.sample(largest,frame,device='cuda'if args.cuda else'cpu')
        loss=model.training_loss(network,sample,norm).mean()
        if not loss.requires_grad or not bool(torch.isfinite(loss)):raise ValueError('Objective is not finite/differentiable')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):raise ValueError('Incomplete finite gradient ownership')
        if not any(bool((p.grad!=0).any())for p in network.parameters()):raise ValueError('All parameter gradients vanished')
        losses.append(float(loss.detach()))
    if inventory!=data.helpers().state_inventory(network)or teacher!=data.helpers().state_inventory(fine_tuning.TEACHER):raise ValueError('Qualification modified weights')
    if any(p.grad is not None or p.requires_grad for p in fine_tuning.TEACHER.parameters()):raise ValueError('Teacher no longer frozen')
    zero_backward=None
    if 'solver_training'in model.ARCHITECTURE:
        small_sample=data.sample(parents[0],0,device='cuda'if args.cuda else'cpu')
        empty={k:torch.zeros_like(v)if k in ['base','rhs','reference','native_input_residual']else v for k,v in small_sample.items()}
        calls=[];hook=network.register_forward_hook(lambda *_:calls.append(1))
        network.zero_grad(set_to_none=True);zero=model.training_loss(network,empty,norm).mean()
        if zero.requires_grad:zero.backward()
        hook.remove()
        bad=[name for name,p in network.named_parameters()if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        if float(zero.detach())!=0 or bad or calls:raise ValueError('GCR zero-equation bypass/backward failed on the actual device')
        zero_backward=dict(loss=0.,nonfinite_gradients=[],network_forwards=0,device=str(zero.device),
            interpretation='Unused fine parameters intentionally have no gradient when the captured predictor bypasses a zero residual.')
    publication=None;label_independent=False;zero_loss=None;raw_parity=None
    if not args.cuda:
        import scalable
        class State:pass
        state=State();state.network=network;state.torch=torch
        _,raw_parity=scalable.qualify(state);predicted={}
        with torch.no_grad():
            sample=data.sample(parents[0],0);prediction=model.predict_pressure(network,sample,norm)
            changed=dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99)
            if not torch.equal(prediction,model.predict_pressure(network,changed,norm)):raise ValueError('Inference accessed pressure labels')
            label_independent=True
            zero={k:torch.zeros_like(v)if k in ['base','rhs','reference','native_input_residual']else v for k,v in sample.items()}
            zero_loss=float(model.training_loss(network,zero,norm).mean())
            if zero_loss!=0:raise ValueError('Zero equation must have zero loss')
            for i,case in enumerate(train.cases(parents)):
                sample=data.sample(parents[case['parent']],case['frame']);pressure=model.predict_pressure(network,sample,norm)
                if not bool(torch.isfinite(pressure).all()):raise ValueError('Publication nonfinite')
                predicted[f'case_{i:03d}']=pressure[0,0].numpy().copy()
        packet=io.BytesIO();data.np.savez(packet,**train.publication_arrays(predicted))
        if len(packet.getvalue())>8*1024**2:raise ValueError('Publication export exceeds ordinary evidence limit')
        publication=dict(bytes=len(packet.getvalue()),full_prediction_inventory=data.inventory(predicted),sampled_export_inventory=data.inventory(train.publication_arrays(predicted)))
        if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
    write(args.output,dict(status='passed',project_sha256=sha(project/'project.json'),source_witness_sha256=sha(project/'source-witness.json'),
        exact_initial_inventory=True,frozen_teacher_unchanged=True,entrypoint_configuration_validated=True,parameters=sum(p.numel()for p in network.parameters()),
        optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized(),largest_counts_xyz=largest['counts_xyz'],largest_losses=losses,finite_largest_grid_backward=True,
        small_replay=dict(matching_updates=1600,exact_sample_subsequence=True),sampling=sampling,inference_label_independent=label_independent,zero_equation_loss=zero_loss,
        zero_equation_backward=zero_backward,
        raw_forward_parity=raw_parity,publication_smoke=publication,source_sha256=sha(__file__),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()if args.cuda else 0,observed_peak_rss_bytes=data.helpers().observed_peak_rss_bytes(),
        elapsed_s=time.perf_counter()-started,scientific_acceptance_or_promotion=False))
    print('Qualified full objective',project.name,'CUDA'if args.cuda else'CPU',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cuda',action='store_true')
    run(parser.parse_args())
