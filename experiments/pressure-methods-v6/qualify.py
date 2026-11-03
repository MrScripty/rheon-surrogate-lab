"""Zero-update objective, device, conditioning and deployment qualification."""
import argparse
import io
import os
from pathlib import Path
import sys,time
from v6_support import ROOT,BASE,read,write,sha

def run(args):
    if args.output.exists():raise ValueError('Fresh qualification required')
    os.sched_setaffinity(0,{22});started=time.perf_counter();project=args.project.resolve()
    witness=read(project/'source-witness.json');slots={r['id']:ROOT/r['path']for r in read(project/'project.json')['data']}
    for name,expected in witness['sources'].items():
        if sha(project/name)!=expected:raise ValueError('Captured source changed')
    for name,spec in witness['inputs'].items():
        if sha(slots[name])!=spec['sha256']:raise ValueError('Captured input changed')
    sys.path.insert(0,str(project))
    import torch,model,data,train,fine_tuning,mixed_order
    device='cuda'if args.cuda else'cpu';model.configure(device);torch.set_num_threads(1)
    config=train.experiment(read(project/'config.json'));model.select(config)
    _,parents=data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    normalization=model.fit_normalization(parents,data.fit_normalization);torch.manual_seed(config['training']['seed'])
    network=model.PressureCorrectionNet().to(device);fine_tuning.initialize(network,slots['initial_weights'],data.helpers())
    inventory=data.helpers().state_inventory(network)
    if inventory!=fine_tuning.SPEC['initial_inventory']or sum(p.numel()for p in network.parameters())!=fine_tuning.SPEC['parameters']:raise ValueError('Exact declared initialization required')
    order=mixed_order.order(parents,config);sampling=dict(mixed_order.STATS)
    small=[p for p in parents if max(p['counts_xyz'])<=16]
    expected=mixed_order.order(small,dict(config,training=dict(config['training'],updates=1600)))
    retained=order[[max(parents[int(v[0,0])]['counts_xyz'])<=16 for v in order]]
    if retained.shape!=expected.shape or not bool((retained==expected).all()):raise ValueError('Original small sample subsequence changed')
    largest=max(parents,key=lambda p:p['arrays']['base'][0].size);losses=[]
    if args.cuda:torch.cuda.reset_peak_memory_stats()
    for frame in range(len(largest['frames'])):
        network.zero_grad(set_to_none=True);sample=data.sample(largest,frame,device=device)
        loss=model.training_loss(network,sample,normalization).mean()
        if not loss.requires_grad or not bool(torch.isfinite(loss)):raise ValueError('Nonfinite/nondifferentiable objective')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):raise ValueError('Incomplete finite parameter gradients')
        if not any(bool(p.grad.ne(0).any())for p in network.parameters()):raise ValueError('No trainable gradient')
        losses.append(float(loss.detach()))
    sample=data.sample(parents[0],0,device=device)
    zero={k:torch.zeros_like(v)if k in ['base','reference','rhs','native_input_residual']else v for k,v in sample.items()}
    calls=[];hook=network.register_forward_hook(lambda *_:calls.append(1));network.zero_grad(set_to_none=True)
    empty=model.training_loss(network,zero,normalization).mean()
    if empty.requires_grad:empty.backward()
    hook.remove()
    if float(empty.detach())!=0 or calls or any(p.grad is not None and not bool(torch.isfinite(p.grad).all())for p in network.parameters()):raise ValueError('Zero-equation bypass/backward failed')
    with torch.no_grad():
        first=model.predict_pressure(network,sample,normalization)
        second=model.predict_pressure(network,dict(sample,rhs=sample['rhs']+123,reference=sample['reference']-99),normalization)
        if not torch.equal(first,second):raise ValueError('Inference accessed labels')
    smoke=None;transfer=[]
    if not args.cuda:
        import training_multilevel
        manifest=read(slots['splits']);index=parents.index(largest);wanted={f'p{index:04d}_base',f'p{index:04d}_native_input_residual'};raw={}
        for slot,shard in zip(['dataset','dataset_extra',*train.VIEW_SLOTS],manifest['shards']):
            if not wanted.intersection(shard['keys']):continue
            with data.np.load(slots[slot],allow_pickle=False)as archive:
                raw.update({k:archive[k].copy()for k in wanted.intersection(shard['keys'])})
        for frame in range(len(largest['frames'])):
            conditioned=data.sample(largest,frame);base=torch.from_numpy(raw[f'p{index:04d}_base'][frame:frame+1]).unsqueeze(1)
            initial=torch.from_numpy(raw[f'p{index:04d}_native_input_residual'][frame:frame+1]).unsqueeze(1)
            delta=training_multilevel.correction(initial,conditioned['cell_size_m'],conditioned['rho'])if fine_tuning.SPEC['coarse_training_cycles']else torch.zeros_like(base)
            expected_base=base+delta;expected_residual=initial-model.full_operator(delta,conditioned['cell_size_m'],conditioned['rho'])
            if not torch.equal(expected_base,conditioned['base'])or not torch.equal(expected_residual,conditioned['native_input_residual']):raise ValueError('Post-coarse training/deployment inputs differ')
            manual=dict(conditioned,base=expected_base,native_input_residual=expected_residual)
            with torch.no_grad():
                if not torch.equal(model.predict_pressure(network,manual,normalization),model.predict_pressure(network,conditioned,normalization)):raise ValueError('Complete staged deployment differs')
            transfer.append(dict(frame=largest['frames'][frame]['index'],counts_xyz=largest['counts_xyz'],bitwise_conditioning_and_prediction_parity=True))
        predicted={}
        with torch.no_grad():
            for i,case in enumerate(train.cases(parents)):
                sample=data.sample(parents[case['parent']],case['frame']);p=model.predict_pressure(network,sample,normalization)
                if not bool(torch.isfinite(p).all()):raise ValueError('Nonfinite publication')
                predicted[f'case_{i:03d}']=p[0,0].numpy().copy()
        packet=io.BytesIO();data.np.savez(packet,**train.publication_arrays(predicted))
        if len(packet.getvalue())>8*1024**2:raise ValueError('Publication exceeds ordinary budget')
        smoke=dict(bytes=len(packet.getvalue()),full_prediction_inventory=data.inventory(predicted))
    if inventory!=data.helpers().state_inventory(network):raise ValueError('Zero-update qualification changed weights')
    write(args.output,dict(status='passed',project_sha256=sha(project/'project.json'),source_witness_sha256=sha(project/'source-witness.json'),
        parameters=fine_tuning.SPEC['parameters'],optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized(),finite_largest_grid_backward=True,largest_counts_xyz=largest['counts_xyz'],largest_losses=losses,
        exact_initial_inventory=True,inference_label_independent=True,frozen_teacher_unchanged=True,teacher_present=False,entrypoint_configuration_validated=True,
        zero_equation_backward=dict(loss=0.,network_forwards=0,nonfinite_gradients=[]),
        small_replay=dict(matching_updates=1600,exact_sample_subsequence=True),sampling=sampling,conditioning=data.CONDITIONING,
        deployment_transfer=transfer,publication_smoke=smoke,source_sha256=sha(__file__),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()if args.cuda else 0,elapsed_s=time.perf_counter()-started,promotion=False))
    print('Qualified tiny staged objective',project.name,device,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cuda',action='store_true');run(parser.parse_args())
