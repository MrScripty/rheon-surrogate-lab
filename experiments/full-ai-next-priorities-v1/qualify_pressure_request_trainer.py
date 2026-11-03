"""Qualify the captured component trainer, sampler, actual data gradients and probe."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import copy
import sys
from study_support import ROOT,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh captured-trainer qualification required')
    os.sched_setaffinity(0,{23});project=args.project.resolve();assembly=read(project/'assembly.json')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured source changed')
    sys.path.insert(0,str(project));import pressure_request_train as train
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    static=assembly['static_inputs']
    context=argparse.Namespace(config=project/'config.json',manifest=ROOT/static['manifest']['path'],
        normalization=ROOT/static['normalization']['path'],
        shards=[ROOT/static[slot]['path']for slot in static if slot.startswith('part_')])
    config,manifest,norm,parents,caches,network=train.construct(context,args.device)
    initial=train.inventory(network)
    rng=np.random.Generator(np.random.PCG64(config['sampling_seed']))
    samples=[train.sample_window(rng)for _ in range(config['updates'])]
    independently=np.random.Generator(np.random.PCG64(config['sampling_seed']));expected=[]
    for _ in range(config['updates']):
        parent=int(independently.integers(16));bucket=independently.random()
        if bucket<.25:step=0
        elif bucket<.5:step=int(independently.integers(1,9))
        else:step=int(independently.integers(9,121))
        expected.append((parent,step))
    if samples!=expected:raise ValueError('Actual sampler differs from registered cold/warm stream')
    rows=[]
    for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
        index=next(i for i,parent in enumerate(parents)if parent['scene']['counts']==counts)
        for step in (0,8,120):
            previous=None
            for repeat in range(2):
                network.zero_grad(set_to_none=True)
                objective=train.forward_objective(network,parents[index],caches[index],step);objective.backward()
                if not bool(torch.isfinite(objective))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                    raise ValueError('Captured trainer actual gradient failed')
                gradients={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
                if previous is not None and previous!=gradients:raise ValueError('Captured trainer repeated gradients differ')
                previous=gradients
            rows.append(dict(counts_xyz=counts,step=step,loss=float(objective.detach()),finite_complete_backward=True))
    if initial!=train.inventory(network):raise ValueError('Zero-update trainer qualification changed parameters')
    probe=None
    if args.device=='cpu':
        first=train.CPU_probe(copy.deepcopy(network),parents,norm);second=train.CPU_probe(copy.deepcopy(network),parents,norm)
        if not np.array_equal(first.view('<u4'),second.view('<u4')):raise ValueError('Actual four-grid CPU probe is not bitwise stable')
        probe=dict(elements=first.size,sha256=train.digest(first.tobytes()),all_four_grids=True,steps=[0,8,120],bitwise_repeat=True)
        if torch.cuda.is_initialized():raise ValueError('CPU trainer qualification initialized CUDA')
    write(args.output,dict(status='passed',project_sha256=assembly['project_sha256'],optimizer_updates=0,device=args.device,
        model_sha256=sha(project/'pressure_request_model.py'),objective_sha256=sha(project/'pressure_request_objective.py'),
        trainer_sha256=sha(project/'pressure_request_train.py'),artifact_helpers_sha256=sha(project/'managed_fit_artifacts.py'),
        data_sha256=sha(project/'pressure_request_data.py'),source_sha256=sha(__file__),
        initial_inventory=initial,registered_sample_order_sha256=train.digest(np.asarray(samples,dtype='<i8').tobytes()),
        all_parents_sampled=sorted({parent for parent,_ in samples}),rows=rows,CPU_probe=probe,
        component_only=True,future_request_input=False,current_pressure_input=False,full_state_temporal_candidate=False,
        cuda_initialized=torch.cuda.is_initialized()))
    print('Qualified captured causal request trainer',args.device,'12actual windows, exact sampler and complete deterministic gradients',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args())
