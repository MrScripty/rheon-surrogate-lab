"""Exercise nonzero gradients through every new global parameter before fitting."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from bundle_model_global import BundleNet
from temporal_features_v2 import packed_state,auxiliary
from temporal_objective import loss
from temporal_feedback import next_input


def run(args):
    if args.output.exists():
        raise ValueError('Fresh active-gradient qualification required')
    os.sched_setaffinity(0,{23})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    bank=read(BASE/'temporal-data-01/bank.json')
    row=next(row for row in bank['trajectories']if row['split']=='train'and row['counts_xyz']==[16,16,16])
    path=ROOT/row['receipt']
    if sha(path)!=row['receipt_sha256']:
        raise ValueError('Training reference changed')
    receipt=read(path)
    scene=receipt['scene']
    norm=read(BASE/'temporal-training-view-02/normalization.json')
    states=[]
    for step in range(8,17):
        record=receipt['states'][step]
        source=path.parent/record['path']
        if sha(source)!=record['sha256']:
            raise ValueError('Physical training state changed')
        with np.load(source,allow_pickle=False)as arrays:
            fields={name:arrays[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
        states.append(packed_state(fields,tuple(scene['counts']),norm['scales'],args.device))
    rows=[]
    for horizons in(1,2,3,4):
        torch.manual_seed(2026101958)
        network=BundleNet(horizons)
        # Qualification-only probe avoids the zero output-head gradient at initialization.
        # These assigned probe weights are never published or used to initialize fitting.
        with torch.no_grad():
            network.head.weight.normal_(std=.01)
            for block in network.global_blocks:
                block.geometry_head[-1].weight.normal_(std=.001)
        reference=copy.deepcopy(network).eval()
        network=network.to(args.device)
        before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
        features=auxiliary(scene,8,horizons,norm['scales'],args.device)
        future=auxiliary(scene,8+horizons,horizons,norm['scales'],args.device)
        gradients=None
        for repeat in range(2):
            first=network(states[0],features)
            second=network(next_input(first,scene['counts'],norm['scales']),future)
            target1=torch.stack(states[1:1+horizons],dim=1)
            target2=torch.stack(states[1+horizons:1+2*horizons],dim=1)
            objective=(loss(first,target1,scene['counts'],norm['scales']['tracer'])+
                       loss(second,target2,scene['counts'],norm['scales']['tracer']))/2
            network.zero_grad(set_to_none=True)
            objective.backward()
            norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in network.named_parameters()}
            if any(value is None or not np.isfinite(value)or value<=0 for value in norms.values()):
                raise ValueError('Qualification probe requires active finite gradients for every parameter tensor')
            current={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
            if gradients is not None and current!=gradients:
                raise ValueError('Repeated active gradients differ')
            gradients=current
        with torch.no_grad():
            cpu=reference(states[0].cpu(),features.cpu())
            difference=float((first.detach().cpu()-cpu).abs().max())
        if difference>1e-5 or any(p.detach().cpu().numpy().tobytes()!=before[name]for name,p in network.named_parameters()):
            raise ValueError('CPU/device parity or unchanged probe inventory failed')
        rows.append(dict(horizons=horizons,counts_xyz=scene['counts'],loss=float(objective.detach()),
            parameter_gradient_norms=norms,all_parameter_tensors_nonzero_finite=True,repeated_gradients_bitwise_equal=True,
            CPU_device_output_max_abs=difference,optimizer_updates=0))
    write(args.output,dict(status='passed',rows=rows,device=args.device,optimizer_updates=0,
        probe_assignments='Nonzero output head and geometry-head last weights only; diagnostic copy, never used for fit initialization',
        model_sha256=sha(HERE/'bundle_model_global.py'),global_operator_sha256=sha(HERE/'learned_global.py'),
        objective_sha256=sha(HERE/'temporal_objective.py'),trainer_sha256=sha(HERE/'temporal_train_global.py'),
        fitting_view_sha256=sha(BASE/'temporal-training-view-02/manifest.json'),source_sha256=sha(__file__),
        CUDA_initialized=torch.cuda.is_initialized(),reserved_scenes_consumed=False))
    print('Qualified nonzero deterministic gradients through every global-model parameter for allK',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args())
