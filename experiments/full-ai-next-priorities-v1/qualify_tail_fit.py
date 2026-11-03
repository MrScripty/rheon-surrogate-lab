"""Zero-update check of actual head gradients, frozen fine tensors and CPU pressure."""
import argparse
import os
from pathlib import Path
import sys
import time
from study_support import ROOT,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh head objective qualification required')
    project=args.project.resolve();witness=read(project/'source-witness.json')
    for name,digest in witness['sources'].items():
        if sha(project/name)!=digest:raise ValueError('Captured head source changed')
    assembly=read(project/'assembly.json');sys.path.insert(0,str(project));import head_train,head_objective,model,torch
    context=argparse.Namespace(config=project/'calibration.json',**{slot:ROOT/spec['path']for slot,spec in assembly['static_inputs'].items()})
    started=time.perf_counter();config,network,initial,features,target,small,large,development=head_train.construct(context,args.device)
    if model.ARCHITECTURE!=witness['architecture']:raise ValueError('Actual head model descriptor differs from captured witness')
    inventory=head_train.inventory(network)
    selected=torch.tensor([*small[:256],*large[:128]],device=args.device)
    objective=head_objective.loss(network,features[selected],target[selected]);objective.backward()
    trainable=[p for p in network.parameters()if p.requires_grad]
    if not bool(torch.isfinite(objective))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in trainable):
        raise ValueError('Incomplete finite head gradient ownership')
    if any(p.grad is not None for name,p in network.named_parameters()if not head_objective.head_name(name)):
        raise ValueError('Frozen fine network acquired gradients')
    if inventory!=head_train.inventory(network)or inventory!=initial['inventory']:raise ValueError('Zero-update frozen-parent/seeded-tail initialization changed')
    probe=None
    if args.device=='cpu':
        result=head_train.probe(network,read(context.normalization))
        probe=dict(elements=result.size,sha256=head_train.digest(result.tobytes()),finite=True)
        if torch.cuda.is_initialized():raise ValueError('CPU qualification initialized CUDA')
    write(args.output,dict(status='passed',optimizer_updates=0,project_sha256=sha(project/'project.json'),
        source_witness_sha256=sha(project/'source-witness.json'),model_sha256=sha(project/'model.py'),
        objective_sha256=sha(project/'head_objective.py'),trainer_sha256=sha(project/'head_train.py'),source_sha256=sha(__file__),
        exact_initial_inventory=True,fine_tensors_bitwise_unchanged=True,all_trainable_gradients_finite=True,
        trainable_parameters=sum(p.numel()for p in trainable),parameters=sum(p.numel()for p in network.parameters()),
        actual_objective=float(objective.detach()),device=args.device,cuda_initialized=torch.cuda.is_initialized(),CPU_pressure_probe=probe,
        elapsed_s=time.perf_counter()-started,candidate_numerical_pressure_solve=False))
    print('Qualified actual head calibration',args.device,'all33982 tail parameters; zero fine updates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');run(parser.parse_args())
