"""Actual train-data, all-horizon deterministic gradient and offline teacher checks."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import torch
from study_support import HERE,BASE,read,write,sha
from pressure_request_data import load,features,inputs
from pressure_request_model import CausalRequestNet
from pressure_request_objective import loss,teacher_pressure
from temporal_objective import laplacian


def run(args):
    if args.output.exists():raise ValueError('Fresh causal request qualification required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=BASE/'temporal-request-training-view-01';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',
        [view/r['path']for r in manifest['shards']],args.device)
    rows=[]
    for k in (1,2,3,4):
        torch.manual_seed(2026101994)
        network=CausalRequestNet(k,norm['scales']['velocity'],norm['scales']['q']).to(args.device)
        before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            parent=next(p for p in parents if p['scene']['counts']==counts)
            cache=features(parent,norm['scales'],k,args.device)
            arguments=inputs(parent,cache,8,k,args.device)
            target=parent['q'][9:9+k].unsqueeze(0);previous=None;tick=time.perf_counter()
            for repeat in range(2):
                network.zero_grad(set_to_none=True)
                predicted=network(*arguments)
                objective=loss(predicted,target);objective.backward()
                if not bool(torch.isfinite(objective))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                    raise ValueError('Actual request objective or complete backward failed')
                gradients={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
                if previous is not None and gradients!=previous:raise ValueError('Repeated gradients nondeterministic')
                previous=gradients
            if before!={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}:
                raise ValueError('Zero-update qualification changed parameters')
            teacher=teacher_pressure(target)
            centered=target-target.mean(dim=(-3,-2,-1),keepdim=True)
            error=float((laplacian(teacher)-centered).norm())/max(float(centered.norm()),1e-30)
            if error>1e-4:raise ValueError('Offline teacher fails independent full-stencil reconstruction')
            rows.append(dict(horizons=k,counts_xyz=counts,parameters=sum(p.numel()for p in network.parameters()),
                objective=float(objective.detach()),teacher_equation_relative_error=error,
                repeated_gradients_bitwise_equal=True,complete_finite_backward=True,seconds=time.perf_counter()-tick))
            print('Qualified causal request',args.device,'K',k,counts,flush=True)
    write(args.output,dict(status='passed',device=args.device,optimizer_updates=0,rows=rows,
        source_sha256=sha(__file__),model_sha256=sha(HERE/'pressure_request_model.py'),
        objective_sha256=sha(HERE/'pressure_request_objective.py'),data_sha256=sha(HERE/'pressure_request_data.py'),
        features_sha256=sha(HERE/'temporal_features_v2.py'),manifest_sha256=sha(view/'manifest.json'),
        normalization_sha256=sha(view/'normalization.json'),CUDA_initialized=torch.cuda.is_initialized(),
        future_request_input=False,current_or_future_pressure_input=False,temporal_candidate=False,
        candidate_numerical_pressure_solve=False,analytical_pressure_inverse='Offline training objective only'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
