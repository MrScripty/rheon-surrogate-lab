"""Qualify learned global transforms and actual deterministic two-block training."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import torch
from study_support import HERE,BASE,read,write,sha
from temporal_data import load,make_features,feature_at
from temporal_objective import loss
from temporal_feedback import next_input
from bundle_model_global import BundleNet
from bundle_model_v5 import BundleNet as Original
from learned_global import bases,restrict,prolong


def run(args):
    if args.output.exists():
        raise ValueError('Fresh global qualification required')
    os.sched_setaffinity(0,{23})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(2026101921)
    axes=bases((7,6,5),args.device)
    signal=torch.randn(2,3,5,6,7,device=args.device)
    reconstructed=prolong(restrict(signal,axes),axes)
    reconstruction=float((reconstructed-signal).abs().max())
    orthogonality=max(float((axis@axis.T-torch.eye(axis.shape[0],device=args.device)).abs().max())for axis in axes)
    if reconstruction>1e-5 or orthogonality>1e-6:
        raise ValueError('Independent complete-basis reconstruction/orthogonality failed')
    view=BASE/'temporal-training-view-02'
    manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',[view/row['path']for row in manifest['shards']],args.device)
    rows=[]
    for horizons in (1,2,3,4):
        torch.manual_seed(2026101921)
        network=BundleNet(horizons).to(args.device)
        original=Original(horizons).to(args.device)
        before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            parent=next(p for p in parents if p['scene']['counts']==counts)
            features=make_features(parent,norm['scales'],horizons,args.device)
            state=parent['states'][8:9]
            feature=feature_at(features,8,horizons)
            with torch.no_grad():
                expected=original(state,feature)
                actual=network(state,feature)
                if not torch.equal(expected,actual):
                    raise ValueError('Seeded capacity model must start at same persistence output function')
            tick=time.perf_counter()
            target1=parent['states'][9:9+horizons].unsqueeze(0)
            target2=parent['states'][9+horizons:9+2*horizons].unsqueeze(0)
            gradients=None
            for repeat in range(2):
                first=network(state,feature)
                feedback=next_input(first,counts,norm['scales'])
                if not feedback.requires_grad:
                    raise ValueError('Own-history feedback detached')
                second=network(feedback,feature_at(features,8+horizons,horizons))
                objective=(loss(first,target1,counts,norm['scales']['tracer'])+loss(second,target2,counts,norm['scales']['tracer']))/2
                network.zero_grad(set_to_none=True)
                objective.backward()
                if not bool(torch.isfinite(objective))or any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in network.parameters()):
                    raise ValueError('Actual deterministic two-block objective/backward failed')
                current={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
                if gradients is not None and current!=gradients:
                    raise ValueError('Repeated gradients are not bitwise deterministic')
                gradients=current
            if any(p.detach().cpu().numpy().tobytes()!=before[name]for name,p in network.named_parameters()):
                raise ValueError('Zero-update qualification changed weights')
            if float(loss(target1,target1,counts,norm['scales']['tracer']))>1e-12:
                raise ValueError('Equal physical states must have zero objective')
            rows.append(dict(horizons=horizons,counts_xyz=counts,parameters=sum(p.numel()for p in network.parameters()),
                loss=float(objective.detach()),seconds=time.perf_counter()-tick,optimizer_updates=0,
                initialization_function_bitwise_matches_original=True,finite_complete_backward=True,
                repeated_gradients_bitwise_equal=True))
    write(args.output,dict(status='passed',rows=rows,device=args.device,optimizer_updates=0,
        model_sha256=sha(HERE/'bundle_model_global.py'),global_operator_sha256=sha(HERE/'learned_global.py'),
        objective_sha256=sha(HERE/'temporal_objective.py'),feedback_sha256=sha(HERE/'temporal_feedback.py'),
        trainer_sha256=sha(HERE/'temporal_train_global.py'),feedback_protocol=2,
        complete_basis_reconstruction_max_abs=reconstruction,basis_orthogonality_max_abs=orthogonality,
        fitting_view_sha256=sha(view/'manifest.json'),normalization_sha256=sha(view/'normalization.json'),
        source_sha256=sha(__file__),cuda_initialized=torch.cuda.is_initialized(),candidate_numerical_pressure_solve=False))
    print('Qualified global neural model: allK, allfourgrids, deterministic two-block gradients and original initial function',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(parser.parse_args())
