"""Exercise actual MSE supervision constructors and deployed two-block gradients."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from temporal_train_increment_supervision import construct_network,construct_objective
from temporal_objective_mse import loss as reference_MSE
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from study_support import HERE,BASE,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh supervision qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',
                        [view/row['path']for row in manifest['shards']],args.device)
    statistics=BASE/'temporal-step-scale-statistics-01.json'
    base_config=dict(width=12,horizons=2,architecture='endpoint',
        step_scale_statistics_sha256=sha(statistics),original_normalization_sha256=sha(view/'normalization.json'))
    rows=[];initial_inventory=None
    for mode in ('full_state','increments'):
        config=dict(base_config,supervision_mode=mode);torch.manual_seed(2026101921)
        seeded=construct_network(config,'cpu',statistics)
        initial={name:value.detach().numpy().tobytes()for name,value in seeded.state_dict().items()}
        if initial_inventory is not None and initial!=initial_inventory:raise ValueError('Supervision arms change seeded parameter initialization')
        initial_inventory=initial
        network=copy.deepcopy(seeded)
        with torch.no_grad():
            for name,module in network.named_modules():
                if name.endswith('head')and isinstance(module,torch.nn.Conv3d):module.weight.normal_(std=.01)
            for block in network.global_blocks:block.geometry_head[-1].weight.normal_(std=.001)
        cpu=copy.deepcopy(network).eval();network=network.to(args.device)
        objective_fn=construct_objective(config,network)
        before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            parent=next(p for p in parents if p['scene']['counts']==counts)
            cache=make_features(parent,norm['scales'],2,args.device)
            for start in (0,8):
                current=parent['states'][start:start+1];feature=feature_at(cache,start,2)
                target1=parent['states'][start+1:start+3].unsqueeze(0)
                target2=parent['states'][start+3:start+5].unsqueeze(0)
                gradients=None
                for repeat in range(2):
                    first=network(current,feature)
                    feedback=next_input(first,counts,norm['scales'])
                    if not feedback.requires_grad:raise ValueError('Feedback detached')
                    second=network(feedback,feature_at(cache,start+2,2))
                    objective=(objective_fn(first,target1,counts,norm['scales']['tracer'])+
                               objective_fn(second,target2,counts,norm['scales']['tracer']))/2
                    if mode=='full_state':
                        expected=(reference_MSE(first,target1,counts,norm['scales']['tracer'])+
                                  reference_MSE(second,target2,counts,norm['scales']['tracer']))/2
                        if not torch.equal(expected,objective):raise ValueError('Full-state arm differs from established relative-MSE control')
                    if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite supervision objective')
                    network.zero_grad(set_to_none=True);objective.backward()
                    norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in network.named_parameters()}
                    if any(value is None or not np.isfinite(value)or value<=0 for value in norms.values()):
                        raise ValueError('Complete active finite gradients required')
                    actual={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
                    if gradients is not None and gradients!=actual:raise ValueError('Nondeterministic repeated gradients')
                    gradients=actual
                with torch.no_grad():
                    equal=float(objective_fn(target1,target1,counts,norm['scales']['tracer']))
                    if equal>1e-12:raise ValueError('Equal physical states must have zero supervision loss')
                    difference=float((first.detach().cpu()-cpu(current.cpu(),feature.cpu())).abs().max())
                    if difference>1e-5:raise ValueError('CPU/device model output parity exceeded tolerance')
                if any(p.detach().cpu().numpy().tobytes()!=before[name]for name,p in network.named_parameters()):
                    raise ValueError('Zero-update qualification changed probe weights')
                rows.append(dict(supervision_mode=mode,architecture='endpoint',horizons=2,counts_xyz=counts,start=start,
                    loss=float(objective.detach()),parameter_gradient_norms=norms,optimizer_updates=0,
                    repeated_gradients_bitwise_equal=True,CPU_device_output_max_abs=difference,
                    established_relative_MSE_bitwise_equal=mode=='full_state',zero_loss_for_equal_states=True))
    files=('bundle_model_step_scaled.py','learned_global.py','temporal_train_increment_supervision.py',
           'temporal_objective_increment.py','temporal_objective_mse.py','temporal_objective.py',
           'temporal_features_v2.py','temporal_feedback.py','temporal_data.py')
    write(args.output,dict(status='passed',rows=rows,device=args.device,optimizer_updates=0,
        source_sha256={name:sha(HERE/name)for name in files},qualification_source_sha256=sha(__file__),
        step_statistics_sha256=sha(statistics),normalization_sha256=sha(view/'normalization.json'),
        fitting_view_sha256=sha(view/'manifest.json'),feedback_protocol=2,
        actual_trainer_constructor_exercised=True,actual_objective_constructor_exercised=True,
        identical_initial_tensors_between_supervision_arms=True,physical_evaluation_criteria_unchanged=True,
        probe_assignments='Nonzero output/geometry diagnostic copies only; never fitting initialization',
        reserved_scenes_consumed=False,candidate_numerical_pressure_solve=False))
    print('Qualified both MSE supervision modes, all four grids, cold/warm starts and deterministic two-block gradients',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
