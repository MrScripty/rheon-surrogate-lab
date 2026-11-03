"""Qualify both deployed sharing paths and active two-block gradients, without fitting."""
import argparse
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from bundle_model_sharing import BundleNet
from temporal_data import load,make_features,feature_at
from temporal_feedback import next_input
from temporal_objective import loss
from study_support import HERE,BASE,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh sharing qualification required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    _,norm,parents=load(view/'manifest.json',view/'normalization.json',
                        [view/row['path']for row in manifest['shards']],args.device)
    rows=[]
    for architecture in ('endpoint','reuse'):
        for horizons in (2,3,4):
            torch.manual_seed(2026101921)
            seeded=BundleNet(horizons,12,architecture)
            seeded_inventory={name:p.detach().numpy().tobytes()for name,p in seeded.named_parameters()}
            # Nonzero diagnostic copies exercise hidden paths that zero output
            # heads intentionally gate at initialization. Never used for fitting.
            network=copy.deepcopy(seeded)
            with torch.no_grad():
                for name,module in network.named_modules():
                    if name.endswith('head')and isinstance(module,torch.nn.Conv3d):module.weight.normal_(std=.01)
                for block in network.global_blocks:block.geometry_head[-1].weight.normal_(std=.001)
            cpu=copy.deepcopy(network).eval();network=network.to(args.device)
            before={name:p.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
            for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
                parent=next(p for p in parents if p['scene']['counts']==counts)
                cache=make_features(parent,norm['scales'],horizons,args.device)
                current=parent['states'][8:9];feature=feature_at(cache,8,horizons)
                target1=parent['states'][9:9+horizons].unsqueeze(0)
                target2=parent['states'][9+horizons:9+2*horizons].unsqueeze(0)
                gradients=None;probe=None
                for repeat in range(2):
                    calls={'encoder':0,'latent_updates':0}
                    def encoder_hook(*unused):calls['encoder']+=1
                    def latent_hook(*unused):calls['latent_updates']+=1
                    handles=[network.fine.register_forward_pre_hook(encoder_hook)]
                    if architecture=='reuse':handles.append(network.latent_update.register_forward_pre_hook(latent_hook))
                    try:
                        first=network(current,feature)
                        feedback=next_input(first,counts,norm['scales'])
                        if not feedback.requires_grad:raise ValueError('Feedback detached')
                        second=network(feedback,feature_at(cache,8+horizons,horizons))
                    finally:
                        for handle in handles:handle.remove()
                    if calls!={'encoder':2,'latent_updates':2*(horizons-1)if architecture=='reuse'else 0}:
                        raise ValueError('Full spatial encoder must execute once per block')
                    objective=(loss(first,target1,counts,norm['scales']['tracer'])+
                               loss(second,target2,counts,norm['scales']['tracer']))/2
                    network.zero_grad(set_to_none=True);objective.backward()
                    norms={name:float(p.grad.norm())if p.grad is not None else None for name,p in network.named_parameters()}
                    if any(value is None or not np.isfinite(value)or value<=0 for value in norms.values()):
                        raise ValueError('All parameter tensors require finite active gradients')
                    current_gradients={name:p.grad.detach().cpu().numpy().tobytes()for name,p in network.named_parameters()}
                    if gradients is not None and gradients!=current_gradients:raise ValueError('Nondeterministic gradients')
                    gradients=current_gradients;probe=first.detach()
                with torch.no_grad():
                    expected=cpu(current.cpu(),feature.cpu())
                    difference=float((probe.cpu()-expected).abs().max())
                    if difference>1e-5:raise ValueError('CPU/device parity exceeded tolerance')
                    changed=current.clone();changed[:,0]+=100
                    if not torch.equal(probe,network(changed,feature)):raise ValueError('Previous-pressure shortcut returned')
                    unforced=feature.clone();unforced[:,7:]=0
                    force_response=float((probe-network(current,unforced)).norm())
                    if force_response<=0:raise ValueError('Known forcing disconnected')
                    dependency=None
                    if architecture=='endpoint':
                        handle=network.endpoint_head.register_forward_hook(lambda module,inputs,output:output+.01)
                        try:dependency=float((probe[:,:-1]-network(current,feature)[:,:-1]).norm())
                        finally:handle.remove()
                        if dependency<=0:raise ValueError('Intermediate decoder does not consume predicted endpoint')
                if any(p.detach().cpu().numpy().tobytes()!=before[name]for name,p in network.named_parameters()):
                    raise ValueError('Qualification changed model parameters')
                if any(p.detach().numpy().tobytes()!=seeded_inventory[name]for name,p in seeded.named_parameters()):
                    raise ValueError('Seeded fit initialization changed')
                rows.append(dict(architecture=architecture,horizons=horizons,counts_xyz=counts,
                    parameters=sum(p.numel()for p in network.parameters()),loss=float(objective.detach()),
                    encoder_calls_per_block=1,latent_updates_per_block=horizons-1 if architecture=='reuse'else 0,
                    predicted_endpoint_dependency_norm=dependency,known_force_response_norm=force_response,
                    previous_pressure_independent=True,CPU_device_output_max_abs=difference,
                    parameter_gradient_norms=norms,repeated_gradients_bitwise_equal=True,optimizer_updates=0))
    files=('bundle_model_sharing.py','learned_global.py','temporal_train_sharing.py','temporal_features_v2.py',
           'temporal_objective.py','temporal_feedback.py','temporal_data.py')
    write(args.output,dict(status='passed',device=args.device,rows=rows,optimizer_updates=0,
        feedback_protocol=2,source_sha256={name:sha(HERE/name)for name in files},
        qualification_source_sha256=sha(__file__),fitting_view_sha256=sha(view/'manifest.json'),
        normalization_sha256=sha(view/'normalization.json'),reserved_scenes_consumed=False,
        probe_assignments='Diagnostic model copies only; seeded zero-head fit initialization unchanged',
        candidate_numerical_pressure_solve=False))
    print('Qualified both sharing architectures, all requested K, all four grids and active two-block gradients',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
