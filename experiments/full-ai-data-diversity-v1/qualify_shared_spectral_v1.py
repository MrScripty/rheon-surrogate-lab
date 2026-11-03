"""Zero-update CUDA constructor/decoder/native-objective qualification."""
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from support import ROOT,read,write,digest
from shared_spectral_model_v1 import SharedSpectralResponse,objective
from modal_basis import bases


def inventory(model):
    return {name:dict(dtype=value.detach().cpu().numpy().dtype.str,shape=list(value.shape),
        sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()) for name,value in model.state_dict().items()}


def run(args):
    if args.output.exists():raise ValueError('Fresh CUDA qualification evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA qualification required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen qualification source changed: '+path)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=ROOT/reg['view'];manifest=read(view/'manifest.json')
    if manifest['status']!='verified_all148train_fixed_shared_spectral_view' or manifest['training_scenes']!=148 or manifest['development_or_reserved_used']:
        raise ValueError('All148scene training view required')
    part=view/manifest['parts'][0]['path']
    if digest(part)!=manifest['parts'][0]['sha256']:raise ValueError('Typed native supervision artifact changed')
    with np.load(part,allow_pickle=False) as archive:
        if set(archive.files)!=set(manifest['arrays']):raise ValueError('Complete native objective view required')
        batch={}
        for name,spec in manifest['arrays'].items():
            value=archive[name]
            if value.dtype.str!=spec['dtype'] or list(value.shape)!=spec['shape'] or not np.isfinite(value).all():
                raise ValueError('Exact finite supervision arrays required')
            batch[name]=torch.from_numpy(value.copy()).to('cuda')
    if int(batch['scene_index'].unique().numel())!=148:raise ValueError('Every expanded scene must contribute')
    model=SharedSpectralResponse(device='cuda');before=inventory(model)
    if sum(p.numel() for p in model.parameters())!=4:raise ValueError('Registered four-parameter learned head required')
    loss=objective(model,batch)
    if not bool(torch.isfinite(loss)):raise ValueError('Finite actual native objective required')
    loss.backward()
    gradient=model.coefficients.grad.detach().clone()
    if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm()>0):raise ValueError('Finite nonzero CUDA gradient required')
    direction=torch.tensor([.25,-.5,.75,-1.],dtype=torch.float64,device='cuda');direction=direction/direction.norm()
    with torch.no_grad():
        saved=model.coefficients.detach().clone();epsilon=reg['finite_difference_epsilon']
        model.coefficients.copy_(saved+epsilon*direction);plus=objective(model,batch)
        model.coefficients.copy_(saved-epsilon*direction);minus=objective(model,batch)
        model.coefficients.copy_(saved)
    expected=(gradient*direction).sum();observed=(plus-minus)/(2*epsilon)
    relative=float((expected-observed).abs()/expected.abs().clamp_min(1e-12))
    if relative>reg['maximum_gradient_relative_error']:raise ValueError('Actual objective gradient finite-difference check failed')
    with torch.no_grad():
        shape=(3,4,5);q=torch.randn((1,1,*shape),dtype=torch.float64,device='cuda');axes=bases(shape,'cuda')
        decoded=model.decode(q,axes)
        if decoded.shape!=q.shape or decoded.dtype!=torch.float64 or not bool(decoded.isfinite().all()) or float(decoded[0,0,0,0,0])!=0:
            raise ValueError('Complete CUDA pressure decoder/gauge contract failed')
    if inventory(model)!=before:raise ValueError('Zero-update qualification changed learned tensors')
    # Exact typed CPU restore checks weights only. No CPU model forward occurs.
    cpu_weights={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    restore=SharedSpectralResponse(device='cpu');restore.load_state_dict(cpu_weights,strict=True)
    if inventory(restore)!=before:raise ValueError('Exact typed CPU weight restore failed')
    for path,expected_sha in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected_sha:raise ValueError('Qualification source changed during execution: '+path)
    write(args.output,dict(status='passed',role='zero_update_cuda_shared_spectral_constructor_decoder_and_objective',
        inference_device='cuda',gpu_model=torch.cuda.get_device_name(),parameter_count=4,parameter_inventory=before,
        constructor_forward_and_decoder_on_cuda=True,native_objective_on_cuda=True,native_objective=float(loss.detach()),
        objective_gradient_directional_relative_error=relative,gradient_finite_nonzero=True,
        all148training_scenes_in_objective=True,typed_cpu_weight_restore_exact=True,cpu_model_forward_executed=False,
        optimizer_updates=0,learned_coefficient_ratio_targets_computed=False,physical_eigenvalues_constructed=False,
        full_model_accuracy_claim=False,latency_claim=False,expanded_reserved_references_opened=0,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
    print('Qualified shared spectral constructor/decoder and actual148scene objective onGPU; zero optimizer updates.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
