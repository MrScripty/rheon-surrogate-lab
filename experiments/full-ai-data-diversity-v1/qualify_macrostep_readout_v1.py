"""All148scene readout objective/gradient controls; zero optimizer updates."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import numpy as np
import torch
from support import ROOT,read,write,digest
from macrostep_readout_v1 import MacroReadout


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh CUDA-only zero-update qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen objective dependency changed: '+p)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{reg['host_cpu']})
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=ROOT/reg['view'];manifest=read(view/'manifest.json');observed=read(view/'progress.json')
    if manifest['status']!='verified_all148_train_macrostep_loss_view' or manifest['development_or_reserved_used'] or observed['status']!='completed_all148_train_macrostep_loss_view':
        raise ValueError('Complete train-only supervised view required')
    rows=[]
    for part in manifest['parts']:
        s=part['stages'];path=view/part['path']
        if digest(path)!=part['sha256']:raise ValueError('Loss data changed')
        with np.load(path,allow_pickle=False) as archive:
            arrays={n:archive[n].copy() for n in archive.files}
        if set(arrays)!=set(part['arrays']):raise ValueError('Complete typed supervised arrays required')
        for n,a in arrays.items():
            spec=part['arrays'][n]
            if a.dtype.str!=spec['dtype'] or list(a.shape)!=spec['shape'] or not np.isfinite(a).all():raise ValueError('Typed finite objective data required')
        if not np.array_equal(arrays['scene_index'],np.arange(148)) or sum(arrays['teacher_substeps']==8)!=29 or sum(arrays['teacher_substeps']==1)!=119:
            raise ValueError('All148scene weighting and1/8teacher coverage required')
        design=torch.from_numpy(arrays['design']).to('cuda');head=MacroReadout(s)
        before=head.residual_weights.detach().cpu().numpy().tobytes();version=head.residual_weights._version
        loss=head.objective(design);loss.backward()
        grad=head.residual_weights.grad
        if grad is None or grad.device.type!='cuda' or not bool(grad.isfinite().all()) or not bool(grad.abs().sum()>0):raise ValueError('Finite nonzero CUDA gradients required')
        expected=float(np.mean([m['zero_readout_relative_error']**2 for r in observed['rows'] if r['student_stages']==s for m in r['metrics'].values()]))
        if abs(float(loss.detach())-expected)>5e-12*max(expected,1e-30)+1e-28:raise ValueError('Complete full-grid zero-head objective differs from compact loss')
        with torch.no_grad():
            per_metric=head.residuals(design).square().sum(-1).sqrt()
            mask=torch.from_numpy(arrays['teacher_substeps']==1).to('cuda')
            if bool((per_metric[mask]>1e-5).any()):raise ValueError('UnchangedN1branch fails a retained training criterion')
        if head.residual_weights._version!=version or head.residual_weights.detach().cpu().numpy().tobytes()!=before:
            raise ValueError('Zero-update qualification changed learned parameters')
        rows.append(dict(stages=s,parameters=3*s,loss=float(loss.detach()),all148training_scenes_in_objective=True,
            every_N1_retention_metric_pass=True,gradient_norms=grad.norm(dim=1).cpu().tolist(),
            initial_worst_relative_by_metric=per_metric.amax(dim=(0,1)).cpu().tolist(),
            parameter_sha256=hashlib.sha256(before).hexdigest(),parameter_version_unchanged=True))
    if {r['stages'] for r in rows}!={3,4}:raise ValueError('Both registered student architectures required')
    if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
        raise ValueError('Objective qualification memory cap exceeded')
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Objective dependency changed during qualification: '+p)
    args.output.mkdir(parents=True)
    write(args.output/'results.json',dict(status='passed',rows=rows,all148training_scenes_in_objective=True,
        optimizer_updates=0,cpu_model_forward=False,all_learning_and_objective_on_cuda=True,
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scientific_model_accuracy_pass=False,latency_claim=False,expanded_reserved52_references_opened=0,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
    for row in rows:print('Zero-update CUDA readout',row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
