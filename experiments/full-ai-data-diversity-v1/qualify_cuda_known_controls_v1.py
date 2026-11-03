"""Compare actual CUDA prescribed fields to the unchanged native control program."""
import argparse
import os
from pathlib import Path
import torch
from support import ROOT,read,write,digest
from cuda_known_controls_v1 import CudaKnownControls
from expanded_cuda_generator_v1 import prescribed_controls

def run(args):
    if args.output.exists():raise ValueError('Fresh known-control qualification evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only candidate required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen known-control source changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model=CudaKnownControls(reg['build']);args.output.mkdir(parents=True)
    result=dict(status='running',rows=[],optimizer_updates=0,cpu_model_forward=False,physical_state_accessed=False,
        inference_device='cuda',gpu_model=torch.cuda.get_device_name(),protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
        expanded_reserved52_references_opened=0,latency_claim=False)
    try:
        with torch.inference_mode():
            for selected in read(ROOT/reg['bank'])['trajectories']:
                receipt=read(ROOT/selected['receipt']);scene=receipt['scene'];n=selected['reference_substeps']
                for start in reg['starts']:
                    expected=prescribed_controls(scene,start,4,n);actual=model.generate(scene,start,4,n)
                    repeat=model.generate(scene,start,4,n)
                    if len(expected)!=len(actual):raise ValueError('All fine controls required')
                    for fine,(a,b,c) in enumerate(zip(expected,actual,repeat)):
                        for component,(left,right,repeated) in enumerate(zip(a,b,c)):
                            if right.dtype!=torch.float64 or right.device.type!='cuda' or right.shape!=left.shape or not bool(right.isfinite().all()):raise ValueError('Exact finite CUDA double control layout required')
                            if not torch.equal(right.view(torch.int64),repeated.view(torch.int64)):raise ValueError('Repeated known controls differ bitwise')
                            delta=left-right;relative=float(delta.norm()/left.norm().clamp_min(1e-30));scaled=float(delta.abs().max()/left.abs().max().clamp_min(1.))
                            passed=relative<=reg['maximum_relative_l2_error'] and scaled<=reg['maximum_scaled_absolute_error']
                            result['rows'].append(dict(identifier=selected['identifier'],start=start,fine=fine,component=component,
                                cells_per_metre=scene['cells_per_metre'],counts_xyz=scene['counts'],internal_substeps=n,
                                relative_l2_error=relative,scaled_absolute_error=scaled,maximum_absolute_error=float(delta.abs().max()),
                                bitwise_equal=torch.equal(left.view(torch.int64),right.view(torch.int64)),all_accuracy_pass=passed))
                write(args.output/'results.json',result);print('CUDA known control qualified',selected['identifier'],flush=True)
        for path,want in reg['source_sha256'].items():
            if digest(ROOT/path)!=want:raise ValueError('Known-control qualification source changed: '+path)
        if len(result['rows'])!=reg['expected_tensors']:raise ValueError('Full registered fine-control population required')
        result.update(status='passed' if all(r['all_accuracy_pass'] for r in result['rows']) else 'failed_accuracy',
            all_accuracy_pass=all(r['all_accuracy_pass'] for r in result['rows']),worst_relative_l2_error=max(r['relative_l2_error'] for r in result['rows']),
            worst_scaled_absolute_error=max(r['scaled_absolute_error'] for r in result['rows']),every_repeat_bitwise_deterministic=True)
        if not result['all_accuracy_pass']:raise ValueError('Known-control precision bound failed')
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:model.close();write(args.output/'results.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
