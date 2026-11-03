"""Finite GPU-only encoding/deployment-precision experiment on one failing rollout."""
import argparse
import hashlib
import math
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from expanded_cuda_graph_v1 import ExpandedCudaGraphGenerator
from cuda_divisor_generator_v1 import FIELDS,prescribed_inputs
from diagnose_causal_composition import measure,inventory


def run(args):
    if args.output.exists():raise ValueError('Fresh precision probe evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only inference required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen precision experiment source changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=read(ROOT/reg['receipt']);scene=receipt['scene'];norm=read(ROOT/reg['normalization']);truth={}
    for spec in receipt['states']:
        path=(ROOT/reg['receipt']).parent/spec['path']
        if digest(path)!=spec['sha256']:raise ValueError('Native rollout labels changed')
        with np.load(path,allow_pickle=False) as archive:truth[spec['step']]={name:archive[name].copy() for name in (*FIELDS,'rhs')}
    args.output=args.output.resolve();args.output.mkdir(parents=True);rows=[];summaries=[]
    with torch.inference_mode():
        for variant in reg['variants']:
            graph=ExpandedCudaGraphGenerator(reg['pair'],reg['shared_pair']);raw=inventory(graph.model.shared_pressure)
            if variant['coefficient_quantization_step'] is not None:
                step=variant['coefficient_quantization_step'];parameter=graph.model.shared_pressure.coefficients
                parameter.copy_((parameter/step).round()*step)
            if variant['fixed_feature_encoding']=='numpy_training_order':
                head=graph.model.shared_pressure
                def encoded_response(shape):
                    mode=np.indices(shape,dtype=np.int64)
                    features=np.stack([np.ones(shape),*(np.cos(math.pi*mode[axis]/shape[axis]) for axis in (2,1,0))],axis=-1).astype('<f8')
                    # Fixed features only onCPU; learned affine readout staysCUDA.
                    return head(torch.from_numpy(features).to('cuda'))
                head.response_grid=encoded_response
            deployed=inventory(graph.model.shared_pressure);before=inventory(graph)
            graph.prepare_geometry(tuple(scene['counts']));current={name:torch.from_numpy(truth[0][name]).to('cuda') for name in FIELDS}
            graph.prepare_block(current,scene,0,1,prescribed_inputs(scene,0,1));saved={};selected=[]
            try:
                for step in range(240):
                    current=graph.predict(current,scene,step,1,prescribed_inputs(scene,step,1))[0]
                    output={name:value.cpu().contiguous() for name,value in current.items()};actual=step+1
                    identities={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in output.items()}
                    if variant['id']=='raw_cuda' and identities!=reg['expected_raw_output_sha256'][str(actual)]:
                        raise ValueError('Control precision variant differs from actual failing rollout bytes')
                    metrics=measure(output,truth[actual],scene,norm)
                    passed=all(value<=reg['maximum_error'] for value in metrics['relative'].values())
                    record=dict(variant=variant['id'],step=actual,metrics=metrics,all_accuracy_pass=passed,output_sha256=identities)
                    rows.append(record);selected.append(record)
                    if actual in (141,216,240):
                        for name,value in output.items():saved[f'step{actual}_{name}']=value.numpy().copy()
                if inventory(graph)!=before:raise ValueError('Probe changed deployed coefficients during inference')
                snapshot=args.output/(variant['id']+'-snapshots.npz');np.savez_compressed(snapshot,**saved)
                summary=dict(variant=variant,all_accuracy_pass=all(row['all_accuracy_pass'] for row in selected),
                    failed_outputs=sum(not row['all_accuracy_pass'] for row in selected),
                    worst_error={field:max(row['metrics']['relative'][field] for row in selected) for field in ('equation','pressure','velocity','tracer')},
                    raw_parameter_inventory=raw,deployed_parameter_inventory=deployed,snapshot=dict(path=snapshot.name,sha256=digest(snapshot)))
                summaries.append(summary);print('GPU precision probe',variant['id'],'failed',summary['failed_outputs'],'worst equation',summary['worst_error']['equation'],flush=True)
            finally:graph.close()
            write(args.output/'progress.json',dict(status='running',completed=summaries))
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Precision experiment source changed during execution: '+path)
    write(args.output/'results.json',dict(status='completed_registered_gpu_precision_probe',rows=rows,summaries=summaries,
        inference_device='cuda',cpu_model_forward=False,optimizer_updates=0,raw_managed_artifacts_unchanged=True,
        population='One previously failing development/consumed regression scene, K1,240outputs; not all-scene or independent qualification.',
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],full_generator_target_pass=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
