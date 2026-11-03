"""Minimal captured-history GPU replay for the registered equation drift failure."""
import argparse
import hashlib
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from expanded_cuda_graph_v1 import ExpandedCudaGraphGenerator
from cuda_divisor_generator_v1 import FIELDS,prescribed_inputs
from diagnose_causal_composition import measure,inventory


def run(args):
    if args.output.exists():raise ValueError('Fresh captured failure replay required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only learned replay required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Captured failure evidence changed: '+path)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=read(ROOT/reg['receipt']);scene=receipt['scene'];norm=read(ROOT/reg['normalization']);start=reg['start'];stop=reg['stop']
    with np.load(ROOT/reg['initial_archive'],allow_pickle=False) as archive:
        current={name:torch.from_numpy(archive[f'step{start}_{name}'].copy()).to('cuda') for name in FIELDS}
    model=ExpandedCudaGraphGenerator(reg['pair'],reg['shared_pair']);before=inventory(model)
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    began=time.monotonic();model.prepare_geometry(tuple(scene['counts']))
    try:
        with torch.inference_mode():
            model.prepare_block(current,scene,start,1,prescribed_inputs(scene,start,1))
            penultimate=None
            for step in range(start,stop):
                if step==stop-1:penultimate={name:value.detach().cpu().numpy().copy() for name,value in current.items()}
                current=model.predict(current,scene,step,1,prescribed_inputs(scene,step,1))[0]
            output={name:value.cpu().contiguous() for name,value in current.items()}
            actual={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in output.items()}
            if actual!=reg['expected_output_sha256']:raise ValueError('Reproduction differs from the actual registered failing output bytes')
            spec=next(row for row in receipt['states'] if row['step']==stop);truth_path=(ROOT/reg['receipt']).parent/spec['path']
            if digest(truth_path)!=spec['sha256']:raise ValueError('Original matching native label changed')
            with np.load(truth_path,allow_pickle=False) as archive:truth={name:archive[name].copy() for name in (*FIELDS,'rhs')}
            metrics=measure(output,truth,scene,norm);bad=metrics['relative']['equation']>reg['maximum_error']
            if inventory(model)!=before:raise ValueError('Replay changed trained tensors')
            snapshot=args.output/f'captured-step{stop-1}.npz';np.savez_compressed(snapshot,**{f'step{stop-1}_{name}':value for name,value in penultimate.items()})
            write(args.output/'results.json',dict(status='reproduced_registered_equation_residual_failure' if bad else 'registered_failure_not_reproduced',
                inference_device='cuda',start=start,stop=stop,learned_advances=stop-start,
                actual_original_failing_output_bytes_bitwise_equal=True,metrics=metrics,failed_original_criterion=bad,
                penultimate=dict(path=snapshot.name,sha256=digest(snapshot)),elapsed_seconds=time.monotonic()-began,
                optimizer_updates=0,cpu_model_forward=False,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
            print('Captured GPU replay',start,'to',stop,'equation error',metrics['relative']['equation'],'threshold',reg['maximum_error'],'original output bytes match',True,flush=True)
            return 2 if bad else 0
    finally:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);raise SystemExit(run(parser.parse_args()))
