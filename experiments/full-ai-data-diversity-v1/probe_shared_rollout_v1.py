"""Controlled GPU pressure-map/current-state probes for the captured drift failure."""
import argparse
import hashlib
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from support import ROOT,read,write,digest
from expanded_cuda_graph_v1 import ExpandedCudaGraphGenerator
from cuda_divisor_generator_v1 import FIELDS,prescribed_inputs
from cuda_divisor_generator_v2 import CudaDivisorGeneratorV2
from diagnose_causal_composition import measure,inventory
from numerics import fine_action


def run(args):
    if args.output.exists():raise ValueError('Fresh controlled GPU probe output required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only model probes required')
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen diagnostic source changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=read(ROOT/reg['receipt']);scene=receipt['scene'];norm=read(ROOT/reg['normalization']);specs={r['step']:r for r in receipt['states']}
    def native(step):
        spec=specs[step];path=(ROOT/reg['receipt']).parent/spec['path']
        if digest(path)!=spec['sha256']:raise ValueError('Native diagnostic current/reference bytes changed')
        with np.load(path,allow_pickle=False) as archive:return {name:archive[name].copy() for name in (*FIELDS,'rhs')}
    target=native(141);rows=[]
    with torch.inference_mode():
        for start,pressure_map in [(0,'shared'),(0,'original_table'),(140,'shared'),(140,'original_table')]:
            graph=ExpandedCudaGraphGenerator(reg['pair'],reg['shared_pair']);original=None
            if pressure_map=='original_table':
                original=CudaDivisorGeneratorV2(reg['pair'],'runs/full-ai-tight-targets-v1/mixed-modal-pair-11')
                original.prepare_geometry(tuple(scene['counts']))
            graph.prepare_geometry(tuple(scene['counts']));before=inventory(graph)
            chosen=original.pressure if original is not None else graph.model.pressure;observed={}
            def capture(q,counts):
                observed['q']=q.clone()
                return chosen(q,counts)
            graph.model.pressure=capture
            initial=native(start);current={name:torch.from_numpy(initial[name]).to('cuda') for name in FIELDS}
            graph.prepare_block(current,scene,start,1,prescribed_inputs(scene,start,1))
            try:
                previous=None
                for step in range(start,141):
                    if step==140:previous=current['pressure'].cpu().clone()
                    current=graph.predict(current,scene,step,1,prescribed_inputs(scene,step,1))[0]
                output={name:value.cpu().contiguous() for name,value in current.items()};metrics=measure(output,target,scene,norm)
                identities={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in output.items()}
                expected=reg['expected_new_output'] if start==0 and pressure_map=='shared' else reg['expected_old_output'] if start==0 else None
                if expected is not None and identities!=expected:raise ValueError('Controlled pressure substitution failed to reproduce corresponding original output bytes')
                shape=(1,1,*reversed(scene['counts']));h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
                implied_rhs=fine_action(previous.reshape(shape),h,rho)+observed['q'].cpu()*(float(h[0])/1000.)
                self_error=float((fine_action(output['pressure'].reshape(shape),h,rho)-implied_rhs).norm()/implied_rhs.norm().clamp_min(1e-30))
                rows.append(dict(start=start,stop=141,pressure_map=pressure_map,native_current_reset=start==140,
                    metrics=metrics,self_equation_error_using_actual_current_q=self_error,
                    original_output_identity_verified=expected is not None,output_sha256=identities))
                if inventory(graph)!=before:raise ValueError('Controlled probe changed learned tensors')
                print('GPU probe',start,pressure_map,'native-target equation',metrics['relative']['equation'],'self-equation',self_error,flush=True)
            finally:
                graph.close()
                if original is not None:original.close()
    args.output.mkdir(parents=True)
    write(args.output/'results.json',dict(status='completed_controlled_gpu_pressure_map_and_state_probes',rows=rows,
        inference_device='cuda',optimizer_updates=0,cpu_model_forward=False,no_model_or_metric_fix_applied=True,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
