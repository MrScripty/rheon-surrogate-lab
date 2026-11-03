"""Finite GPU-only diagnostic profile; instrumented times are not benchmarks."""
import argparse
import collections
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT, read, write, digest
from cuda_graph_divisor_v2 import CudaGraphDivisorGeneratorV2
from cuda_divisor_generator_v1 import FIELDS, prescribed_inputs
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists():raise ValueError('Fresh GPU profile required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only inference required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen profile dependency changed: '+path)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model=CudaGraphDivisorGeneratorV2(reg['pair'],reg['pressure_pair']);before=inventory(model)
    path=ROOT/reg['trajectory']['receipt'];receipt=read(path)
    if digest(path)!=reg['trajectory']['receipt_sha256']:raise ValueError('Profile reference changed')
    spec=next(row for row in receipt['states'] if row['step']==reg['start'])
    source=path.parent/spec['path']
    if digest(source)!=spec['sha256']:raise ValueError('Profile state changed')
    with np.load(source,allow_pickle=False) as archive:
        state={name:torch.from_numpy(archive[name].copy()).to('cuda') for name in FIELDS}
    scene=receipt['scene'];external=prescribed_inputs(scene,reg['start'],reg['K'])
    model.prepare_geometry(tuple(scene['counts']));model.prepare_block(state,scene,reg['start'],reg['K'],external)
    rows=[]
    try:
        with torch.inference_mode():
            for mode,owner in [('eager',model.model),('graph',model)]:
                for _ in range(reg['warmup_blocks']):owner.predict(state,scene,reg['start'],reg['K'],external)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(reg['profiled_blocks']):owner.predict(state,scene,reg['start'],reg['K'],external)
                    torch.cuda.synchronize()
                events=list(prof.events());cuda=[event for event in events if event.device_type==torch.autograd.DeviceType.CUDA]
                kinds=collections.Counter(event.name for event in cuda)
                durations=collections.defaultdict(float)
                for event in cuda:durations[event.name]+=event.device_time_total
                rows.append(dict(mode=mode,profiled_blocks=reg['profiled_blocks'],cuda_activity_events=len(cuda),
                    cuda_activity_events_per_generated_step=len(cuda)/(reg['profiled_blocks']*reg['K']),
                    cuda_activity_total_ms=sum(event.device_time_total for event in cuda)/1000,
                    host_self_cpu_total_ms=sum(event.self_cpu_time_total for event in events if event.device_type==torch.autograd.DeviceType.CPU)/1000,
                    launch_api_counts=dict(collections.Counter(event.name for event in events if 'Launch' in event.name)),
                    top_cuda_activities=[dict(name=name,calls=kinds[name],total_ms=duration/1000)
                        for name,duration in sorted(durations.items(),key=lambda item:item[1],reverse=True)[:15]],
                    activity_label='CUPTI CUDA activities, including copies/memsets if present; not necessarily one event per source operator'))
                print('GPU diagnostic profile',mode,'CUDA activities',len(cuda),'for',reg['profiled_blocks'],'blocks',flush=True)
        if inventory(model)!=before:raise ValueError('Profiling changed learned tensors')
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('Profile dependency changed during execution: '+path)
        args.output.mkdir(parents=True)
        write(args.output/'results.json',dict(status='completed_cuda_execution_profile',inference_device='cuda',display_label='GPU (CUDA)',
            gpu_model=torch.cuda.get_device_name(),rows=rows,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
            counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],K=reg['K'],start=reg['start'],
            diagnostic_only=True,instrumented_times_are_not_acceptance_benchmarks=True,
            optimizer_updates=0,cpu_model_forward=False,latency_target_pass=False))
    finally:model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
