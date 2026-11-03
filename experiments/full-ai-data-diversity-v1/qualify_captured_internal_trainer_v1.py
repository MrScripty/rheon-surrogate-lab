"""Exercise the actual captured trainer's full objective without an optimizer."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from managed_input_stream_v1 import streamed_identity
    reg=json.loads(args.protocol.read_text())
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh CUDA-only trainer check required')
    for name,expected in reg['source_sha256'].items():
        if streamed_identity(ROOT/name)['sha256']!=expected:raise ValueError('Captured trainer qualification source changed')
    project=ROOT/reg['project'];assembly=json.loads((project/'assembly.json').read_text());config=assembly['config']
    for row in assembly['sources']:
        if streamed_identity(ROOT/row['path'])['sha256']!=row['sha256']:raise ValueError('Captured source changed')
    for row in assembly['static_inputs'].values():
        if streamed_identity(ROOT/row['path'])!={k:row[k] for k in ('bytes','sha256')}:raise ValueError('Typed actual static input changed')
    sys.path.insert(0,str(project));trainer=importlib.import_module('train_internal_residual_v1')
    import torch
    from capture_memory_telemetry_v1 import process_memory
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    static={n:ROOT/s['path'] for n,s in assembly['static_inputs'].items()}
    model=trainer.InternalGroupResidual(normalization=trainer.read(static['normalization']))
    before=trainer.inventory(model)
    if before!=config['qualified_initial_inventory']:raise ValueError('Actual captured initialization differs')
    view=trainer.TrainingView(static['manifest'],static['part_00']);started=time.monotonic()
    try:observed=trainer.evaluate(model,view,trainer.read(static['optimization_normalization']),config['evaluation_normalization'])
    finally:view.close()
    if trainer.inventory(model)!=before:raise ValueError('Trainer evaluation changed parameters')
    if abs(observed['objective']-config['qualified_initial_objective'])>1e-12*config['qualified_initial_objective']:
        raise ValueError('Actual captured trainer objective differs from qualified gradient program')
    memory=process_memory()
    if memory['peak_since_exec_bytes']>reg['maximum_peak_rss_bytes'] or torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes']:
        raise ValueError('Actual trainer memory gate failed')
    if time.monotonic()-started>reg['maximum_wall_seconds']:raise ValueError('Actual trainer deadline exceeded')
    args.output.mkdir(parents=True)
    trainer.write(args.output/'results.json',dict(status='passed_actual_captured_trainer_full148scene_objective',
        optimizer_updates=0,new_parameter_fits=0,all148training_scenes_in_objective=True,
        zero_update_parameters_unchanged=True,observed=observed,memory_counters=memory,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,
        source_sha256=reg['source_sha256'],cpu_model_forward=False,expanded_reserved52_references_opened=0,
        native_accuracy_claim=False,latency_claim=False,full_generator_target_pass=False))
    print('Actual captured trainer full148scene objective verified; no optimizer updates.',flush=True)


if __name__=='__main__':main()
