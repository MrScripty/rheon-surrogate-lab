"""Bounded GPU operator profile of a native-scored complete residual generator."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from internal_residual_generator_v1 import InternalResidualGenerator
from cuda_native_transfer_v1 import FIELDS
from benchmark_internal_residual_v1 import hashes


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only profile required')
    reg = read(args.protocol)
    for p, expected in reg['source_sha256'].items():
        if digest(ROOT / p) != expected:
            raise ValueError('Frozen profile source changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native = read(ROOT / reg['native_results'])
    case = next(c for c in read(ROOT / reg['case_selection'])['cases'] if c['identifier'] == reg['identifier'])
    spec = next(s for s in case['states'] if s['step'] == reg['start'])
    if digest(ROOT / spec['path']) != spec['sha256'] or digest(ROOT / case['receipt']) != case['receipt_sha256']:
        raise ValueError('Exact consumed native current state required')
    with np.load(ROOT / spec['path'], allow_pickle=False) as captured:
        fields = {n: torch.from_numpy(captured[n].copy()) for n in FIELDS}
    scene = read(ROOT / case['receipt'])['scene']
    expected = [r['output_sha256'] for r in native['rows'] if
        r['identifier'] == reg['identifier'] and r['start'] == reg['start'] and r['K'] == reg['K']]
    if len(expected) != reg['K']:
        raise ValueError('Complete native-scored profile outputs required')
    model = InternalResidualGenerator(reg['residual_pair'], reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    args.output.mkdir(parents=True)
    try:
        initial = hashes(fields)
        for _ in range(3):
            outputs = model.predict_native(fields, scene, reg['start'], reg['K'], case['reference_substeps'])
            if [hashes(o) for o in outputs] != expected:
                raise ValueError('Profile warmup differs from native-scored outputs')
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True) as profiler:
            for _ in range(3):
                outputs = model.predict_native(fields, scene, reg['start'], reg['K'], case['reference_substeps'])
                torch.cuda.synchronize()
        if time.monotonic() - tick > reg['maximum_profile_seconds']:
            raise ValueError('Bounded profile deadline exceeded')
        if hashes(fields) != initial or [hashes(o) for o in outputs] != expected:
            raise ValueError('Profile fields or caller input changed')
        rows = [dict(name=event.key, device_type=str(event.device_type), calls=event.count, self_cpu_us=event.self_cpu_time_total,
                     total_cpu_us=event.cpu_time_total, self_device_us=event.self_device_time_total,
                     total_device_us=event.device_time_total, input_shapes=event.input_shapes)
                for event in profiler.key_averages(group_by_input_shape=True)]
        write(args.output / 'results.json', dict(status='completed_diagnostic_GPU_operator_profile',
            rows=rows, repeats=3, K=reg['K'], identifier=reg['identifier'], start=reg['start'],
            counts_xyz=case['counts_xyz'], cells_per_metre=scene['cells_per_metre'],
            GPU_model=torch.cuda.get_device_name(), inference_device='cuda', display_label='GPU (CUDA)',
            scope='Profiler-instrumented complete native-input/output calls. CPU event durations include host dispatch; device event durations describe GPU work. Neither is acceptance latency.',
            exact_native_scored_outputs=True, caller_fields_unchanged=True,
            acceptance_latency_qualification=False, cpu_model_forward=False, optimizer_updates=0,
            new_parameter_fits=0, expanded_reserved52_references_opened=0, full_generator_target_pass=False,
            source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol)))
        print('GPU operator profile complete; native-scored output bytes unchanged.', flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
