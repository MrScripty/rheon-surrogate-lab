"""Bounded current-backend complete-entry profile; no acceptance timing claim."""
import argparse
import collections
import hashlib
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.scalar_geometry_graph_v1 import ScalarGeometryGraph as NoScratchFillGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory


def hashes(outputs):
    return [{n: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n, v in row.items()} for row in outputs]


def run(args):
    if args.output.exists():
        raise ValueError('Fresh bounded profile evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('GPU-only profile required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen profile dependency changed: ' + path)
    for path in reg['full_history_results']:
        q = read(ROOT / path)
        if not q['all_accuracy_pass'] or q['status'] != 'completed_streamed_expanded_gpu_240step_scene':
            raise ValueError('All original full-history accuracy cases must pass before profiling')
    for proof in reg['execution_equivalence']:
        evidence = read(ROOT / proof['path'])
        if evidence['status'] != proof['status'] or len(evidence['rows']) != proof['complete_outputs'] or not evidence[proof['required_true']]:
            raise ValueError('All exact affine execution gates required before profiling')
    baseline = read(ROOT / reg['baseline_outputs'])
    if not baseline['all_accuracy_pass']:
        raise ValueError('Current complete native-qualified baseline required')
    expected = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256'] for r in baseline['rows']}
    population = {r['identifier']: r for r in read(ROOT / reg['bank'])['trajectories']}
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = NoScratchFillGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    rows = []
    args.output.mkdir(parents=True)
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                selected = population[case['identifier']]
                if digest(ROOT / selected['receipt']) != selected['receipt_sha256']:
                    raise ValueError('Profile scene receipt changed')
                scene = read(ROOT / selected['receipt'])['scene']
                start, k, n = case['start'], case['K'], selected['reference_substeps']
                spec = next(s for s in selected['states'] if s['step'] == start)
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Actual current native profile state changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    host = {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
                resident = upload_state(host)
                model.prepare_geometry(tuple(scene['counts']))
                model.prepare_block(resident, scene, start, k, n)
                golden = [expected[(case['identifier'], start, k, lead)] for lead in range(1, k + 1)]
                for _ in range(reg['warmup_blocks']):
                    copied = download_outputs(model.predict(resident, scene, start, k, n))
                    if hashes(copied) != golden:
                        raise ValueError('Warmup differs from qualified actual field bytes')
                captured = []
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(reg['profiled_blocks']):
                        with torch.profiler.record_function('complete_native_upload'):
                            uploaded = upload_state(host)
                        with torch.profiler.record_function('complete_dynamic_GPU_generator'):
                            outputs = model.predict(uploaded, scene, start, k, n)
                        with torch.profiler.record_function('complete_native_download'):
                            captured.append(download_outputs(outputs))
                    torch.cuda.synchronize()
                if any(hashes(outputs) != golden for outputs in captured):
                    raise ValueError('Actual profiled complete outputs differ from qualified bytes')
                events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                calls = collections.Counter(e.name for e in events)
                totals = collections.defaultdict(float)
                for e in events:
                    totals[e.name] += float(e.device_time_total)
                families = collections.defaultdict(lambda: dict(calls=0, instrumented_ms=0.))
                for name, micros in totals.items():
                    family = next((label for label in ('fixed_axis_transform', 'learned_transport', 'learned_linear', 'learned_gather', 'known_axes_dynamic', 'known_control_dynamic') if label in name), 'other_CUDA_activity')
                    if 'geometry_transport_' in name:
                        family = 'learned_transport'
                    families[family]['calls'] += calls[name]
                    families[family]['instrumented_ms'] += micros / 1000
                rows.append(dict(identifier=case['identifier'], start=start, K=k, counts_xyz=scene['counts'],
                    cells_per_metre=scene['cells_per_metre'], internal_substeps=n, actual_output_sha256=golden,
                    profiled_complete_calls=reg['profiled_blocks'], cuda_activity_events=len(events),
                    cuda_activity_events_per_generated_step=len(events) / (reg['profiled_blocks'] * k),
                    families=dict(families),
                    top_cuda_activities=[dict(name=name, calls=calls[name], instrumented_ms=micros / 1000)
                                         for name, micros in sorted(totals.items(), key=lambda p: p[1], reverse=True)[:25]],
                    top_cpu_activities=[dict(name=e.key, calls=int(e.count), self_cpu_ms=float(e.self_cpu_time_total) / 1000)
                                        for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:25]]))
                model.release_graphs()
                print('Exact current GPU profile', case['identifier'], 'N', n, 'CUDAactivities/output', rows[-1]['cuda_activity_events_per_generated_step'], flush=True)
            if inventory(model) != before:
                raise ValueError('Profiler changed learned parameters')
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT / path) != expected:
                raise ValueError('Profile source/evidence changed during execution: ' + path)
        write(args.output / 'results.json', dict(status='completed_current_generic_GPU_complete_entry_profile', rows=rows,
            inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
            diagnostic_only=True, latency_acceptance_claim=False, components_not_assumed_additive=True,
            actual_profiled_and_warmup_outputs_match_native_qualified_baseline=True,
            cpu_model_forward=False, optimizer_updates=0, expanded_reserved52_references_opened=0,
            graph_setup_observations=model.setup_observations, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256']))
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
