"""Bounded exact-output and complete-call GPU diagnostic of unchanged graph execution."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from internal_residual_generator_v2 import InternalResidualGenerator
from internal_residual_fused_graph_v2 import InternalResidualFusedGraphGenerator as InternalResidualGraphGenerator
from cuda_native_transfer_v1 import FIELDS, upload_state
from benchmark_internal_residual_v1 import hashes
from capture_memory_telemetry_v1 import process_memory
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only graph diagnostic required')
    reg = read(args.protocol)
    for p, expected in reg['source_sha256'].items():
        if digest(ROOT / p) != expected:
            raise ValueError('Frozen graph diagnostic source changed: ' + p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native = read(ROOT / reg['native_results'])
    case = next(c for c in read(ROOT / reg['case_selection'])['cases'] if c['identifier'] == reg['identifier'])
    scene = read(ROOT / case['receipt'])['scene']
    if digest(ROOT / case['receipt']) != case['receipt_sha256']:
        raise ValueError('Exact consumed native scene required')
    expected = {(r['start'], r['K'], r['lead']): r['output_sha256'] for r in native['rows'] if r['identifier'] == reg['identifier']}
    inputs = {}
    for start in reg['starts']:
        spec = next(s for s in case['states'] if s['step'] == start)
        if digest(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('Exact native current state required')
        with np.load(ROOT / spec['path'], allow_pickle=False) as captured:
            inputs[start] = {n: torch.from_numpy(captured[n].copy()) for n in FIELDS}
    models = {name: cls(reg['residual_pair'], reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        for name, cls in [('eager', InternalResidualGenerator), ('fused_graph', InternalResidualGraphGenerator)]}
    before = {n: inventory(m) for n, m in models.items()}
    args.output.mkdir(parents=True)
    started = time.monotonic()
    rows, checked = [], []
    result = dict(status='running', inference_device='cuda', display_label='GPU (CUDA)',
        full_generator_target_pass=False, acceptance_latency_qualification=False,
        optimizer_updates=0, new_parameter_fits=0, cpu_model_forward=False,
        expanded_reserved52_references_opened=0)
    def verify(output, start, k):
        if len(output) != k or [hashes(o) for o in output] != [expected[start, k, lead] for lead in range(1, k + 1)]:
            raise ValueError('CUDAgraph changed native-scored output bytes')
    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds'] or
                torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes'] or
                process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes']):
            raise ValueError('Finite graph diagnostic resource budget exceeded')
    try:
        setup = time.perf_counter()
        retained = None
        for start, fields in inputs.items():
            initial = hashes(fields)
            for k in (1, 2, 3, 4):
                output = models['fused_graph'].predict_native(fields, scene, start, k, case['reference_substeps'])
                verify(output, start, k)
                if retained is not None and hashes(retained[0]) != retained[1]:
                    raise ValueError('Replay overwrote retained caller output')
                retained = (output[-1], hashes(output[-1]))
                if hashes(fields) != initial:
                    raise ValueError('Graph changed caller input')
                checked.append(dict(start=start, K=k, complete_outputs=k))
                bounded()
        setup = time.perf_counter() - setup
        # Also exercise resident output ownership through another replay.
        fields = inputs[reg['timing_start']]
        resident = upload_state(fields)
        own = models['fused_graph'].predict(resident, scene, reg['timing_start'], 2, case['reference_substeps'])
        saved = [hashes({n: v.cpu() for n, v in o.items()}) for o in own]
        models['fused_graph'].predict(resident, scene, reg['timing_start'] + 1, 2, case['reference_substeps'])
        if saved != [hashes({n: v.cpu() for n, v in o.items()}) for o in own]:
            raise ValueError('Resident outputs alias future replays')
        for model in models.values():
            for k in (1, 2, 3, 4):
                verify(model.predict_native(fields, scene, reg['timing_start'], k, case['reference_substeps']), reg['timing_start'], k)
        for repeat in range(5):
            for k in ((1, 2, 3, 4) if repeat % 2 == 0 else (4, 3, 2, 1)):
                for mode in (('eager', 'fused_graph') if repeat % 2 == 0 else ('fused_graph', 'eager')):
                    torch.cuda.synchronize()
                    tick = time.perf_counter_ns()
                    output = models[mode].predict_native(fields, scene, reg['timing_start'], k, case['reference_substeps'])
                    elapsed = (time.perf_counter_ns() - tick) / 1e6
                    verify(output, reg['timing_start'], k)
                    rows.append(dict(mode=mode, repeat=repeat, K=k, block_ms=elapsed, ms_per_generated_step=elapsed/k))
                    bounded()
        if any(inventory(m) != before[n] for n, m in models.items()):
            raise ValueError('Graph execution changed learned parameters')
        summaries = []
        for mode in models:
            for k in (1, 2, 3, 4):
                values = [r['ms_per_generated_step'] for r in rows if r['mode'] == mode and r['K'] == k]
                summaries.append(dict(mode=mode, K=k, samples=len(values), median_ms_per_generated_step=statistics.median(values), maximum_ms_per_generated_step=max(values)))
        result.update(status='completed_exact_fused_graph_subset_and_paired_GPU_diagnostic',
            rows=rows, summaries=summaries, checked=checked, setup_seconds=setup,
            identifier=reg['identifier'], cells_per_metre=scene['cells_per_metre'], counts_xyz=scene['counts'],
            every_checked_output_byte_exact=True, retained_native_and_resident_outputs_own_storage=True,
            GPU_model=torch.cuda.get_device_name(), native_accuracy_failure_preserved=not native['all_accuracy_pass'],
            full_native_population_qualified=False, learned_parameters_unchanged=True,
            scope='Complete native input/output wall-clock, all fields and intermediate outputs, H2D/D2H and synchronization. Graph construction is measured separately as setup. Single-scene diagnostic; no acceptance latency claim.',
            memory_counters=process_memory(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    except BaseException as error:
        result.update(status='failed', error=repr(error), rows=rows, checked=checked)
        raise
    finally:
        for model in models.values():
            model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
