"""One balanced complete-call GPU study: allocation-fill removal versus generic affine fusion.

Requires all four uninterrupted accuracy replays and all exact execution-equivalence suites
to have completed successfully. Five warmups and five balanced repeats are fixed;
every actual timed/warmup output is checked against the current qualified bytes.
"""
import argparse
import collections
import hashlib
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.affine_generator_graph_v2 import AffineGeneratorGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from debug.no_scratch_fill_graph_v1 import NoScratchFillGraph
from diagnose_causal_composition import inventory


def hashes(outputs):
    return [{name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
             for name, value in row.items()} for row in outputs]


def run(args):
    if args.output.exists():
        raise ValueError('Fresh finite benchmark evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('GPU-only inference required')
    if reg['modes'] != ['no_scratch_fill', 'affine_fused'] or reg['repetitions'] != 5 or reg['warmup_calls'] != 5:
        raise ValueError('Exactly the registered two modes/five balanced repeats required')
    if reg['horizons'] != [1, 2, 3, 4]:
        raise ValueError('All four registered generator horizons required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen benchmark dependency changed: ' + path)
    for proof in reg['execution_equivalence']:
        q = read(ROOT / proof['path'])
        if (q['status'] != proof['status'] or len(q['rows']) != proof['complete_outputs']
                or not q[proof['required_true']]):
            raise ValueError('All execution equivalence gates must pass before timing')
    if len(reg['full_streamed_qualifications']) != 4:
        raise ValueError('Every fixed full-history accuracy case required before timing')
    accuracy_protocol = read(ROOT / reg['accuracy_protocol'])
    cases = {case['identifier']: case for case in accuracy_protocol['cases']}
    seen_cases = set()
    for path in reg['full_streamed_qualifications']:
        q = read(ROOT / path)
        custody = q['native_process']
        if (q['status'] != 'completed_streamed_expanded_gpu_240step_scene'
                or not q['all_accuracy_pass'] or len(q['rows']) != 960
                or not custody['reaped'] or custody['returncode'] != 0
                or not custody['graceful_protocol_complete']):
            raise ValueError('All native references must terminate and every accuracy output must pass')
        if q['identifier'] in seen_cases or q['identifier'] not in cases:
            raise ValueError('Each fixed full-history scene must qualify exactly once')
        case = cases[q['identifier']]
        if (q['counts_xyz'] != case['counts_xyz'] or q['cells_per_metre'] != case['cells_per_metre']
                or q['internal_substeps'] != case['reference_substeps']
                or q['protocol_sha256'] != digest(ROOT / reg['accuracy_protocol'])):
            raise ValueError('Full-history proof must use the exact current candidate protocol and geometry')
        seen_cases.add(q['identifier'])
    if seen_cases != set(cases):
        raise ValueError('Complete fixed native-reference population required')
    baseline = read(ROOT / reg['baseline_outputs'])
    expected = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256'] for r in baseline['rows']}
    if not baseline['all_accuracy_pass'] or len(expected) != 960:
        raise ValueError('Complete native-qualified current-backend block outputs required')
    population = {r['identifier']: r for r in read(ROOT / reg['bank'])['trajectories']}
    windows = {}
    for source in reg['window_sources']:
        for row in read(ROOT / source['controller'])['completed']:
            if row['identifier'] in windows:
                raise ValueError('Unique native window population required')
            windows[row['identifier']] = row
    if len(population) != 24 or set(population) != set(windows):
        raise ValueError('All24development scenes required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    models = {
        'no_scratch_fill': NoScratchFillGraph(reg['transport_pair'], reg['shared_pair'], reg['baseline_build'], reg['control_build']),
        'affine_fused': AffineGeneratorGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])}
    torch.cuda.synchronize()
    constructor_ms = 1000 * (time.monotonic() - started)
    before = {mode: inventory(value) for mode, value in models.items()}
    if before['no_scratch_fill'] != before['affine_fused']:
        raise ValueError('Paired models must restore exactly identical learned inventories')
    args.output.mkdir(parents=True)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = dict(status='running', active=None, rows=[], first_pre_warmup_calls=[], setups=[], summaries=[],
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=props.name,
                  gpu_uuid=str(getattr(props, 'uuid', 'unavailable')),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  driver_version=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=10).strip(),
                  host_cpu=reg['host_cpu'], pressure_dtype='float64', carrier_dtype='float32', tf32=False, autocast=False,
                  constructor_setup_ms=constructor_ms, timed_complete_outputs=0, warmup_complete_outputs=0,
                  cpu_model_forward=False, optimizer_updates=0, expanded_reserved52_references_opened=0,
                  new_independent_confirmation=False, full_generator_target_verified=False,
                  historical_CPU_latency_target_verified=False,
                  scope='Complete current native NumPy fields -> CPU tensor views -> native upload -> known-input construction and all GPU force/learned stages/internal advances -> all intermediate native CPU fields. All validation, staging/copies, synchronization and returned-output allocation/copy are included; both execution variants use the same packed adapter. Dataset reads, model/geometry/graph setup and explicit warmups are separate.',
                  maximum_is_finite_observation_not_hard_real_time=True,
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    write(args.output / 'results.json', result)
    window_index = 0
    began = time.monotonic()
    def checkpoint():
        result['graph_setup_observations'] = {mode: value.setup_observations for mode, value in models.items()}
        result['peak_gpu_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        result['elapsed_diagnostic_seconds'] = time.monotonic() - began
        write(args.output / 'results.json', result)
    def verify(outputs, identifier, start, k):
        actual = hashes(outputs)
        required = [expected[(identifier, start, k, lead)] for lead in range(1, k + 1)]
        if actual != required:
            raise ValueError('Actual measured or warmup complete output bytes differ from qualified baseline')
        return actual
    try:
        with torch.inference_mode():
            for identifier, selected in population.items():
                if digest(ROOT / selected['receipt']) != selected['receipt_sha256']:
                    raise ValueError('Native scene receipt changed')
                scene = read(ROOT / selected['receipt'])['scene']
                n = selected['reference_substeps']
                for mode, active_model in models.items():
                    tick = time.perf_counter()
                    active_model.prepare_geometry(tuple(scene['counts']))
                    torch.cuda.synchronize()
                    result['setups'].append(dict(identifier=identifier, mode=mode, geometry_setup_ms=1000 * (time.perf_counter() - tick)))
                for window in windows[identifier]['windows']:
                    if time.monotonic() - began > reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite balanced benchmark deadline reached')
                    start = window['start']
                    if digest(ROOT / window['current_state_path']) != window['current_state_sha256']:
                        raise ValueError('Current native state archive changed')
                    with np.load(ROOT / window['current_state_path'], allow_pickle=False) as archive:
                        current = {name: archive[name].copy() for name in FIELDS}
                    native = {name: torch.from_numpy(value) for name, value in current.items()}
                    resident = upload_state(native)
                    for active_model in models.values():
                        for k in reg['horizons']:
                            active_model.prepare_block(resident, scene, start, k, n)
                    def execute(mode, k):
                        host = {name: torch.from_numpy(current[name]) for name in FIELDS}
                        uploaded = upload_state(host)
                        return download_outputs(models[mode].predict(uploaded, scene, start, k, n))
                    result['active'] = dict(identifier=identifier, start=start)
                    for k in reg['horizons']:
                        modes = reg['modes'][window_index % 2:] + reg['modes'][:window_index % 2]
                        for mode in modes:
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            copied = execute(mode, k)
                            elapsed = 1000 * (time.perf_counter() - tick)
                            verify(copied, identifier, start, k)
                            result['warmup_complete_outputs'] += k
                            result['first_pre_warmup_calls'].append(dict(identifier=identifier, start=start, K=k, mode=mode,
                                full_block_ms=elapsed, ms_per_generated_step=elapsed / k, internal_substeps=n))
                        for _ in range(reg['warmup_calls']):
                            for mode in modes:
                                copied = execute(mode, k)
                                verify(copied, identifier, start, k)
                                result['warmup_complete_outputs'] += k
                    for repeat in range(reg['repetitions']):
                        shift = (repeat + window_index) % 4
                        horizons = reg['horizons'][shift:] + reg['horizons'][:shift]
                        for k in horizons:
                            shift = (repeat + window_index + k) % 2
                            modes = reg['modes'][shift:] + reg['modes'][:shift]
                            for mode in modes:
                                torch.cuda.synchronize()
                                tick = time.perf_counter()
                                copied = execute(mode, k)
                                elapsed = 1000 * (time.perf_counter() - tick)
                                actual = verify(copied, identifier, start, k)
                                result['timed_complete_outputs'] += k
                                result['rows'].append(dict(identifier=identifier, start=start, K=k, repeat=repeat, mode=mode,
                                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], internal_substeps=n,
                                    internal_advances_per_block=k * n, full_block_ms=elapsed, ms_per_generated_step=elapsed / k,
                                    actual_complete_output_sha256=actual))
                    window_index += 1
                for active_model in models.values():
                    active_model.release_graphs()
                checkpoint()
                print('Balanced complete GPU execution comparison', identifier, 'allK/four windows checked', flush=True)
        if (len(result['rows']) != reg['expected_timed_calls']
                or result['timed_complete_outputs'] != reg['expected_timed_outputs']
                or result['warmup_complete_outputs'] != reg['expected_warmup_outputs']
                or {mode: inventory(value) for mode, value in models.items()} != before):
            raise ValueError('Complete registered timed/warmup population and unchanged weights required')
        groups = collections.defaultdict(list)
        for row in result['rows']:
            groups[(row['mode'], row['K'], row['cells_per_metre'], tuple(row['counts_xyz']), row['internal_substeps'])].append(row)
        for (mode, k, density, counts, n), rows in sorted(groups.items()):
            values = [r['ms_per_generated_step'] for r in rows]
            ceiling = 1.2 * max(1., math.prod(counts) / (32**3))
            result['summaries'].append(dict(mode=mode, K=k, cells_per_metre=density, counts_xyz=list(counts), internal_substeps=n,
                measured_calls=len(rows), median_ms_per_generated_step=statistics.median(values),
                p95_ms_per_generated_step=float(np.percentile(values, 95)), maximum_ms_per_generated_step=max(values),
                voxel_budget_ms_per_generated_step=float(ceiling), observed_latency_within_voxel_budget=bool(max(values) <= ceiling)))
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT / path) != expected:
                raise ValueError('Benchmark dependency changed during measurement: ' + path)
        result.update(status='completed_balanced_execution_gpu_entry_benchmark', active=None,
                      every_actual_timed_and_warmup_output_hash_verified=True,
                      graph_setup_observations={mode: value.setup_observations for mode, value in models.items()},
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_diagnostic_seconds=time.monotonic() - began)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            for active_model in models.values():
                active_model.close()
        finally:
            result['active'] = None
            checkpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
