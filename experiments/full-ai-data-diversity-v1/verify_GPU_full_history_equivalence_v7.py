"""GPU-only own-history byte proof against all24development and two consumed larger native-qualified runs.

After native state0, candidate inputs are exclusively its own generated fields
and prescribed scene controls. Reference evidence consists of the already
qualified hashes, never future physical inputs. No teacher or timing ROI runs.
"""
import argparse
import hashlib
import os
import resource
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils import deterministic
from support import ROOT, read, write, digest
from debug.gather_fused_graph_v1 import GatherFusedGraph as NoScratchFillGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists():
        raise ValueError('Fresh complete-history byte-equivalence evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('GPU-only inference required')
    if len(reg['cases']) != 26 or len({c['identifier'] for c in reg['cases']}) != 26:
        raise ValueError('All24development plus two distinct consumed larger cases required')
    if sum(c['split']=='development' for c in reg['cases'])!=24 or sum(c['split']=='train' for c in reg['cases'])!=2:
        raise ValueError('Exactly24development and two consumed training cases required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen equivalence dependency changed: ' + path)
    proof = read(ROOT / reg['block_qualification'])
    if not proof['all_accuracy_pass'] or not (proof['all_non64_outputs_bitwise_equal_corrected_baseline'] and proof['all64_outputs_bitwise_equal_native']):
        raise ValueError('Every current block must match the native-qualified baseline first')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    model = NoScratchFillGraph(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    started = time.monotonic()
    result = dict(status='running', active=None, rows=[], completed_cases=[],
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  every_intermediate_required=True, own_history_without_teacher_resets=True,
                  reference_teacher_executed=False, cpu_model_forward=False, optimizer_updates=0,
                  latency_claim=False, expanded_reserved52_references_opened=0,
                  output_timestep_s=1 / 60, physical_frequency_is_separate_from_render_rate=True,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save():
        write(args.output / 'results.json', result)
    save()
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                baseline = read(ROOT / case['baseline_results'])
                if (baseline['status'] != 'completed_streamed_expanded_gpu_240step_scene'
                        or not baseline['all_accuracy_pass'] or len(baseline['rows']) != 960):
                    raise ValueError('Complete original native-qualified baseline required')
                if (baseline['identifier'] != case['identifier'] or baseline['counts_xyz'] != case['counts_xyz']
                        or baseline['cells_per_metre'] != case['cells_per_metre']
                        or baseline['protocol_sha256'] != case['baseline_protocol_sha256']):
                    raise ValueError('Exact original native case/protocol binding required')
                expected = {(r['K'], r['step']): r['output_sha256'] for r in baseline['rows']}
                if len(expected) != 960 or not all(r['metrics']['all_accuracy_criteria_pass'] for r in baseline['rows']):
                    raise ValueError('Every baseline physical output must have passed native criteria')
                if digest(ROOT / case['receipt']) != case['receipt_sha256']:
                    raise ValueError('Native scene receipt changed')
                n = case['reference_substeps']
                scene = dict(read(ROOT / case['receipt'])['scene'], reference_substeps=n, forcing_control_substeps=1)
                spec = next(s for s in case['states'] if s['step'] == 0)
                if digest(ROOT / spec['path']) != spec['sha256']:
                    raise ValueError('Actual initial native fields changed')
                with np.load(ROOT / spec['path'], allow_pickle=False) as archive:
                    initial = {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
                model.prepare_geometry(tuple(scene['counts']))
                for k in reg['horizons']:
                    current = upload_state(initial)
                    model.prepare_block(current, scene, 0, k, n)
                    if not deterministic.fill_uninitialized_memory or not torch.are_deterministic_algorithms_enabled():
                        raise ValueError('Graph setup failed to restore allocation flag/deterministic algorithms')
                    for start in range(0, reg['rollout_steps'], k):
                        if time.monotonic() - started > reg['maximum_wall_seconds']:
                            raise TimeoutError('Full-history equivalence deadline reached')
                        result['active'] = dict(identifier=case['identifier'], K=k, step=start)
                        outputs = model.predict(current, scene, start, k, n)
                        copied = download_outputs(outputs)
                        if len(copied) != k:
                            raise ValueError('Every K intermediate physical state required')
                        for lead, row in enumerate(copied, 1):
                            step = start + lead
                            actual = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in row.items()}
                            if actual != expected[(k, step)]:
                                raise ValueError(f'Actual output bytes changed for {case["identifier"]}, K={k}, step={step}')
                            result['rows'].append(dict(identifier=case['identifier'], K=k, step=step,
                                output_sha256=actual, native_qualified_baseline_exact=True))
                        current = outputs[-1]
                    model.release_graphs()
                    save()
                    print('GPU complete-history byte proof', case['identifier'], 'K', k,
                          reg['rollout_steps'], 'physical outputs exact', flush=True)
                if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']
                        or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
                    raise ValueError('Full-history equivalence memory ceiling exceeded')
                result['completed_cases'].append(dict(identifier=case['identifier'], counts_xyz=case['counts_xyz'],
                    cells_per_metre=case['cells_per_metre'], internal_substeps=n, complete_outputs=960,
                    baseline_results=case['baseline_results'], baseline_sha256=digest(ROOT / case['baseline_results']),
                    baseline_native_worst_error=baseline['worst_error']))
                save()
        if len(result['rows']) != reg['expected_complete_outputs'] or inventory(model) != before:
            raise ValueError('Full population and unchanged learned tensors required')
        actual_keys = {(r['identifier'], r['K'], r['step']) for r in result['rows']}
        required_keys = {(c['identifier'], k, step) for c in reg['cases'] for k in reg['horizons']
                         for step in range(1, reg['rollout_steps'] + 1)}
        if actual_keys != required_keys:
            raise ValueError('Every case/K/physical step must be verified exactly once')
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT / path) != expected:
                raise ValueError('Equivalence evidence changed during replay: ' + path)
        result.update(status='passed_all26_GPU_own_history_byte_equivalence', active=None,
                      every_output_exact_to_native_qualified_baseline=True, learned_parameters_unchanged=True,
                      deterministic_algorithms_and_allocation_flag_restored=True,
                      elapsed_diagnostic_seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        model.close()
        result['active'] = None
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
