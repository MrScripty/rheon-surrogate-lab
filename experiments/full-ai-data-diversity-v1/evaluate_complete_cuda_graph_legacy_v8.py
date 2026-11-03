"""Score expanded shared-head GPU own-history on consumed small/32cubed regressions."""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.local_free_affine_graph_v1 import LocalFreeAffineGraph as CompleteCudaGraphGenerator
from cuda_known_controls_v3 import CudaKnownControls
from cuda_native_transfer_v1 import upload_state,download_outputs
from cuda_divisor_generator_v1 import FIELDS, prescribed_inputs
from diagnose_causal_composition import inventory, measure


def run(args):
    if args.output.exists():
        raise ValueError('Fresh GPU own-history evaluation required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda' or reg['inference_device'] != 'cuda':
        raise ValueError('Actual CUDA runtime required; no CPU fallback')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen GPU rollout dependency changed: '+path)
    prerequisite = read(ROOT/reg['block_qualification'])
    if prerequisite['status'] != 'completed_expanded_cuda_full_state_block_evaluation' or not prerequisite['all_accuracy_pass']:
        raise ValueError('Complete GPU block accuracy qualification required first')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    baseline = read(ROOT/reg['baseline_outputs'])
    if not baseline['all_accuracy_pass'] or len(baseline['rows']) != reg['expected_complete_states']:
        raise ValueError('Complete corrected own-history baseline required')
    expected_outputs = {(r['identifier'], r['K'], r['step']):r['output_sha256'] for r in baseline['rows']}
    if len(expected_outputs) != reg['expected_complete_states']:raise ValueError('Unique baseline output population required')
    model = CompleteCudaGraphGenerator(reg['pair'], reg['pressure_pair'], reg['build'], reg['control_build'])
    controls = model.controls
    if not read(ROOT/reg['control_qualification'])['all_accuracy_pass']:raise ValueError('Qualified GPU known controls required')
    if read(ROOT/reg['transfer_qualification'])['status']!='passed_actual_native_transfer_bytes_and_ownership':
        raise ValueError('Exact native transfer and output ownership qualification required')
    before = inventory(model)
    if any(value.device.type != 'cuda' for value in model.state_dict().values()):
        raise ValueError('All learned weights must reside on GPU')
    result = dict(status='running', active=None, rows=[], summaries=[],
        inference_device='cuda', display_label='GPU (CUDA)',
        gpu_model=torch.cuda.get_device_name(), precision=dict(pressure='float64', carriers='float32', tf32=False, autocast=False),
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'],
        model_identity=model.asset_identity, complete_output_states=0,
        reference_sources={}, snapshots=[], optimizer_updates=0, cpu_model_forward=False,
        cpu_latency_target_verified=False, full_generator_target_pass=False,
        own_history_without_teacher_resets=True, rollout_steps=reg['rollout_steps'],
        output_dt_s=1/60, physical_forcing_frequency_is_separate_from_output_cadence=True,
        newly_independent_holdout=False, reused_consumed_confirmation_scenes=True,
        expanded_reserved52_references_opened=0, latency_claim=False)
    write(args.output/'results.json', result)
    norm = read(ROOT/reg['normalization']); began = time.monotonic()
    try:
        with torch.inference_mode():
            for selected in reg['trajectories']:
                path = ROOT/selected['receipt']
                if digest(path) != selected['receipt_sha256']:
                    raise ValueError('Frozen reference receipt changed')
                receipt = read(path); scene = receipt['scene']; truth_cache = {}
                specs = {row['step']: row for row in receipt['states']}
                if set(specs) != set(range(reg['rollout_steps']+1)):
                    raise ValueError('Reference for every physical output required')
                def native(step):
                    if step not in truth_cache:
                        spec = specs[step]; source = path.parent/spec['path']
                        if digest(source) != spec['sha256']:
                            raise ValueError('Actual native reference bytes changed')
                        with np.load(source, allow_pickle=False) as archive:
                            truth_cache[step] = {name:archive[name].copy() for name in (*FIELDS, 'rhs')}
                        result['reference_sources'][str(source.relative_to(ROOT))] = spec['sha256']
                    return truth_cache[step]
                model.prepare_geometry(tuple(scene['counts']))
                for k in reg['horizons']:
                    if reg['rollout_steps'] % k:
                        raise ValueError('Physical endpoint must align with full block size')
                    initial = native(0)
                    current = upload_state({name:torch.from_numpy(initial[name]) for name in FIELDS})
                    
                    model.prepare_block(current, scene, 0, k)
                    result['active'] = dict(identifier=selected['identifier'], K=k, step=0)
                    write(args.output/'results.json', result)
                    snapshots = {}
                    for start in range(0, reg['rollout_steps'], k):
                        if time.monotonic()-began > reg['maximum_wall_seconds']:
                            raise TimeoutError('Finite GPU evaluation budget reached')
                        outputs = model.predict(current, scene, start, k)
                        if len(outputs) != k:
                            raise ValueError('Every intermediate physical state required')
                        host_outputs = download_outputs(outputs)
                        for lead, output in enumerate(outputs, 1):
                            step = start+lead
                            if set(output) != set(FIELDS) or any(value.device.type != 'cuda' for value in output.values()):
                                raise ValueError('Complete physical fields must be generated on GPU')
                            emitted = host_outputs[lead-1]
                            for name, value in emitted.items():
                                if value.dtype != (torch.float64 if name == 'pressure' else torch.float32) or not bool(value.isfinite().all()):
                                    raise ValueError('Exact finite GPU output dtypes required')
                            identity = {name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in emitted.items()}
                            bitwise_baseline = identity == expected_outputs[(selected['identifier'], k, step)]
                            metrics = measure(emitted, native(step), scene, norm)
                            metrics['all_accuracy_criteria_pass'] = all(value <= reg['maximum_error'] for value in metrics['relative'].values())
                            result['rows'].append(dict(identifier=selected['identifier'], K=k, step=step, lead=lead,
                                inference_device='cuda', counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                metrics=metrics, bitwise_equal_corrected_baseline=bitwise_baseline, output_sha256={name:hashlib.sha256(value.numpy().tobytes()).hexdigest()
                                                               for name, value in emitted.items()}))
                            result['complete_output_states'] += 1
                            if step in reg['snapshots']:
                                for name, value in emitted.items():snapshots[f'step{step}_{name}'] = value.numpy().copy()
                        current = outputs[-1]
                    file = args.output/(selected['identifier']+f'-k{k}.npz')
                    np.savez_compressed(file, **snapshots)
                    result['snapshots'].append(dict(path=file.name, sha256=digest(file), bytes=file.stat().st_size))
                    result['active'] = None
                    write(args.output/'results.json', result)
                    model.release_graphs()
                    print('GPU expanded-head own-history', selected['identifier'], 'K', k, reg['rollout_steps'], 'complete states scored', flush=True)
            if result['complete_output_states'] != reg['expected_complete_states'] or inventory(model) != before:
                raise ValueError('Full registered population and unchanged weights required')
        grouped = collections.defaultdict(list)
        for row in result['rows']:grouped[(row['K'], row['cells_per_metre'], tuple(row['counts_xyz']))].append(row)
        result['summaries'] = [dict(K=k, cells_per_metre=density, counts_xyz=list(grid), complete_states=len(rows),
            all_accuracy_pass=all(row['metrics']['all_accuracy_criteria_pass'] for row in rows),
            worst_error={name:max(row['metrics']['relative'][name] for row in rows) for name in ('equation','pressure','velocity','tracer')})
            for (k,density,grid), rows in sorted(grouped.items())]
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path) != expected:
                raise ValueError('GPU rollout dependency changed during evaluation: '+path)
        result.update(status='completed_expanded_head_cuda_own_history_regression', active=None,
            all_accuracy_pass=all(row['metrics']['all_accuracy_criteria_pass'] for row in result['rows']),
            every_intermediate_output_scored=True,every_output_bitwise_equal_corrected_baseline=all(r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_different_complete_outputs=sum(not r['bitwise_equal_corrected_baseline'] for r in result['rows']),baseline_outputs=reg['baseline_outputs'],known_control_identity=controls.identity,transfer_adapter='cuda_native_transfer_v1:packed_pinned_upload_and_async_native_download', graph_setup_observations=model.setup_observations,
            elapsed_diagnostic_wall_seconds=time.monotonic()-began,
            diagnostic_wall_time_is_not_inference_benchmark=True, learned_inventory_before_and_after=before)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        model.close(); result['active'] = None; write(args.output/'results.json', result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
