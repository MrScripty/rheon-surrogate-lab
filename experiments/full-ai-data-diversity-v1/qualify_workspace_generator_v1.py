"""Accuracy-only full-generator integration of reusable native CUDA transfers.

All returned field bytes must match the source-bound, native-qualified current
backend. No inference timing is measured. Legacy continuation uses only its own
returned state after the initial native state; block windows have explicit starts.
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
from support import ROOT, read, write, digest
from debug.cube128_exact_transform_candidate_v2 import Cube128ExactTransformCandidate
from cuda_native_transfer_v1 import FIELDS, upload_state
from cuda_native_workspace_v1 import NativeStateWorkspace
from diagnose_causal_composition import inventory


def hashes(row):
    return {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in row.items()}


def run(args):
    if args.output.exists():
        raise ValueError('Fresh workspace/model integration evidence required')
    reg = read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda' or reg['suite'] not in ('blocks', 'legacy'):
        raise ValueError('Registered CUDA-only accuracy suite required')
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen workspace/model dependency changed: ' + path)
    prerequisite = read(ROOT / reg['workspace_qualification'])
    if prerequisite['status'] != 'passed_native_workspace_bytes_ownership_stream_and_failure_recovery':
        raise ValueError('Complete workspace byte/ownership qualification required')
    baseline = read(ROOT / reg['baseline_outputs'])
    if not baseline['all_accuracy_pass'] or len(baseline['rows']) != reg['expected_complete_states']:
        raise ValueError('Complete native-qualified current-backend outputs required')
    if not all(r['metrics']['all_accuracy_criteria_pass'] for r in baseline['rows']):
        raise ValueError('Every baseline output must pass all original native accuracy gates')
    def key(row):
        return ((row['identifier'], row['start'], row['K'], row['lead']) if reg['suite'] == 'blocks'
                else (row['identifier'], row['K'], row['step']))
    expected = {key(row): row['output_sha256'] for row in baseline['rows']}
    if len(expected) != reg['expected_complete_states']:
        raise ValueError('Unique complete baseline output population required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    model = Cube128ExactTransformCandidate(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    if any(v.device.type != 'cuda' for v in model.state_dict().values()):
        raise ValueError('Every learned tensor must reside on CUDA')
    result = dict(status='running', active=None, suite=reg['suite'], rows=[], setups=[],
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  cpu_model_forward=False, optimizer_updates=0, latency_claim=False,
                  own_history_without_teacher_resets=reg['suite'] == 'legacy',
                  every_intermediate_output_required=True, expanded_reserved52_references_opened=0,
                  newly_independent_holdout=False, baseline_outputs=reg['baseline_outputs'],
                  baseline_sha256=digest(ROOT / reg['baseline_outputs']),
                  protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    started = time.monotonic()
    def save():
        write(args.output / 'results.json', result)
    def budget():
        if time.monotonic() - started > reg['maximum_wall_seconds']:
            raise TimeoutError('Workspace integration deadline reached')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']
                or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
            raise ValueError('Workspace integration memory ceiling exceeded')
    def load(path, checksum):
        if digest(path) != checksum:
            raise ValueError('Native current-state archive changed')
        with np.load(path, allow_pickle=False) as archive:
            return {name: torch.from_numpy(archive[name].copy()) for name in FIELDS}
    def verify(output, row):
        if set(output) != set(FIELDS) or any(v.device.type != 'cpu' or v.is_pinned() for v in output.values()):
            raise ValueError('Complete independently owned ordinary CPU output required')
        actual = hashes(output)
        if actual != expected[key(row)]:
            raise ValueError('Actual workspace model output differs from native-qualified baseline')
        result['rows'].append(dict(row, output_sha256=actual, native_qualified_baseline_bytes_exact=True))
        return actual
    def retain_check(retained):
        if any(hashes(row) != identity for row, identity in retained):
            raise ValueError('A later actual model call overwrote retained physical fields')
    save()
    try:
        with torch.inference_mode():
            if reg['suite'] == 'blocks':
                bank = read(ROOT / reg['bank'])
                population = {r['identifier']: r for r in bank['trajectories']}
                windows = {}
                for source in reg['window_sources']:
                    for row in read(ROOT / source['controller'])['completed']:
                        if row['identifier'] in windows:
                            raise ValueError('Unique complete development window population required')
                        windows[row['identifier']] = row
                if len(population) != 24 or set(population) != set(windows):
                    raise ValueError('All24development scenes required')
                for identifier, selected in population.items():
                    if digest(ROOT / selected['receipt']) != selected['receipt_sha256']:
                        raise ValueError('Development receipt changed')
                    scene = read(ROOT / selected['receipt'])['scene']
                    n = selected['reference_substeps']
                    model.prepare_geometry(tuple(scene['counts']))
                    workspace = None
                    retained = []
                    for window in windows[identifier]['windows']:
                        start = window['start']
                        current = load(ROOT / window['current_state_path'], window['current_state_sha256'])
                        if workspace is None:
                            workspace = NativeStateWorkspace(current)
                            result['setups'].append(dict(identifier=identifier, **workspace.setup))
                        resident = upload_state(current)
                        for k in reg['horizons']:
                            budget()
                            result['active'] = dict(identifier=identifier, start=start, K=k)
                            model.prepare_block(resident, scene, start, k, n)
                            outputs = workspace.run(current, lambda fields: model.predict(fields, scene, start, k, n))
                            if len(outputs) != k:
                                raise ValueError('Every complete K intermediate required')
                            for lead, output in enumerate(outputs, 1):
                                identity = verify(output, dict(identifier=identifier, start=start, K=k, lead=lead,
                                    step=start + lead, counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], internal_substeps=n))
                                retained.append((output, identity))
                            retain_check(retained)
                    model.release_graphs()
                    save()
                    del workspace, retained
                    print('Workspace full-generator blocks', identifier, 'allK/four windows exact', flush=True)
            else:
                for selected in reg['trajectories']:
                    path = ROOT / selected['receipt']
                    if digest(path) != selected['receipt_sha256']:
                        raise ValueError('Legacy receipt changed')
                    receipt = read(path)
                    scene = receipt['scene']
                    initial_spec = next(s for s in receipt['states'] if s['step'] == 0)
                    initial = load(path.parent / initial_spec['path'], initial_spec['sha256'])
                    model.prepare_geometry(tuple(scene['counts']))
                    workspace = NativeStateWorkspace(initial)
                    result['setups'].append(dict(identifier=selected['identifier'], **workspace.setup))
                    retained = []
                    for k in reg['horizons']:
                        if reg['rollout_steps'] % k:
                            raise ValueError('Complete physical endpoint must align with K')
                        current = initial
                        resident = upload_state(initial)
                        model.prepare_block(resident, scene, 0, k)
                        for start in range(0, reg['rollout_steps'], k):
                            budget()
                            result['active'] = dict(identifier=selected['identifier'], K=k, step=start)
                            outputs = workspace.run(current, lambda fields: model.predict(fields, scene, start, k))
                            if len(outputs) != k:
                                raise ValueError('Every own-history physical output required')
                            for lead, output in enumerate(outputs, 1):
                                step = start + lead
                                identity = verify(output, dict(identifier=selected['identifier'], K=k, step=step, lead=lead,
                                    counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], internal_substeps=1))
                                if step in reg['retained_steps']:
                                    retained.append((output, identity))
                            current = outputs[-1]
                        retain_check(retained)
                        model.release_graphs()
                        save()
                        print('Workspace full-generator history', selected['identifier'], 'K', k, reg['rollout_steps'], 'states exact', flush=True)
                    del workspace, retained
            if len(result['rows']) != reg['expected_complete_states'] or inventory(model) != before:
                raise ValueError('Complete population and unchanged learned tensors required')
            if {key(row) for row in result['rows']} != set(expected):
                raise ValueError('Every registered physical output must be verified exactly once')
        result.update(status='passed_full_generator_workspace_integration', active=None,
                      every_output_bitwise_equal_native_qualified_baseline=True,
                      native_accuracy_verified_by_exact_baseline_bytes=True,
                      retained_actual_model_outputs_unchanged=True,
                      learned_inventory_before_and_after=before,
                      elapsed_diagnostic_seconds=time.monotonic() - started)
        for path, checksum in reg['source_sha256'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Workspace integration dependency changed: ' + path)
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
