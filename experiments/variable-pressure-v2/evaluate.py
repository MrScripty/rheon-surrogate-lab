"""Native per-grid pressure diagnostics with unchanged accuracy and fallback gates."""
import math
from pathlib import Path
import sys
import time

import numpy as np

from policy import LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import build_native
import data

METHODS = ('learned_jacobi', 'temporal_full_budget', 'temporal_jacobi', 'zero_jacobi')


def historical_oracles():
    paths = list(sys.path)
    try:
        sys.path.insert(0, str(LAB/'experiments/global-pressure-v1'))
        oracle = load_module('variable_eval_independent_oracle', LAB/'experiments/global-pressure-v1/verify_capture.py')
        diagnostics = load_module('variable_eval_common_quality', LAB/'experiments/global-pressure-v1/groupnorm-native-dev-v1/admission.py')
        return oracle, diagnostics
    finally:
        sys.path[:] = paths


def check_response(request, result, oracle, diagnostics):
    require(result['schema'] == 'rheon-variable-pressure-teacher-diagnostic-v1' and result['state_advanced'] is False,
            'diagnostic_scope', 'teacher-state snapshot only')
    for name in ('reference', 'raw_learned', 'raw_temporal', 'raw_zero'):
        diagnostics.validate_quality(result[name], request)
    require(result['reference']['qualified'], 'reference', 'native reference gates')
    counts = request['counts_xyz']; cells = math.prod(counts)
    action = oracle.full_action(request['candidate_pressure'], counts, request['spacing_m'], request['density_kg_m3'])
    residual = [b-a for b,a in zip(request['original_rhs'], action)]
    independent_l2 = math.sqrt(math.fsum(v*v for v in residual))
    measured = result['raw_learned'].get('residual_l2')
    require(measured is not None and abs(independent_l2-measured) <= 1e-12*max(independent_l2, 1e-9),
            'prediction_operator', 'independent full operator on this exact cubic grid')
    for name in METHODS:
        value = result[name]
        require(value['qualified'] is True, 'terminal_pressure', name)
        if name in ('learned_jacobi', 'temporal_jacobi'):
            attempts = [(value['warm'], 32, request['candidate_pressure'] if name == 'learned_jacobi' else request['previous_pressure']),
                        (value['reset'], 3000, request['previous_pressure'])]
        else:
            attempts = [(value, 3000, None if name == 'zero_jacobi' else request['previous_pressure'])]
        for attempt, limit, guess in attempts:
            if attempt is None:
                continue
            require(attempt['iteration_limit'] == limit and type(attempt['iterations']) is int and 0 <= attempt['iterations'] <= limit,
                    'iteration_count', 'known completed native work')
            raw_guess = bytes(cells*8) if guess is None else np.asarray(guess, dtype='<f8').tobytes()
            require(attempt['initial_guess_sha256'] == sha(raw_guess), 'native_start', 'actual per-grid guess bytes')
            diagnostics.validate_quality(attempt['quality'], request)
            require(attempt['qualified'] is (attempt['solver_error'] is None and attempt['quality']['qualified']), 'solver_outcome', name)
    return independent_l2


def summary(records):
    result = {}
    for name in METHODS:
        attempts, wasted = [], []
        for record in records:
            value = record[name]
            if name in ('learned_jacobi', 'temporal_jacobi'):
                attempts += [a for a in (value['warm'], value['reset']) if a is not None]
                if value['reset'] is not None and value['warm'] is not None:
                    wasted.append(value['warm'])
            else:
                attempts.append(value)
        result[name] = {'qualified': sum(r[name]['qualified'] for r in records),
                        'completed_iterations': sum(a['iterations'] for a in attempts),
                        'wasted_warm_iterations': sum(a['iterations'] for a in wasted),
                        'fallbacks': sum(r[name].get('reset_fallback', False) for r in records)}
    return result


def run(packet, binary, training, restored, output, observer=None):
    start = time.monotonic()
    config, parents = data.load(read(packet/'dataset.npz', 32*MIB), decode(read(packet/'splits.json', 4*MIB)))
    trained = decode(read(training/'receipt.json')); cold = decode(read(restored/'receipt.json'))
    require(trained['schema'] == cold['schema'] == 'rheon-variable-pressure-training-v2' and trained['mode'] == 'train'
            and cold['mode'] == 'restore' and cold['optimizer_updates'] == 0 and trained['optimizer_updates'] == config['training']['updates']
            and trained['closure'] == cold['closure'] and cold['originating_training_run'] is not None, 'trained_restore', 'actual managed final state and zero-update restore')
    for name, declaration in cold['outputs'].items():
        raw = read(restored/name, 4*MIB)
        require(sha(raw) == declaration['sha256'] and len(raw) == declaration['bytes'], 'artifact_hash', name)
        if name != 'summary.json':
            require(raw == read(training/name, 4*MIB), 'restore_parity', name)
    require(cold['cases'] == trained['cases'], 'case_mapping', 'same per-grid population')
    arrays = data.helpers().read_npz(read(restored/'predictions.npz', 2*MIB), cold['prediction_inventory'], 2*MIB)
    native = build_native.admit(binary)
    require(native == decode(read(packet/'splits.json', 4*MIB))['native_build_receipt'], 'native_residual', 'same packet and native operator')
    oracle, diagnostics = historical_oracles(); output = fresh(output); records = []
    try:
        for index, case in enumerate(cold['cases']):
            parent = parents[case['parent']]; root = Path(parent['capture_root'])
            raw_manifest = read(root/'manifest.json', 4*MIB)
            require(sha(raw_manifest) == parent['capture_manifest_sha256'], 'capture_drift', 'qualified parent origin')
            trajectory = next(t for t in decode(raw_manifest)['trajectories'] if t['trajectory_id'] == case['trajectory_id'])
            frame = trajectory['frames'][case['frame']]; counts = parent['counts_xyz']
            velocity = [oracle.load_array(root, frame['arrays']['pre_'+axis], oracle.shape(d, counts), '<f4', 'm/s') for d, axis in enumerate('uvw')]
            fields = {name: parent['arrays'][name][case['frame']].reshape(-1).tolist() for name in data.FIELDS}
            candidate = arrays[f'case_{index:03d}'].reshape(-1).tolist()
            request = {'schema': 'rheon-variable-pressure-teacher-case-v1', 'counts_xyz': counts, 'spacing_m': [parent['cell_size_m']]*3,
                       'density_kg_m3': 1.25, 'dt_s': parent['entry']['dt_s'], 'original_rhs': fields['rhs'],
                       'reference_pressure': fields['reference'], 'previous_pressure': fields['base'], 'pre_velocity': velocity,
                       'candidate_pressure': candidate, 'candidate_error': None}
            response, process = build_native.call(binary, '--evaluate', request)
            independent = check_response(request, response, oracle, diagnostics)
            record = {**case, **response, 'process': process, 'independent_candidate_residual_l2': independent}
            with (output/'cases.jsonl').open('ab') as stream:
                stream.write(canonical(record)+b'\n'); stream.flush()
            records.append(record)
            if observer:
                entry = {**parent['entry'], 'counts_xyz': counts, 'extents_m': trajectory['geometry']['extents_m']}
                label = 'Learned pressure (Pa); raw physics gates passed' if response['raw_learned']['qualified'] else 'Learned pressure proposal (Pa); raw physics gates failed; reference tracer'
                preview = observer.frame(root, entry, case['frame'], pressure=np.asarray(candidate), pressure_label=label)
                observer.publish('Seven-channel native diagnostics', f'{counts}; cubic h={parent["cell_size_m"]} m; raw pressure {"passed" if response["raw_learned"]["qualified"] else "failed"}', index+1, len(cold['cases']), preview=preview)
        require(build_native.admit(binary) == native, 'native_drift', 'unchanged qualified closure')
        result = {'status': 'complete_diagnostic', 'cases': len(records), 'raw_qualified': sum(r['raw_learned']['qualified'] for r in records),
                  'methods': summary(records), 'shapes_xyz': [p['counts_xyz'] for p in parents], 'cubic_cells': True,
                  'workflow_elapsed_s': time.monotonic()-start, 'scope': 'teacher-state development snapshots; no autonomous rollout or acceleration claim'}
        write(output/'evaluation.json', result)
        if observer:
            observer.publish('Variable-grid diagnostics complete', f'{len(records)} cases; raw learned pressure qualified {result["raw_qualified"]}; see retained native results', len(records), len(records), state='completed')
        return result
    except BaseException as error:
        write(output/'failure.json', {'status': 'failed', 'completed_cases': len(records), 'error': str(error)})
        if observer:
            observer.publish('Variable-grid diagnostics failed', 'Retained evidence is diagnostic; no completed evaluation claim', state='failed')
        raise
