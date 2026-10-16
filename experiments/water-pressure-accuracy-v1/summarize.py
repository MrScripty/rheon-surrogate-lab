"""Retain small pressure/call evidence while raw inputs stay in ignored runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from diagnose import LAB, digest, read, write


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = read(args.config)
    root = LAB/config['root']
    reports = args.reports
    reports.mkdir(exist_ok=True)
    recovery = read(root/'trajectory-recoveries.json')
    replacements = {r['original']: r['accepted'] for r in recovery['records']}
    trajectories = []
    all_steps = []
    failures = []
    for witness in config['witnesses']:
        historical = lines(LAB/witness['historical_frames'])
        for model in config['models']:
            for policy in config['policies']:
                name = f'{witness["name"]}-{model}-{policy}'
                selected = replacements.get(name, name)
                folder = root/selected/'native'
                receipt = read(folder/'receipt.json')
                frames = lines(folder/'frames.jsonl')
                callbacks = {r['step']: r for r in lines(folder/'callbacks.jsonl')}
                assert receipt['status'] == 'completed' and receipt['frames'] == witness['scene']['steps']
                assert receipt['native_process']['reaped'] and receipt['native_process']['returncode'] == 0
                assert not receipt['cuda_initialized'] and not receipt['native_pressure_refinement']
                assert len(callbacks) == receipt['result']['ai_steps'] == witness['scene']['steps']-15
                assert receipt['result']['qualified_ai_steps'] == len(callbacks)
                for frame in frames:
                    assert frame['reference_quality']['qualified']
                    if frame['ai_quality']:
                        callback = callbacks[frame['step']]
                        assert frame['ai_quality']['qualified'] and callback['network_passes'] <= config['maximum_calls']
                        assert callback['trace'][-1]['equation_passed']
                        all_steps.append({'case': name, 'witness': witness['name'], 'model': model,
                            'policy': policy, 'step': frame['step'], 'time_s': frame['time_s'],
                            'network_calls': callback['network_passes'],
                            'correction_seconds': callback['model_correction_seconds'],
                            'pressure_relative_l2': frame['relative_pressure_difference'],
                            'equation_relative_residual': frame['ai_quality']['relative_residual'],
                            'actual_divergence': frame['ai_quality']['actual_divergence'], 'qualified': True})
                observed = [r for r in all_steps if r['case'] == name]
                peak = max(observed, key=lambda r: r['pressure_relative_l2'])
                old_parity = None
                if model == 'targeted' and policy == 'original':
                    assert len(frames) == len(historical)
                    old_parity = max(abs(f['relative_pressure_difference']-old['relative_pressure_difference'])
                                     for f, old in zip(frames, historical))
                    assert old_parity < 1e-12
                trajectories.append({'name': name, 'accepted_attempt': selected,
                    'scene': witness['scene'], 'model': model, 'policy': policy, 'ai_steps': len(observed),
                    'all_physical_checks_passed': True, 'network_calls': sum(r['network_calls'] for r in observed),
                    'mean_calls': statistics.mean(r['network_calls'] for r in observed),
                    'max_calls': max(r['network_calls'] for r in observed),
                    'worst_pressure_relative_l2': peak['pressure_relative_l2'],
                    'first': observed[0], 'peak': peak, 'last': observed[-1],
                    'callback_seconds': sum(r['correction_seconds'] for r in observed),
                    'native_trajectory_elapsed_seconds': receipt['elapsed_s'],
                    'cpu_affinity': receipt['cpu_affinity'], 'cpu_threads': receipt['cpu_threads'],
                    'historical_pressure_metric_max_absolute_difference': old_parity,
                    'native_process': receipt['native_process'],
                    'receipt_path': str((folder/'receipt.json').relative_to(LAB)),
                    'receipt_sha256': digest(folder/'receipt.json'),
                    'callbacks_sha256': digest(folder/'callbacks.jsonl'), 'frames_sha256': digest(folder/'frames.jsonl')})
                if name in replacements:
                    failed = read(root/name/'native/receipt.json')
                    timing = read(root/(name+'-timing.json'))
                    assert failed['status'] == 'failed' and failed['error'] == 'TimeoutError: native trajectory deadline'
                    assert failed['native_process']['reaped'] and failed['native_process']['terminated']
                    assert timing['status'] == 'failed' and timing['child_reaped']
                    failures.append({'name': name, 'native_receipt': failed,
                        'receipt_sha256': digest(root/name/'native/receipt.json'), 'command_timing': timing,
                        'recovery': selected, 'recovery_timing': read(root/(selected+'-timing.json'))})
    aggregate = []
    for model in config['models']:
        for policy in config['policies']:
            rows = [r for r in trajectories if r['model'] == model and r['policy'] == policy]
            steps = sum(r['ai_steps'] for r in rows)
            aggregate.append({'model': model, 'policy': policy, 'simulations': len(rows), 'ai_steps': steps,
                'qualified_simulations': len(rows), 'network_calls': sum(r['network_calls'] for r in rows),
                'mean_calls': sum(r['network_calls'] for r in rows)/steps,
                'max_calls': max(r['max_calls'] for r in rows),
                'worst_pressure_relative_l2': max(r['worst_pressure_relative_l2'] for r in rows),
                'callback_seconds': sum(r['callback_seconds'] for r in rows),
                'elapsed_scope': 'concurrent, some strict-8 attempts recovered separately; not an isolated speed comparison'})
    replays = []
    frontiers = []
    replay_paths = []
    for witness in config['witnesses']:
        for source in config['models']:
            captured = read(root/f'{witness["name"]}-{source}-original/capture.json')
            for sample in captured['samples']:
                replay_paths.extend(root/f'replay-{witness["name"]}-{source}-{sample["step"]:03d}-{model}.json'
                                    for model in config['models'])
    for path in replay_paths:
        row = read(path)
        assert row['oracle_relative_error'] < 1e-9 and not row['cuda_initialized'] and row['optimizer_updates'] == 0
        for model, binding in config['models'].items():
            if binding == row['model']:
                break
        else:
            raise ValueError('unbound replay model')
        assert all(r is not None for r in row['selected'].values())
        assert row['negative_zero_correction']['pressure_relative_l2'] > row['selected']['original']['pressure_relative_l2']
        replays.append({'name': path.stem, 'model_name': model,
            **{k: v for k, v in row.items() if k != 'frontier'},
            'full_trace_sha256': digest(path), 'full_trace_path': str(path.relative_to(LAB)),
            'fixed_8': row['frontier'][8], 'fixed_16': row['frontier'][16], 'fixed_32': row['frontier'][32]})
        for stage in row['frontier']:
            frontiers.append({'name': path.stem, 'model': model, 'sample_sha256': row['sample_sha256'],
                **{k: v for k, v in stage.items() if k not in ('quality', 'top_modes_zyx')},
                'equation_relative_residual': stage['quality']['original']['relative_residual'],
                **{p+'_passes': stage['quality'][p]['equation_passed'] for p in config['policies']}})
    assert len(replays) == 36 and len(trajectories) == 18 and len(failures) == 2
    replay_aggregate = []
    for model in config['models']:
        for policy in config['policies']:
            selected = [r['selected'][policy] for r in replays if r['model_name'] == model]
            replay_aggregate.append({'model': model, 'policy': policy, 'cases': len(selected),
                'mean_calls': statistics.mean(r['calls'] for r in selected),
                'max_calls': max(r['calls'] for r in selected),
                'worst_pressure_relative_l2': max(r['pressure_relative_l2'] for r in selected),
                'worst_mean_free_pressure_relative_l2': max(r['mean_free_pressure_relative_l2'] for r in selected),
                'worst_gradient_relative_l2': max(r['gradient_relative_l2'] for r in selected),
                'median_low_mode_error_energy_fraction': statistics.median(r['low_mode_error_energy_fraction'] for r in selected)})
    for name, rows in [('trajectory-steps', all_steps), ('frontiers', frontiers)]:
        with (reports/(name+'.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    bindings = {name: {**model, 'weights_sha256': digest(LAB/model['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/model['artifacts']/'normalization.json'),
        'restore_receipt_sha256': digest(LAB/model['artifacts']/'receipt.json'),
        'source_witness_sha256': digest(LAB/model['project']/'source-witness.json')}
        for name, model in config['models'].items()}
    fixture_failed = read(root/'fixture/native/receipt.json')
    assert fixture_failed['native_process']['reaped'] and fixture_failed['status'] == 'failed'
    failures.append({'name': 'fixture-relative-path', 'native_receipt': fixture_failed,
        'failed_source_sha256': digest(root/'fixture/failed-source.py'),
        'recovery': read(root/'fixture-1/capture.json'), 'recovery_command_timing': read(root/'fixture-1-timing.json')})
    write(reports/'reconciliations.json', {'scientific_attempt_failures': failures,
        'setup': [{'operation': 'initial configuration setup', 'outcome': 'python unavailable, exit127 before mutation', 'recovery': 'explicit installed python3'},
                  {'operation': 'project snapshot registration', 'outcome': 'HTTP400 invalid:66 exceeds64 references', 'rejected_manifest_sha256': digest(root/'rejected-project-manifest.json'), 'recovery': 'historical index retains64 references'},
                  {'operation': 'reuse immutable project ID', 'outcome': 'HTTP409 conflict; previous snapshot retained', 'rejected_manifest_sha256': digest(root/'conflicting-project-manifest.json'), 'recovery': 'new snapshot in same catalog'},
                  {'operation': 'initial result summarization', 'outcome': 'KeyError:oracle_relative_error; broad replay glob also selected timing files, before report writes', 'failed_source_sha256': digest(root/'summarize-failed-source.py'), 'recovery': 'enumerate exact config/capture-derived replay paths;36 expected replay results accepted'}]})
    write(reports/'replays.json', {'records': replays, 'aggregate': replay_aggregate,
        'unique_input_arrays': len({r['sample_sha256'] for r in replays}),
        'scope': '18 captured states, including duplicate first inputs from shared native warm-up; both checkpoints on each identical input; consumed cases only'})
    write(reports/'trajectories.json', {'records': trajectories, 'aggregate': aggregate,
        'scope': 'independently evolving native paired simulations; full-pressure relative L2 difference, not same-RHS solve error'})
    timings = {p.name: read(p) for p in sorted(root.glob('*-timing.json'))}
    execution = {p.name: read(p) for p in [root/'trajectories-execution.json', root/'replays-execution.json', root/'trajectory-recoveries.json']}
    archived_controller = reports/'executed-controller.py.txt'
    assert all(execution[p]['source_sha256'] == digest(archived_controller)
               for p in ('trajectories-execution.json', 'replays-execution.json'))
    write(reports/'timings.json', {'commands': timings, 'phases': execution,
        'scope': 'each command/phase wall boundary retained; parallel durations do not add to elapsed makespan; recovery contention differs; optimizer updates=0'})
    summary = {'schema': 'rheon-pressure-accuracy-summary-v1', 'bindings': bindings,
        'config_sha256': digest(args.config), 'binary_sha256': digest(LAB/config['native_binary']),
        'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        'source_scope': 'current retained harness; diagnose.py numeric/capture source unchanged throughout scientific execution; initial controller retained separately by exact hash; final controller adds cancellation/core leases, covered by real-child tests',
        'executed_controller_sha256': digest(archived_controller),
        'qualified_simulations': len(trajectories), 'qualified_ai_steps': len(all_steps),
        'maximum_network_calls': max(r['network_calls'] for r in all_steps),
        'replay_evaluations': len(replays), 'unique_inputs': len({r['sample_sha256'] for r in replays}),
        'worst_oracle_defect': max(r['oracle_relative_error'] for r in replays),
        'actual_replay_cpus': sorted({c for r in replays for c in r['cpu_affinity']}),
        'optimizer_updates': 0, 'cuda_initialized': False, 'promoted': False,
        'retained_model': bindings['retained']['weights_sha256'], 'aggregate': aggregate,
        'replay_aggregate': replay_aggregate, 'summarize_seconds': time.perf_counter()-started}
    write(reports/'summary.json', summary)
    print({k: summary[k] for k in ('qualified_simulations', 'qualified_ai_steps', 'replay_evaluations', 'unique_inputs', 'worst_oracle_defect', 'actual_replay_cpus')})


if __name__ == '__main__':
    main()
