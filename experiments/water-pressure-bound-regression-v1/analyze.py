"""Compare all consumed populations without hiding failures in an aggregate."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    with path.open('x') as stream:
        stream.write(buffer.getvalue())


def describe(cases):
    calls = [n for row in cases for n in row['calls']]
    return {'scenes': len(cases), 'qualified_scenes': sum(r['qualified'] for r in cases),
        'complete_scenes': sum(r['complete_window'] for r in cases), 'ai_steps': len(calls),
        'mean_network_evaluations': statistics.mean(calls) if calls else None,
        'maximum_network_evaluations': max(calls, default=None),
        'worst_paired_pressure_relative_l2': max((r['worst_paired_pressure_relative_l2'] for r in cases
            if r['worst_paired_pressure_relative_l2'] is not None), default=None)}


def gates(reference, candidate, expected):
    complete = all(r['scenes'] == r['complete_scenes'] == r['qualified_scenes'] == expected
                   for r in (reference, candidate))
    return {'all_declared_scenes_qualified': complete,
        **{name: bool(complete and candidate[key] <= reference[key]) for name, key in (
            ('worst_pressure_not_regressed', 'worst_paired_pressure_relative_l2'),
            ('mean_calls_not_regressed', 'mean_network_evaluations'),
            ('maximum_calls_not_regressed', 'maximum_network_evaluations'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    protocol, bindings = read(args.protocol), read(args.bindings)
    root = LAB/protocol['root']
    report = read(root/'development/report.json')
    assert report.get('state') in ('completed', 'failed')
    assert digest(args.config) == bindings['regression_config_sha256']
    assert report['config_sha256'] == bindings['regression_config_sha256']
    assert report['binary_sha256'] == bindings['binary_sha256']
    for model in report['models']:
        assert model['weights_sha256'] == bindings['models'][model['name']]['weights_sha256']
        assert model['normalization_sha256'] == bindings['models'][model['name']]['normalization_sha256']
        profile = model.get('stopping_profile', report['configuration']['profile'])
        assert digest(LAB/profile) == bindings['models'][model['name']]['stopping_profile_sha256']
    candidates = {r['scene']: r for r in report['records'] if r['model'] == 'late-pressure'}
    references = {r['scene']: r for r in bindings['reference_rows']}
    for info in bindings['cases']:
        assert digest(LAB/info['source_report']) == info['source_report_sha256']
        path = LAB/info['source_receipt']
        assert digest(path) == info['source_receipt_sha256']
        assert digest(path.parent/'frames.jsonl') == info['source_frames_sha256']
        assert digest(path.parent/'callbacks.jsonl') == info['source_callbacks_sha256']
    cases, trajectories, sparse, failures = [], [], [], []
    bound_checks = 0
    for info in bindings['cases']:
        for model, rows in (('reference-current', references), ('late-pressure', candidates)):
            row = rows.get(info['scene'])
            directory = LAB/row['output'] if row else None
            native = directory/'native' if directory else None
            frames = [json.loads(s) for s in (native/'frames.jsonl').read_text().splitlines()] if native and (native/'frames.jsonl').exists() else []
            frames = [f for f in frames if f.get('ai_quality')]
            receipt = read(native/'receipt.json') if native and (native/'receipt.json').exists() else None
            callbacks = [json.loads(s) for s in (native/'callbacks.jsonl').read_text().splitlines()] if native and (native/'callbacks.jsonl').exists() else []
            by_step = {c['step']: c for c in callbacks}
            expected = report['configuration']['scenes'][info['scene']]
            complete = bool(receipt and receipt['result'] and not receipt['result']['hard_stop']
                and frames and frames[-1]['step'] == expected['steps']
                and len(frames) == len(callbacks) == expected['steps']-expected['warmup_steps'])
            qualified = bool(complete and row['qualified'] and receipt['native_process']['reaped'])
            calls = [c['network_passes'] for c in callbacks]
            peak = max(frames, key=lambda f: f['relative_pressure_difference']) if frames else None
            result = {'scene': info['scene'], 'population': info['population'], 'group': info['group'], 'model': model,
                'qualified': qualified, 'complete_window': complete, 'calls': calls,
                'worst_paired_pressure_relative_l2': peak['relative_pressure_difference'] if peak else None,
                'peak_step': peak['step'] if peak else None,
                'source_receipt_sha256': digest(native/'receipt.json') if receipt else None,
                'observed_native_reaped': receipt['native_process']['reaped'] if receipt else None}
            cases.append(result)
            if not qualified:
                failures.append({k: result[k] for k in ('scene', 'population', 'model', 'complete_window', 'observed_native_reaped')})
            for frame in frames:
                callback = by_step.get(frame['step'])
                trace = callback['trace'][-1] if callback else {}
                trajectories.append({'group': info['group'], 'population': info['population'], 'scene': info['scene'],
                    'model': model, 'step': frame['step'], 'time_s': frame['time_s'],
                    'paired_pressure_relative_l2': frame['relative_pressure_difference'],
                    'ai_relative_residual': frame['ai_quality']['relative_residual'],
                    'ai_actual_divergence': frame['ai_quality']['actual_divergence'],
                    'network_evaluations': callback['network_passes'] if callback else None,
                    'pressure_bound_relative_error': trace.get('mean_free_pressure_relative_error_bound'),
                    'declared_stopping_passed': trace.get('equation_passed'),
                    'physical_checks_passed': frame['ai_quality']['qualified']})
            diagnostic = directory/'diagnostics.json' if directory else None
            if diagnostic and diagnostic.exists():
                observed = read(diagnostic)
                if model == 'late-pressure':
                    assert observed['weights_sha256'] == bindings['models'][model]['weights_sha256']
                    assert observed['config_sha256'] == bindings['regression_config_sha256']
                for state in observed['samples']:
                    if model == 'late-pressure':
                        assert state['pressure_bound_covers_oracle']
                        bound_checks += 1
                    sparse.append({'group': info['group'], 'population': info['population'], 'scene': info['scene'],
                        'model': model, **{k: v for k, v in state.items() if k != 'top_modes_zyx'}})
            elif model == 'late-pressure':
                failures.append({'scene': info['scene'], 'model': model, 'failure': 'independent sparse diagnostics unavailable'})
    groups, populations = {}, {}
    for grouping, names, destination in (
            ('group', protocol['replacement_groups'], groups),
            ('population', list(dict.fromkeys(c['population'] for c in bindings['cases'])), populations)):
        for name in names:
            selected = [c for c in cases if c[grouping] == name]
            models = {m: describe([r for r in selected if r['model'] == m]) for m in ('reference-current', 'late-pressure')}
            expected = sum(c[grouping] == name for c in bindings['cases'])
            checks = gates(models['reference-current'], models['late-pressure'], expected)
            destination[name] = {'models': models, 'gates': checks, 'passed': all(checks.values())}
    eligible = not failures and all(g['passed'] for g in groups.values())
    result = {'schema': 'rheon-pressure-bound-regression-result-v1', 'state': 'completed',
        'replacement_groups': groups, 'subgroups': populations, 'cases': cases,
        'consumed_acceptance_passed': eligible, 'candidate_promoted': False, 'deployment_changed': False,
        'fresh_validation_authorized_by_numeric_gates': eligible, 'optimizer_updates': 0,
        'independent_pressure_bound_checks': bound_checks, 'failures': failures,
        'elapsed_seconds': report['elapsed_seconds'], 'source_report_sha256': digest(root/'development/report.json'),
        'binding_report_sha256': digest(args.bindings), 'source_sha256': digest(__file__),
        'scope': 'three consumed replacement groups evaluated independently; archived numerical references, current candidate; no concurrent latency claim',
        'baseline_sparse_diagnostics': 'unavailable in archived references; no zero-error substitution'}
    write(args.reports/'regression-summary.json', result)
    write_csv(args.reports/'trajectory-metrics.csv', trajectories)
    write_csv(args.reports/'sparse-metrics.csv', sparse)
    write_csv(args.reports/'case-comparison.csv', [{**{k: v for k, v in c.items() if k != 'calls'},
        'mean_calls': statistics.mean(c['calls']) if c['calls'] else None,
        'maximum_calls': max(c['calls'], default=None)} for c in cases])
    print(json.dumps({'groups': groups, 'eligible_for_fresh_validation': eligible,
                      'bound_checks': bound_checks, 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
