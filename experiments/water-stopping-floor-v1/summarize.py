"""Retain trajectory, sparse same-RHS, and measured timing evidence separately."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    with path.open('x') as target:
        target.write(stream.getvalue())


def aggregate(report):
    summaries, trajectories, sparse = {}, [], []
    for model in report['models']:
        rows = [r for r in report['records'] if r['model'] == model['name']]
        pressure, calls, divergence, samples, times = [], [], [], [], []
        for row in rows:
            output = LAB/row['output']
            receipt = read(output/'native/receipt.json')
            if not receipt['native_process']['reaped'] or receipt['result'] is None:
                raise ValueError('completed/reaped native receipt required before scientific summary')
            frames = [json.loads(line) for line in (output/'native/frames.jsonl').read_text().splitlines()]
            frames = [f for f in frames if f.get('ai_quality')]
            if len(frames) != receipt['result']['ai_steps']:
                raise ValueError('complete frame population required')
            for f in frames:
                pressure.append(f['relative_pressure_difference'])
                divergence.append(f['ai_quality']['actual_divergence'])
                trajectories.append({'model': model['name'], 'scene': row['scene'], 'repetition': row['repetition'],
                    'step': f['step'], 'time_s': f['time_s'], 'paired_pressure_relative_l2': f['relative_pressure_difference'],
                    'ai_relative_residual': f['ai_quality']['relative_residual'],
                    'ai_actual_divergence': f['ai_quality']['actual_divergence'],
                    'qualified': f['ai_quality']['qualified']})
            calls.extend(row['network_passes'])
            times.extend(row['ai_seconds'])
            diagnostics = read(output/'diagnostics.json')
            for sample in diagnostics['samples']:
                band = sample['low_mode_error_energy_fraction']
                point = {'model': model['name'], 'scene': row['scene'], 'repetition': row['repetition'], **sample}
                point.pop('top_modes_zyx')
                point.update(low_band_relative_l2=sample['mean_free_pressure_relative_l2']*math.sqrt(band),
                             other_band_relative_l2=sample['mean_free_pressure_relative_l2']*math.sqrt(max(0., 1-band)))
                sparse.append(point)
                samples.append(point)
        summaries[model['name']] = {'episodes': len(rows), 'qualified_episodes': sum(r['qualified'] for r in rows),
            'ai_steps': len(pressure), 'mean_network_evaluations': statistics.mean(calls),
            'minimum_network_evaluations': min(calls), 'maximum_network_evaluations': max(calls),
            'worst_paired_pressure_relative_l2': max(pressure),
            'median_paired_pressure_relative_l2': statistics.median(pressure),
            'worst_actual_divergence': max(divergence),
            'steady_ai_seconds_median': statistics.median(times),
            'per_episode_ai_seconds_median': [statistics.median(r['ai_seconds']) for r in rows],
            'sampled_same_rhs_states': len(samples),
            'sparse_worst_pressure_relative_l2': max((r['pressure_relative_l2'] for r in samples), default=None),
            'sparse_worst_gradient_relative_l2': max((r['gradient_relative_l2'] for r in samples), default=None),
            'sparse_worst_low_band_relative_l2': max((r['low_band_relative_l2'] for r in samples), default=None),
            'sparse_worst_other_band_relative_l2': max((r['other_band_relative_l2'] for r in samples), default=None)}
    return summaries, trajectories, sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', default='late-minimum8')
    args = parser.parse_args()
    assert not args.output.exists(), 'retain earlier analysis attempts'
    report = read(args.report)
    assert report['state'] == 'completed'
    models, trajectories, sparse = aggregate(report)
    reference = models['reference-current']
    candidate = models[args.candidate]
    gates = {'all_episodes_qualified': all(m['qualified_episodes'] == m['episodes'] for m in models.values()),
        'worst_pressure_not_regressed': candidate['worst_paired_pressure_relative_l2'] <= reference['worst_paired_pressure_relative_l2'],
        'mean_calls_not_regressed': candidate['mean_network_evaluations'] <= reference['mean_network_evaluations'],
        'maximum_calls_not_regressed': candidate['maximum_network_evaluations'] <= reference['maximum_network_evaluations']}
    result = {'schema': 'rheon-stopping-floor-comparison-v1', 'phase': report['phase'],
        'source_report_sha256': digest(args.report), 'source_sha256': digest(__file__),
        'models': models, 'candidate': args.candidate,
        'consumed_acceptance_gates': gates, 'consumed_acceptance_passed': all(gates.values()),
        'elapsed_seconds': report['elapsed_seconds'], 'timing_scope': report['timing_scope'],
        'network_call_scope': 'all AI steps, including first AI step; steady timing excludes first AI step',
        'sparse_band_scope': 'mean-free spatial pressure error; first3 cosine modes/axis excluding constant gauge; components normalized against entire exact mean-free pressure, not band-specific denominators',
        'pressure_scope': 'paired_pressure compares independently advanced native/AI tracks; sparse_same_rhs errors solve each actual supplied state independently',
        'optimizer_updates': 0, 'candidate_promoted': False,
        'native_children': len(report['records']),
        'all_native_children_reaped': all(r.get('native_process', {}).get('reaped') for r in report['records'])}
    write_csv(args.output.with_suffix('.trajectories.csv'), trajectories)
    if sparse:
        write_csv(args.output.with_suffix('.sparse.csv'), sparse)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
