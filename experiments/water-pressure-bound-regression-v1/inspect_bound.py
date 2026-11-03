"""Describe retained bound conservatism and call spikes without new inference."""
import argparse
import json
from pathlib import Path
import statistics

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    root = LAB/read(args.protocol)['root']
    report = read(root/'development/report.json')
    samples, spikes = [], []
    for row in report['records']:
        if row['model'] != 'late-pressure':
            continue
        directory = LAB/row['output']
        callbacks = [json.loads(s) for s in (directory/'native/callbacks.jsonl').read_text().splitlines()]
        by_step = {c['step']: c for c in callbacks}
        peak = max(callbacks, key=lambda c: c['network_passes'])
        spikes.append({'scene': row['scene'], 'counts_xyz': report['configuration']['scenes'][row['scene']]['counts'],
            'maximum_calls': peak['network_passes'], 'first_maximum_call_step': peak['step'],
            'same_step_final_trace': peak['trace'][-1],
            'call_count_histogram': {str(n): sum(c['network_passes'] == n for c in callbacks)
                                     for n in sorted({c['network_passes'] for c in callbacks})}})
        for point in read(directory/'diagnostics.json')['samples']:
            actual = point['mean_free_pressure_relative_l2']
            samples.append({'scene': row['scene'], 'step': point['step'],
                'network_calls': by_step[point['step']]['network_passes'],
                'actual_same_rhs_mean_free_pressure_error': actual,
                'certified_relative_bound': point['pressure_bound_relative_l2'],
                'bound_over_observed_error': point['pressure_bound_relative_l2']/actual if actual > 0 else None,
                'covered': point['pressure_bound_covers_oracle']})
    ratios = [s['bound_over_observed_error'] for s in samples if s['bound_over_observed_error'] is not None]
    result = {'schema': 'rheon-pressure-bound-retained-inspection-v1', 'samples': samples,
        'largest_call_cases': sorted(spikes, key=lambda s: s['maximum_calls'], reverse=True)[:10],
        'states': len(samples), 'all_bounds_cover_observed_error': all(s['covered'] for s in samples),
        'median_bound_over_observed_error': statistics.median(ratios),
        'maximum_bound_over_observed_error': max(ratios),
        'new_simulations': 0, 'new_network_evaluations': 0, 'optimizer_updates': 0,
        'scope': 'post-measurement inspection of saved sparse diagnostics/callback traces; no evidence about unrecorded intermediate pressure fields or counterfactual stopping counts'}
    with (args.reports/'bound-conservatism.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('states', 'median_bound_over_observed_error', 'maximum_bound_over_observed_error')}))


if __name__ == '__main__':
    main()
