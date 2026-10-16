"""Summarize per-correction contraction from retained mixed-mode probe traces."""
import argparse
import hashlib
from pathlib import Path
import statistics

from execution import read, write


def summarize(path):
    report = read(path)
    groups = []
    for shape, population in sorted({(tuple(c['counts_xyz']), c['population'])
                                     for c in report['cases']}):
        cases = [c for c in report['cases']
                 if tuple(c['counts_xyz']) == shape and c['population'] == population]
        ratios = [b['relative_residual']/a['relative_residual'] for c in cases
                  for a,b in zip(c['trace'],c['trace'][1:]) if a['relative_residual'] > 0]
        groups.append({'counts_xyz': shape, 'mixture_population': population,
            'signed_cases': len(cases), 'equation_passed': sum(c['equation_passed'] for c in cases),
            'median_first_relative_residual': statistics.median(c['first_relative_residual'] for c in cases),
            'worst_first_relative_residual': max(c['first_relative_residual'] for c in cases),
            'first_correction_increases_residual': sum(c['first_relative_residual'] > 1 for c in cases),
            'median_successive_residual_ratio': statistics.median(ratios),
            'maximum_successive_residual_ratio': max(ratios),
            'nondecreasing_successive_corrections': sum(r >= 1 for r in ratios),
            'successive_corrections': len(ratios),
            'mean_network_evaluations': statistics.mean(c['network_evaluations'] for c in cases)})
    return {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'weights_sha256': report['weights_sha256'], 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write(args.output, {'schema': 'rheon-probe-correction-feedback-v1',
        'models': [summarize(path) for path in args.reports],
        'scope': 'diagnostic ratios on retained signed low/broad mixed-mode traces, including final numerical floors; not pure-mode attribution, global contraction proof or inference-policy change'})


if __name__ == '__main__':
    main()
