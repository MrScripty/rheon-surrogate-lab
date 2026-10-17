"""Compare completed fixed-input transfer diagnostics without selecting a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def identity(row):
    return (tuple(row['counts_xyz']), row['low_mode'], row['kind'],
            row['low_amplitude_pa'], row['high_amplitude_pa'], row['calls'])


def describe(values):
    return dict(median=statistics.median(values), maximum=max(values))


def compare(paths, reference):
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    anchor = reports[reference]
    reference_rows = {identity(row): row for row in anchor['rows']}
    if len(reference_rows) != 252:
        raise ValueError('complete unique reference diagnostics required')
    sources, summaries, flat = {}, {}, []
    for name, report in reports.items():
        rows = report['rows']
        if (report['status'] != 'passed' or report['format'] != 'pressure_spatial_transfer_v1'
                or len(rows) != 252 or {identity(row) for row in rows} != set(reference_rows)
                or report['sources'] != anchor['sources'] or report['method'] != anchor['method']
                or report['network_calls'] != 756 or report['optimizer_updates'] != 0
                or report['simulations_advanced'] != 0 or report['cuda_initialized']
                or report['cpu_threads'] != 1):
            raise ValueError('matched complete CPU diagnostics required: ' + name)
        sources[name] = dict(path=str(paths[name]),
                             sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                             weights_sha256=report['weights_sha256'],
                             witness_sha256=report['source_witness_sha256'],
                             elapsed_s=report['elapsed_s'], peak_rss_bytes=report['peak_rss_bytes'])
        groups = []
        for calls in (1, 4, 8, 12):
            for kind in ('pure-low', 'low-plus-high', 'high-only'):
                for alpha in sorted({row['low_amplitude_pa'] for row in rows if row['kind'] == kind}):
                    chosen = [row for row in rows if row['calls'] == calls
                              and row['kind'] == kind and row['low_amplitude_pa'] == alpha]
                    metrics = {key: describe([row[key] for row in chosen]) for key in
                               ('pressure_relative_l2', 'gradient_relative_l2', 'equation_relative_l2')}
                    metrics['absolute_low_coefficient_error_pa'] = describe(
                        [abs(row['low_coefficient_error_pa']) for row in chosen])
                    for band in ('low_pa2', 'remaining_pa2'):
                        metrics['error_' + band] = describe([row['error_energy'][band] for row in chosen])
                    metrics['pressure_improved_vs_reference'] = sum(
                        row['pressure_relative_l2'] < reference_rows[identity(row)]['pressure_relative_l2']
                        for row in chosen)
                    groups.append(dict(calls=calls, kind=kind, low_amplitude_pa=alpha,
                                       cases=len(chosen), metrics=metrics))
        summaries[name] = groups
        for row in rows:
            other = reference_rows[identity(row)]
            if row['signal_energy'] != other['signal_energy']:
                raise ValueError('manufactured signal differs: ' + name)
            flat.append(dict(model=name, counts_xyz='x'.join(map(str, row['counts_xyz'])),
                             **{k: v for k, v in row.items() if k not in
                                ('counts_xyz', 'signal_energy', 'error_energy')},
                             **{'signal_' + k: v for k, v in row['signal_energy'].items()},
                             **{'error_' + k: v for k, v in row['error_energy'].items()}))
    return dict(format='pressure_transfer_comparison_v1', status='passed', reference=reference,
                sources=sources, groups=summaries, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Matched stationary analytic inputs; groups retain amplitudes and pure/mixed/high-only separation. All per-case tails retained in CSV. Absolute energies and ratios are descriptive, not autonomous drift qualification or causal noise-floor evidence. No model selection or promotion.'), flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, help='LABEL=completed transfer JSON')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.csv').exists():
        raise ValueError('fresh JSON and CSV required')
    paths = {}
    for item in args.model:
        label, separator, value = item.partition('=')
        if not separator or not label or label in paths:
            raise ValueError('unique model labels required')
        paths[label] = Path(value).resolve(strict=True)
    if args.reference not in paths:
        raise ValueError('explicit measured reference required')
    result, rows = compare(paths, args.reference)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with args.output.with_suffix('.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status='passed', models=list(paths), rows=len(rows))))


if __name__ == '__main__':
    main()
