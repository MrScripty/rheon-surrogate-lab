"""Derive the recurrence comparison from sealed trial and native receipts."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(name, trial, fixed):
    adaptive = trial['adaptive_development']
    return {
        'model': name,
        'training_evaluations': trial['configuration']['correction']['steps'],
        'seed': trial['configuration']['training']['seed'],
        'sampling_seed': trial['configuration']['training']['sampling_seed'],
        'updates': trial['cuda_optimizer_updates'],
        'updates_per_s': trial['optimizer_updates_per_s'],
        'optimizer_seconds': trial['optimizer_interval_seconds'],
        'forward_cell_evaluations': trial['training_exposure']['forward_cell_evaluation_proxy'],
        'development_cases': adaptive['native']['cases'],
        'adaptive_passed': adaptive['native']['float32_qualified'],
        'adaptive_mean_evaluations': adaptive['network_evaluations']['mean'],
        'adaptive_maximum_evaluations': adaptive['network_evaluations']['maximum'],
        'fixed_eight_passed': fixed['native']['float32_qualified'],
        'fixed_eight_median_residual': fixed['native']['relative_residual']['median'],
        'fixed_eight_worst_residual': fixed['native']['relative_residual']['maximum'],
        'weights_sha256': trial['weights_sha256'],
        'normalization_sha256': trial['normalization_sha256'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'records', 'reference-record', 'reference-fixed', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-fixed-name', required=True)
    args = parser.parse_args()
    reference = read(args.reference_record)
    fixed = read(args.reference_fixed)['results'][args.reference_fixed_name]
    rows = [comparison('cohort-two-reference', reference, fixed)]
    sources = {str(p): digest(p) for p in (args.reference_record, args.reference_fixed)}
    managed_stages = []
    for path in sorted(args.records.glob('trial-*.json')):
        trial = read(path)
        fixed_path = args.root/trial['trial']/'fixed-eight/report.json'
        sources.update({str(p): digest(p) for p in (path, fixed_path)})
        rows.append(comparison(trial['trial'], trial, read(fixed_path)))
        managed_stages.extend(trial['managed_stages'])
    result = {
        'schema': 'rheon-recurrence-comparison-v1',
        'models': rows,
        'new_cuda_updates': sum(row['updates'] for row in rows[1:]),
        'managed_stages': managed_stages,
        'sources': sources,
        'scope': 'development comparison; unchanged equation gates; adaptive and fixed-eight policies are distinct; update counts do not imply equal compute',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    args.output.with_suffix('.csv').write_text(stream.getvalue())
    print(json.dumps({'models': len(rows), 'new_cuda_updates': result['new_cuda_updates']}))


if __name__ == '__main__':
    main()
