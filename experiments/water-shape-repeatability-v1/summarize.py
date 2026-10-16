"""Compare sealed trials and retain distinct optimizer, managed and CPU costs."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import sys

from execution import LAB, read, utc, write

sys.path.insert(0, str(LAB/'experiments/water-recurrence-generalization-v1'))
# Load by file identity: this module has the same basename as the earlier owner.
from importlib.util import module_from_spec, spec_from_file_location
spec = spec_from_file_location('recurrence_comparison',
                               LAB/'experiments/water-recurrence-generalization-v1/summarize.py')
previous = module_from_spec(spec)
spec.loader.exec_module(previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--reference-record', type=Path, required=True)
    parser.add_argument('--reference-index', type=Path, required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    reference = read(args.reference_record)
    reference_root = (LAB/sequence['reference']['project']).parent
    fixed = read(reference_root/'fixed-eight/report.json')
    if reference['weights_sha256'] != digest(LAB/sequence['reference']['artifacts']/'weights.npz'):
        raise ValueError('reference checkpoint differs from retained trial')
    baseline = read(args.reference_index)
    reference_qualified = all(
        baseline['populations'][name]['models'][reference['trial']]['qualified_scenes']
        == baseline['populations'][name]['models'][reference['trial']]['scenes']
        for name in ('fresh_development','long_development','prior_grids'))
    rows = [previous.comparison(sequence['reference']['name'], reference, fixed)]
    rows[0].update(managed_seconds=None, cpu_qualification_seconds=None,
                   all_development_qualified=reference_qualified)
    stages = []
    intervals = []
    sources = {str(args.reference_record): digest(args.reference_record),
               str(args.reference_index): digest(args.reference_index),
               str(args.sequence): digest(args.sequence)}
    for trial in sequence['trials']:
        directory = root/trial['name']
        path = args.reports/('trial-'+trial['name']+'.json')
        record = read(path)
        fixed_path = directory/'fixed-eight/report.json'
        row = previous.comparison(trial['name'], record, read(fixed_path))
        row.update(managed_seconds=read(directory/'managed-timing.json')['elapsed_seconds'],
                   cpu_qualification_seconds=read(directory/'qualification-timing.json')['elapsed_seconds'],
                   all_development_qualified=record['cpu_pipeline']['all_development_qualified'])
        rows.append(row)
        sources.update({str(p): digest(p) for p in (path, fixed_path)})
        history = read(directory/'managed-full/stage-00/artifacts/history.json')
        intervals.append({'trial': trial['name'],
                          'started_unix': history['training_started_unix'],
                          'finished_unix': history['training_finished_unix']})
        for index, stage in enumerate(record['managed_stages']):
            run = read(directory/f'managed-full/stage-{index:02d}/run.json')
            def difference(end, start):
                return (datetime.fromisoformat(run[end])-datetime.fromisoformat(run[start])).total_seconds()
            stages.append({'trial': trial['name'], 'stage': stage['name'],
                'run_id': run['id'], 'state': run['state'], 'custody_phase': run['custody_phase'],
                'created_at': run['created_at'], 'started_at': run['started_at'],
                'finished_at': run['finished_at'], 'queue_seconds': difference('started_at','created_at'),
                'run_seconds': difference('finished_at','started_at')})
    intervals.sort(key=lambda item: item['started_unix'])
    if any(a['finished_unix'] > b['started_unix'] for a,b in zip(intervals,intervals[1:])):
        raise ValueError('one optimizer at a time required')
    if len({stage['run_id'] for stage in stages}) != len(stages):
        raise ValueError('each real managed Run must be counted once')
    execution = read(root/'execution.json')
    if execution['status'] != 'completed':
        raise ValueError('inspect incomplete sequence; no completed comparison claim')
    result = {'schema': 'rheon-shape-repeatability-comparison-v1', 'recorded_at_utc': utc(),
        'models': rows, 'new_cuda_updates': sum(r['updates'] for r in rows[1:]),
        'new_optimizer_seconds': sum(r['optimizer_seconds'] for r in rows[1:]),
        'sequence_elapsed_seconds': execution['elapsed_seconds'],
        'capture_elapsed_seconds': read(root/'capture.json')['elapsed_seconds'],
        'optimizer_intervals': intervals, 'optimizer_intervals_disjoint': True,
        'managed_stages': stages, 'sources': sources,
        'scope': 'managed sequence overlaps CPU qualification; sum of job times is not elapsed makespan; isolated inference latency unavailable'}
    write(args.reports/'comparison.json', result)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    (args.reports/'comparison.csv').write_text(stream.getvalue())
    print({'trials': len(rows)-1, 'new_cuda_updates': result['new_cuda_updates']})


if __name__ == '__main__':
    main()
