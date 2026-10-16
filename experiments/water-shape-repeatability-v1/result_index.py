"""Index native-controlled populations and verify final selection ordering."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from execution import LAB, read, utc, write

sys.path.insert(0, str(LAB/'experiments/water-recurrence-generalization-v1'))
from finalize import population
from summarize import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--reference-record', type=Path, required=True)
    parser.add_argument('--reference-studies', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    sources = {}
    studies = {}
    for path in [*root.rglob('report.json'), *(p/'report.json' for p in args.reference_studies)]:
        report = read(path)
        if report.get('schema') == 'rheon-multigrid-trajectory-report-v1':
            sha = digest(path)
            studies[sha] = report
            sources[str(path)] = sha
    definitions = {'fresh_development': ['fresh-drift.json'],
                   'prior_grids': ['prior-drift.json'], 'long_development': ['long-drift.json'],
                   'final_holdout': ['final-holdout-0-drift.json','final-holdout-1-drift.json']}
    populations = {name: population([args.reports/p for p in paths], studies)
                   for name,paths in definitions.items()}
    reference = read(args.reference_record)
    alias = sequence['reference']['name']
    for name in ('fresh_development','prior_grids','long_development'):
        models = populations[name]['models']
        prior = models.pop(reference['trial'])
        if prior['weights_sha256'] != [reference['weights_sha256']]:
            raise ValueError('development reference identity differs')
        models[alias] = prior
    selection = read(args.reports/'final-selection.json')
    frozen = datetime.fromisoformat(selection['frozen_at_utc'])
    observed_final_steps = 0
    for index, config in enumerate(selection['configs']):
        report = read(root/f'final-holdout-{index}/report.json')
        if datetime.fromisoformat(report['started_at_utc']) <= frozen:
            raise ValueError('final population must execute after selection')
        if digest(LAB/config['runtime']) != config['runtime_sha256']:
            raise ValueError('frozen final configuration differs')
        candidate = next(m for m in report['models'] if m['name'] == selection['candidate']['name'])
        if candidate['weights_sha256'] != selection['candidate_weights_sha256']:
            raise ValueError('candidate changed after selection')
        for row in report['records']:
            if row['model'] == 'native':
                continue
            directory = root/f'final-holdout-{index}'/f'scene-{row["scene"]:02d}'/f'{row["model"]}-{row["repetition"]}'
            frames = [json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            frames = [f for f in frames if f.get('ai_quality')]
            callbacks = [json.loads(line) for line in (directory/'callbacks.jsonl').read_text().splitlines()]
            if len(frames) != len(callbacks) or len(callbacks) != len(row['network_passes']):
                raise ValueError('complete actual final inference observations required')
            if [c['network_passes'] for c in callbacks] != row['network_passes']:
                raise ValueError('callback work differs from retained summary')
            observed_final_steps += len(frames)
    native = [read(p) for p in root.glob('**/scene-*/*/receipt.json')]
    if not native or not all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in native):
        raise ValueError('native trajectory lifecycle must drain')
    result = {'schema': 'rheon-shape-repeatability-result-index-v1', 'recorded_at_utc': utc(),
        'comparison_sha256': digest(args.reports/'comparison.json'), 'populations': populations,
        'development_reference_alias': {reference['trial']: alias},
        'final_selection_sha256': digest(args.reports/'final-selection.json'),
        'final_population_executed_after_selection': True, 'final_population_now_consumed': True,
        'actual_final_ai_observations_verified': observed_final_steps,
        'native_trajectory_children_exit_zero_reaped': len(native), 'study_sources': sources,
        'deployment_changed': False,
        'scope': 'finite physical coverage with unchanged gates; CPU iterative inference, no isolated latency or universal convergence claim'}
    write(args.reports/'summary.json', result)
    print({'final_observations': observed_final_steps, 'native_children_reaped': len(native)})


if __name__ == '__main__':
    main()
