"""Index sealed training records and native-controlled trajectory populations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from summarize import digest, read


POPULATIONS = {
    'fresh_development': ['fresh-pressure-drift.json'],
    'long_development': ['long-pressure-drift.json'],
    'prior_grids': ['prior-pressure-drift.json'],
    'final_holdout': ['final-holdout-0-drift.json', 'final-holdout-1-drift.json'],
}


def population(paths, studies):
    cases = [case for path in paths for case in read(path)['cases']]
    source_hashes = {case['source_report_sha256'] for case in cases}
    controls = []
    for sha in source_hashes:
        controls.extend(row for row in studies[sha]['records'] if row['model'] == 'native')
    if not all(row['returncode'] == 0 and row['all_ai_steps_qualified']
               and row['all_reference_steps_qualified'] and not row['result']['hard_stop']
               for row in controls):
        raise ValueError('native controls must qualify before interpreting candidate failures')
    models = {}
    for name in sorted({case['model'] for case in cases}):
        rows = [case for case in cases if case['model'] == name]
        counts = []
        for case in rows:
            report = studies[case['source_report_sha256']]
            observations = [row for row in report['records']
                            if row.get('receipt_sha256') == case['source_receipt_sha256']]
            if len(observations) != 1 or len(observations[0]['network_passes']) != case['ai_steps']:
                raise ValueError('exact complete inference work inventory required')
            counts.extend(observations[0]['network_passes'])
        models[name] = {
            'scenes': len(rows),
            'qualified_scenes': sum(case['completed_requested_window']
                                    and case['qualified_ai_steps'] == case['ai_steps'] for case in rows),
            'attempted_ai_steps': sum(case['ai_steps'] for case in rows),
            'qualified_ai_steps': sum(case['qualified_ai_steps'] for case in rows),
            'hard_stops': sum(case['hard_stop'] for case in rows),
            'mean_network_evaluations': statistics.mean(counts),
            'maximum_network_evaluations': max(counts),
            'cap_hits': counts.count(32),
            'worst_pressure_relative_l2': max(case['worst_pressure_relative_l2'] for case in rows),
            'median_final_pressure_relative_l2': statistics.median(
                case['final_pressure_relative_l2'] for case in rows),
            'weights_sha256': sorted({case['weights_sha256'] for case in rows}),
        }
    return {'models': models, 'native_controls': len(controls),
            'all_native_controls_qualified': True,
            'source_reports': {str(path): digest(path) for path in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    studies = {}
    study_sources = {}
    for path in args.root.rglob('report.json'):
        report = read(path)
        if report.get('schema') == 'rheon-multigrid-trajectory-report-v1':
            sha = digest(path)
            studies[sha] = report
            study_sources[str(path)] = sha
    populations = {name: population([args.reports/p for p in paths], studies)
                   for name, paths in POPULATIONS.items()}
    selection = read(args.reports/'final-selection.json')
    frozen = datetime.fromisoformat(selection['frozen_at_utc'])
    final_reports = [read(args.root/f'final-holdout-{i}/report.json')
                     for i in range(len(selection['configs']))]
    for config, report in zip(selection['configs'], final_reports):
        if datetime.fromisoformat(report['started_at_utc']) <= frozen:
            raise ValueError('final population must execute after candidate selection')
        if digest(Path(config['runtime'])) != config['runtime_sha256']:
            raise ValueError('frozen final configuration changed')
        candidate = next(model for model in report['models']
                         if model['name'] == selection['candidate']['name'])
        if candidate['weights_sha256'] != selection['candidate_weights_sha256']:
            raise ValueError('candidate changed after selection')
    comparison = read(args.reports/'comparison.json')
    trials = [read(path) for path in sorted(args.reports.glob('trial-*.json'))]
    stages = [stage for trial in trials for stage in trial['managed_stages']]
    if len({stage['run_id'] for stage in stages}) != len(stages):
        raise ValueError('count each actual managed Run once')
    intervals = []
    for trial in trials:
        history = read(args.root/trial['trial']/'managed-full/stage-00/artifacts/history.json')
        intervals.append({'trial': trial['trial'], 'started_unix': history['training_started_unix'],
                          'finished_unix': history['training_finished_unix']})
    intervals.sort(key=lambda value: value['started_unix'])
    if any(a['finished_unix'] > b['started_unix'] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('one optimizer policy violated')
    native = [read(path) for path in args.root.rglob('receipt.json')
              if path.parent.parent.name.startswith('scene-')]
    if not all(case['native_process']['reaped'] and case['native_process']['returncode'] == 0
               for case in native):
        raise ValueError('native trajectory lifecycle not drained')
    result = {
        'schema': 'rheon-recurrence-final-index-v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'new_cuda_updates': comparison['new_cuda_updates'],
        'completed_drained_managed_stages': len(stages),
        'optimizer_intervals': intervals,
        'optimizer_intervals_disjoint': True,
        'training_comparison': comparison['models'],
        'populations': populations,
        'final_selection_sha256': digest(args.reports/'final-selection.json'),
        'final_population_executed_after_selection': True,
        'final_population_now_consumed': True,
        'native_trajectory_children_exit_zero_reaped': len(native),
        'study_sources': study_sources,
        'deployment_changed': False,
        'scope': 'float32 learned parameters; iterative CPU inference capped at32; unchanged physics gates; no isolated timing or universal resolution claim',
    }
    (args.reports/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'updates': result['new_cuda_updates'], 'managed_stages': len(stages),
                      'native_trajectory_children_reaped': len(native),
                      'final_population': populations['final_holdout']['models']}))


if __name__ == '__main__':
    main()
