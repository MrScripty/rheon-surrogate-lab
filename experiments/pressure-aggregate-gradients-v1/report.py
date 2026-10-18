"""Project immutable geometry receipts into an explicitly non-causal report."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def report(config_path, execution_path, directory):
    if (directory/'results.json').exists():
        raise ValueError('Fresh report required')
    config = read(config_path)
    execution = read(execution_path)
    assert execution['status'] == 'completed'
    assert all(row['reaped'] and row['returncode'] == 0 for row in execution['workers'])
    models, summary, sources = {}, [], {}
    target = config['target_case']
    for identifier in config['models']:
        path = directory/'reports'/(identifier+'-gradients.json')
        row = read(path)
        models[identifier] = {key:value for key,value in row.items() if key not in ('vectors','sources','across','component_to_full_dots')}
        models[identifier]['term_norms'] = {name:group['norms'] for name,group in row['across'].items()}
        policies = {}
        for name, policy in row['policies'].items():
            total = sum(case['gradient_l2']*case['pre_mean_cap_scale'] for case in policy['cases'])
            policies[name] = dict(target_cosine=policy['cases'][target]['cosine_mean'],
                target_final_cosine=policy['cases'][target]['terms']['final']['cosine_mean'],
                target_early_cosine=policy['cases'][target]['terms']['early']['cosine_mean'],
                opposing_cases=[case['case'] for case in policy['cases'] if case['cosine_mean'] is not None and case['cosine_mean'] < 0],
                effective_norm_budget_shares=[case['gradient_l2']*case['pre_mean_cap_scale']/total if total else None for case in policy['cases']])
        limited = sum(not case['separate_gradient_precision_check_passed'] for case in row['cases'])
        summary.append(dict(model=identifier, policies=policies, separate_gradient_precision_limited_cases=limited,
            worst_separate_gradient_relative_l2=max(case['separate_gradient_relative_l2'] for case in row['cases']),
            early_gradient_exceeds_final=sum(a>b for a,b in zip(row['across']['early']['norms'],row['across']['final']['norms'],strict=True)),
            elapsed_s=row['elapsed_s'], peak_rss_bytes=row['peak_rss_bytes']))
        sources[str(path.relative_to(ROOT))] = sha(path)
    current = next(row for row in summary if row['model'] == 'C69')
    assert current['policies']['raw_mean']['target_cosine'] > 0
    assert current['policies']['raw_mean']['target_final_cosine'] > 0
    assert current['policies']['raw_mean']['target_early_cosine'] > 0
    decision = dict(status='diagnosis_completed_with_precision_limits', baseline_retained='C53',
        model_promoted=False, training_started=False, optimizer_updates=0,
        joint_gradient_equivalence_passes=sum(case['joint_gradient_exact'] for row in models.values() for case in row['cases']),
        separate_gradient_precision_limited_cases=sum(row['separate_gradient_precision_limited_cases'] for row in summary),
        finding='Case 7 opposes the common mean at C53/C66, but aligns with full/final/early mean directions at C69. Per-case capping preserves the number of opposed cases and does not explain the stalled training.',
        next_action='Propose a bounded matched C69 replay retaining actual Adam moments and per-cycle update directions before choosing a new optimizer/loss policy. No training started.')
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in
        (config_path, execution_path, Path(__file__).resolve())})
    write(directory/'results.json', dict(schema='rheon-aggregate-gradient-diagnosis-results-v1', status='completed',
        target_case=target, models=models, summary=summary, execution=execution, decision=decision, sources=sources))
    write(directory/'decision.json', decision)
    print('Recorded aggregate direction diagnosis; target C69 raw/capped',
        [row['target_cosine'] for row in current['policies'].values()], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'execution', 'directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report(args.config.resolve(), args.execution.resolve(), args.directory.resolve())
