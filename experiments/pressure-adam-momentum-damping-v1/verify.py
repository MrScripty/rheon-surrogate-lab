"""Recompute guards and compare preserved controls against the prior CPU screen."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def verify(directory, output):
    if output.exists():
        raise ValueError('Fresh verification receipt required')
    result = read(directory / 'results.json')
    config, execution = result['config'], result['execution']
    prior = read(ROOT / config['prior_screen'])
    assert execution == read(directory / 'reports/execution.json')
    assert execution['config_sha256'] == sha(HERE / 'config.json')
    assert execution['source_sha256'] == sha(HERE / 'run.py')
    assert execution['status'] == 'completed' and execution['all_owned_children_reaped']
    assert execution['maximum_active_workers'] == 3 and execution['elapsed_s'] <= config['wall_seconds']
    assert execution['minimum_host_available_bytes'] >= config['host_reserve_bytes']
    assert all(w['reaped'] and w['returncode'] == 0 and
               w['peak_rss_bytes'] <= config['worker_rss_limit_bytes'] for w in execution['workers'])
    qualification = read(directory / 'reports/qualification.json')
    assert qualification == result['qualification'] and qualification['status'] == 'passed'
    assert qualification['source_sha256'] == sha(HERE / 'check.py')
    assert qualification['steps_source_sha256'] == sha(HERE / 'steps.py')
    passing, aliases, states = [], 0, 0
    assert [s['update'] for s in result['snapshots']] == config['updates']
    for snapshot in result['snapshots']:
        measurement = read(directory / 'reports' / ('update-%03d.json' % snapshot['update']))
        assert measurement['status'] == 'completed' and measurement['saved_state_vectors_unchanged']
        assert measurement['threads'] == 1 and measurement['parameter_dtype'] == 'float32'
        assert not measurement['cuda_initialized'] and measurement['optimizer_updates'] == measurement['teacher_forwards'] == 0
        assert measurement['vector_workspace_bound_bytes'] <= config['vector_workspace_limit_bytes']
        assert measurement['adam_reconstruction_max_absolute_error'] <= config['adam_parameter_absolute_tolerance']
        for name, identity in measurement['sources'].items():
            assert sha(ROOT / name) == identity, name
        assert snapshot['cases'] == measurement['cases']
        assert snapshot['measurements'] == measurement['evaluations']
        def metrics(row):
            return measurement['evaluations'][row['parameter_sha256']]['metrics']
        def pressure(values):
            return {(r['case'], r['passes']): r['relative_equation_residual'] for r in values['pressure']}
        expected = {(i, n) for i in range(8) for n in (1, 2, 4, 8)}
        for item in measurement['evaluations'].values():
            m = item['metrics']
            assert len(m['losses']) == 8 and set(pressure(m)) == expected
            assert all(math.isfinite(v) for v in list(pressure(m).values()) + m['losses'])
        rows = snapshot['rows']
        assert len(rows) == 7
        assert [r['factor'] for r in rows if r['policy'] == 'damped'] == config['retention_factors']
        assert [r['policy'] for r in rows] == ['before', 'observed'] + ['damped'] * 5
        controls = {r['policy']: r for r in rows if r['policy'] != 'damped'}
        baseline, observed = metrics(controls['before']), metrics(controls['observed'])
        assert snapshot['baseline'] == baseline and snapshot['observed_after'] == observed
        old = next(s for s in prior['snapshots'] if s['update'] == snapshot['update'])
        for policy, factor, old_policy, fraction in [('before', None, 'observed', 0),
                                                    ('observed', None, 'observed', 1),
                                                    ('damped', 1, 'preserved', 1)]:
            current = next(r for r in rows if r['policy'] == policy and r['factor'] == factor)
            historical = next(r for r in old['rows'] if r['policy'] == old_policy and r['fraction'] == fraction)
            assert current['parameter_sha256'] == historical['parameter_sha256']
            assert metrics(current) == old['measurements'][historical['parameter_sha256']]['metrics']
        bp, op = pressure(baseline), pressure(observed)
        gb, ga = measurement['cuda_observed_before'], measurement['cuda_observed_after']
        gbp, gap = pressure(gb), pressure(ga)
        weak, target = config['weak_case'], config['target_relative_residual']
        assert target == 1e-6 and weak == 7 and config['calibration_multiplier'] == 2
        loss_gap = max(abs(baseline['losses'][weak]-gb['losses'][weak]), abs(observed['losses'][weak]-ga['losses'][weak]))
        residual_gap = max(abs(bp[weak, 4]-gbp[weak, 4]), abs(op[weak, 4]-gap[weak, 4]))
        for row in rows:
            p = pressure(metrics(row))
            bad = [(i, n) for i in range(8) for n in (1, 2) if p[i, n] > bp[i, n]]
            lost = [(i, n) for i in range(8) for n in (4, 8) if bp[i, n] <= target < p[i, n]]
            robust = (baseline['losses'][weak]-metrics(row)['losses'][weak] > 2*loss_gap and
                      bp[weak, 4]-p[weak, 4] > 2*residual_gap)
            admitted = (row['policy'] == 'damped' and row['factor'] < 1 and row['delta_l2'] > 0
                        and robust and not bad and not lost and max(p[i, 4] for i in range(8)) < max(bp[i, 4] for i in range(8)))
            assert [(r['case'], r['passes']) for r in row['short_pass_regressions']] == bad
            assert [(r['case'], r['passes']) for r in row['lost_previously_fitted']] == lost
            assert row['weak_improvement_exceeds_empirical_gap'] == robust
            assert row['guard_passed'] == admitted
            assert row['weak_four_pass_residual'] == p[weak, 4]
            assert row['four_pass_qualified'] == sum(p[i, 4] <= target for i in range(8))
            assert row['eight_pass_qualified'] == sum(p[i, 8] <= target for i in range(8))
            assert row['cosine_to_preserved'] is None or abs(row['cosine_to_preserved']) <= 1+1e-12
            if admitted:
                passing.append(dict(update=snapshot['update'], policy=row['policy'], factor=row['factor']))
        if snapshot['update'] == 1:
            assert measurement['initialization_all_factors_exact']
            assert len({r['parameter_sha256'] for r in rows if r['policy'] == 'damped'}) == 1
        aliases += len(rows)
        states += len(measurement['evaluations'])
    assert result['candidate_aliases'] == aliases and result['unique_evaluations'] == states
    assert result['decision']['guard_passing_candidates'] == passing
    assert not result['decision']['model_promoted'] and not result['decision']['new_cuda_training_started']
    for name, identity in result['sources'].items():
        assert sha(ROOT / name) == identity, name
    cancellation = read(directory / 'reports/cancellation.json')
    assert cancellation['status'] == 'passed' and len(cancellation['tests']) == 2
    for test in cancellation['tests']:
        assert test['controller_reaped'] and test['execution']['status'] == 'cancelled'
        assert test['execution']['all_owned_children_reaped']
        assert test['execution']['source_sha256'] == execution['source_sha256']
    browser = read(directory / 'reports/browser.json')
    assert browser['status'] == 'passed' and browser['browser_reaped'] and not browser['console_errors']
    assert browser['html_sha256'] == sha(directory / 'index.html')
    repeat = read(directory / 'reports/repeatability.json')
    assert repeat['all_parameter_hashes_and_metrics_exact']
    for name in ('plan.md', 'commands.md', 'execution-ledger.md', 'next-steps.md'):
        assert (directory / name).is_file()
    write(output, dict(status='passed', unique_states=states, entries=aliases,
                       guard_passing_candidates=passing, historical_controls_exact=True,
                       all_sources_match=True, all_owned_children_reaped=True,
                       results_sha256=sha(directory / 'results.json'), source_sha256=sha(__file__),
                       scope='Five saved training states; consistency and scalar Adam oracles, no retrained trajectory or external physics conformance.'))
    print('Verified damping guards, preserved historical controls, source custody and lifecycle', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.directory.resolve(), args.output.resolve())
