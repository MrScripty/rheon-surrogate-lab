"""Recompute cohort-2 gates and paired comparisons from recorded per-case evidence."""
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def report():
    runs = ROOT/'runs/pressure-methods-v1'
    reports = ROOT/'docs/plans/pressure-methods-v1'
    config = read(runs/'config.json')
    execution = read(reports/'cohort-02-execution.json')
    if execution['status'] != 'completed' or not all(w['reaped'] and w['returncode'] == 0 for w in execution['workers']):
        raise ValueError('All screen workers must complete and drain')
    controls, comparisons, evidence, tables = [], [], {}, []
    for identifier in config['checkpoints']:
        path = runs/'cohort-02'/identifier/'results.json'
        data = read(path)
        if (data['status'] != 'completed' or data['config_sha256'] != sha(runs/'config.json')
                or data['bank_sha256'] != sha(runs/'manufactured-bank/bank.json')
                or data['optimizer_updates'] != 0 or data['cuda_initialized']):
            raise ValueError('Worker closure failed: '+identifier)
        evidence[str(path.relative_to(ROOT))] = sha(path)
        actual = {}
        for summary in data['summary']:
            group = [r for r in data['rows'] if r['population'] == summary['population'] and r['policy'] == summary['policy']]
            expected_size = dict(consumed128=128, manufactured240=240, mixed32=32)[summary['population']]
            if len(group) != expected_size or len({r['case'] for r in group}) != expected_size:
                raise ValueError('Incomplete/duplicate case population')
            recomputed = dict(cases=len(group), qualified=sum(r['relative_equation_residual'] <= config['target_relative_residual'] for r in group),
                worst_relative_equation_residual=max(r['relative_equation_residual'] for r in group),
                median_relative_equation_residual=statistics.median(r['relative_equation_residual'] for r in group),
                worst_pressure_relative_l2=max(r['mean_free_pressure_relative_l2'] for r in group),
                worst_gradient_relative_l2=max(r['gradient_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(statistics.median(r['timing_samples_s']) for r in group))
            if any(summary[k] != v for k, v in recomputed.items()):
                raise ValueError('Summary differs from raw measurements')
            actual[summary['population'], summary['policy']] = summary
        prior_path = ROOT/'runs/pressure-detached-correction-v1/cpu-01'/identifier/'snapshots.json'
        prior = read(prior_path)
        if prior['weights_sha256'] != data['weights_sha256']:
            raise ValueError('Prior baseline weights differ')
        for n in config['neural_only_passes']:
            expected = {(r['trajectory_id'], r['native_frame']):r['relative_residual'] for r in prior['rows'] if r['passes'] == n}
            got = {(r['trajectory_id'], r['native_frame']):r['relative_equation_residual'] for r in data['rows'] if r['population'] == 'consumed128' and r['policy'] == f'neural-{n}'}
            if got != expected:
                raise ValueError('Pure-neural baseline does not exactly reproduce prior consumed-request results')
        controls.append(dict(checkpoint=identifier, weights_sha256=data['weights_sha256'], prior_exact=True,
                             prior_path=str(prior_path.relative_to(ROOT)), prior_sha256=sha(prior_path)))
        for population in ['consumed128', 'manufactured240', 'mixed32']:
            for ordering in config['orderings']:
                for n in config['hybrid_passes']:
                    control = actual[population, f'neural-{n}']
                    candidate = actual[population, f'{ordering}-{n}']
                    comparisons.append(dict(checkpoint=identifier, population=population, policy=f'{ordering}-{n}',
                        control=f'neural-{n}', worst_error_ratio=candidate['worst_relative_equation_residual']/control['worst_relative_equation_residual'],
                        median_time_ratio=candidate['median_policy_ms']/control['median_policy_ms'],
                        success_change=candidate['qualified']-control['qualified'],
                        dominates_matched_control=(candidate['worst_relative_equation_residual'] <= control['worst_relative_equation_residual']
                            and candidate['median_policy_ms'] <= control['median_policy_ms']
                            and candidate['qualified'] >= control['qualified']
                            and (candidate['worst_relative_equation_residual'] < control['worst_relative_equation_residual']
                                 or candidate['median_policy_ms'] < control['median_policy_ms']))))
        tables.append((identifier, actual))
    full = tables[0][1]['consumed128', 'numerical-full-only']
    c53eight = tables[0][1]['consumed128', 'neural-8']
    result = dict(status='offline_screen_verified', controls=controls, comparisons=comparisons,
        full_numerical_control=dict(**full, c53_eight_pass_time_ratio=full['median_policy_ms']/c53eight['median_policy_ms']),
        evidence=evidence, source_sha256=sha(__file__), new_optimizer_updates=0, holdout_scenes_consumed=0,
        native_followup_required=True, all_eight_cohorts_complete=False,
        conclusion='Report accuracy/time tradeoffs separately; no hybrid policy qualifies deployment from this offline screen. Full numerical control triggers native and whole-policy cost follow-up.')
    write(reports/'cohort-02-results.json', result)
    lines = ['# Frozen-model numerical correction: offline screen', '',
        'All three workers completed. Pure-neural residuals exactly reproduce the prior',
        '128-request measurements at every tested pass count. No new training or',
        'unused scenes were consumed. This completes the first offline screen, not',
        'the native follow-up or the eight-cohort campaign.', '',
        '| Checkpoint | Policy | Real requests at 1e-6 | Worst real residual | Median policy ms |',
        '| --- | --- | ---: | ---: | ---: |']
    for identifier, actual in tables:
        for name in ['neural-4', 'before-4', 'after-4', 'neural-8']:
            r = actual['consumed128', name]
            lines.append(f"| {identifier} | {name} | {r['qualified']}/128 | {r['worst_relative_equation_residual']:.6g} | {r['median_policy_ms']:.3f} |")
    lines += [f"| No network | Full numerical spectral | {full['qualified']}/128 | {full['worst_relative_equation_residual']:.6g} | {full['median_policy_ms']:.3f} |", '',
        'These timings start from prepared physical tensors and end after the final',
        'equation check. They include numerical transforms, reconstruction and neural',
        'recurrence; they exclude loading, reference metrics, IPC and simulation.',
        'They must not be compared directly with the existing HTML chart times.', '',
        'Numerical low-mode success on pure modes inside its subspace follows by',
        'construction. The 240 manufactured cases and 32 mixed cases are separate',
        'populations in the raw reports; pure-mode success does not establish',
        'generalization. Per-ordering/per-checkpoint tradeoffs remain explicit.', '',
        'The full numerical solver is promising under the current sealed uniform-box',
        'contract. Follow up with independent native trajectories and full-policy',
        'timing; this does not demonstrate a learned hybrid or arbitrary geometry.', '',
        '[Scalar comparisons and evidence hashes](cohort-02-results.json).', '']
    (reports/'cohort-02-results.md').write_text('\n'.join(lines))
    campaign = read(reports/'campaign.json')
    for row in campaign['cohorts']:
        if row['id'] == 2:
            row.update(status='offline_screen_completed_followup_pending', results='cohort-02-results.json')
        elif row['id'] in [1, 3, 7]:
            row.update(status='native_pilot_producer_prepared')
    write(reports/'campaign.json', campaign)
    print('Verified offline screen and wrote cohort report', flush=True)


if __name__ == '__main__':
    report()
