"""Retain strict guards, empirical value calibration and an offline step viewer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def summarize(config_path,execution_path,directory):
    if (directory/'results.json').exists():raise ValueError('Fresh summary required')
    config=read(config_path);execution=read(execution_path)
    assert execution['status']=='completed' and execution['all_owned_children_reaped']
    assert execution['maximum_active_workers']<=config['workers']
    assert execution['config_sha256']==sha(config_path)
    qualification=read(directory/'reports/qualification.json')
    assert qualification['status']=='passed' and qualification['full_retention_exact'] and qualification['zero_retention_differs_from_full_reset']
    snapshots=[];sources=[config_path,execution_path,HERE/'report.py',HERE/'viewer.html']
    for update in config['updates']:
        path=directory/'reports'/('update-%03d.json'%update);measurement=read(path);sources.append(path)
        assert measurement['status']=='completed' and measurement['update']==update
        assert not measurement['cuda_initialized'] and measurement['optimizer_updates']==0 and measurement['teacher_forwards']==0
        assert measurement['adam_reconstruction_max_absolute_error']<=config['adam_parameter_absolute_tolerance']
        if update==1:assert measurement['initialization_all_factors_exact']
        for name,identity in measurement['sources'].items():assert sha(ROOT/name)==identity,name
        def metrics(policy,factor=None):
            candidate=next(r for r in measurement['candidates'] if r['policy']==policy and r['factor']==factor)
            return measurement['evaluations'][candidate['parameter_sha256']]['metrics']
        def pressures(m):return {(r['case'],r['passes']):r['relative_equation_residual'] for r in m['pressure']}
        baseline=metrics('before');after=metrics('observed')
        bp=pressures(baseline);ap=pressures(after)
        gb,ga=measurement['cuda_observed_before'],measurement['cuda_observed_after']
        gbp,gap=pressures(gb),pressures(ga)
        loss_gaps=[max(abs(baseline['losses'][i]-gb['losses'][i]),abs(after['losses'][i]-ga['losses'][i])) for i in range(8)]
        pressure_gaps={key:max(abs(bp[key]-gbp[key]),abs(ap[key]-gap[key])) for key in bp}
        weak=config['weak_case'];target=config['target_relative_residual'];rows=[]
        for candidate in measurement['candidates']:
            m=measurement['evaluations'][candidate['parameter_sha256']]['metrics'];p=pressures(m)
            weak_loss_improvement=baseline['losses'][weak]-m['losses'][weak]
            weak_pressure_improvement=bp[weak,4]-p[weak,4]
            short=[dict(case=i,passes=n,before=bp[i,n],after=p[i,n]) for i in range(8) for n in (1,2) if p[i,n]>bp[i,n]]
            lost=[dict(case=i,passes=n,before=bp[i,n],after=p[i,n]) for i in range(8) for n in (4,8) if bp[i,n]<=target<p[i,n]]
            robust=(weak_loss_improvement>config['calibration_multiplier']*loss_gaps[weak]
                and weak_pressure_improvement>config['calibration_multiplier']*pressure_gaps[weak,4])
            worst=max(p[i,4] for i in range(8));worst_before=max(bp[i,4] for i in range(8))
            rows.append(dict(**candidate,weak_loss=m['losses'][weak],weak_four_pass_residual=p[weak,4],
                weak_loss_improvement=weak_loss_improvement,weak_pressure_improvement=weak_pressure_improvement,
                mean_loss=sum(m['losses'])/8,worst_four_pass_residual=worst,
                four_pass_qualified=sum(p[i,4]<=target for i in range(8)),eight_pass_qualified=sum(p[i,8]<=target for i in range(8)),
                short_pass_regressions=short,lost_previously_fitted=lost,
                weak_improvement_exceeds_empirical_gap=robust,
                guard_passed=candidate['policy']=='damped' and candidate['factor']<1 and candidate['delta_l2']>0 and robust and not short and not lost and worst<worst_before))
        assert measurement['saved_state_vectors_unchanged']
        assert len(rows)==7
        snapshots.append(dict(update=update,learning_rate=measurement['learning_rate'],cases=measurement['cases'],rows=rows,
            calibration=dict(loss_absolute_gaps=loss_gaps,pressure_absolute_gaps=[dict(case=i,passes=n,gap=v) for (i,n),v in sorted(pressure_gaps.items())],
                multiplier=config['calibration_multiplier'],scope='Empirical pre/post value gaps only; not a bound on unmeasured states or a relaxed physical acceptance target.'),
            baseline=baseline,observed_after=after,measurements=measurement['evaluations'],
            unique_states=len(measurement['evaluations']),elapsed_s=measurement['elapsed_s'],evaluation_s=measurement['evaluation_s'],
            peak_rss_bytes=measurement['peak_rss_bytes'],cpu=measurement['cpu'],
            adam_reconstruction_max_absolute_error=measurement['adam_reconstruction_max_absolute_error']))
    passing=[dict(update=s['update'],policy=r['policy'],factor=r['factor']) for s in snapshots for r in s['rows'] if r['guard_passed']]
    decision=dict(status='cpu_screen_completed',model_promoted=False,baseline_retained='C53',new_cuda_training_started=False,
        guard_passing_candidates=passing,matched_training_policy_qualified=False,
        reason='One-step CPU improvements do not qualify a CUDA training trajectory. Inspect strict guard failures and snapshot coverage before choosing one policy.')
    result=dict(schema='rheon-adam-momentum-screen-results-v1',status='completed',config=config,snapshots=snapshots,decision=decision,
        execution=execution,qualification=qualification,optimizer_updates=0,cuda_initialized=False,
        unique_evaluations=sum(s['unique_states'] for s in snapshots),candidate_aliases=sum(len(s['rows']) for s in snapshots),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        limits=['Five consumed training snapshots; no simulated rollout or retrained history.',
            'Same captured CUDA gradient used for every Adam endpoint; no CPU derivative substitution.',
            'Only the saved first moment is damped; second moment, bias-correction counter, gradient and LR retained. Step length can change.',
            'CPU/GPU value gaps are empirical cautions, not rigorous numerical bounds.',
            'Strict short-pass and previously-fitted guards are never relaxed by the empirical gap.'])
    write(directory/'results.json',result)
    write(directory/'decision.json',decision)
    print('Guard-passing candidates:',passing,flush=True)
    for snapshot in snapshots:
        best=min((r for r in snapshot['rows'] if r['policy']=='damped' and r['factor']<1),key=lambda r:r['weak_four_pass_residual'])
        print(snapshot['update'],'best weak',best['policy'],best['factor'],best['weak_four_pass_residual'],
            'short regressions',len(best['short_pass_regressions']),'lost fitted',len(best['lost_previously_fitted']),flush=True)
    render(result,directory)


def render(result, directory):
    template = Path(__file__).with_name('viewer.html').read_text()
    payload = json.dumps(result, allow_nan=False).replace('</', '<\\/')
    (directory / 'index.html').write_text(template.replace('STUDY_DATA', payload))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'execution', 'directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.config.resolve(), args.execution.resolve(), args.directory.resolve())
