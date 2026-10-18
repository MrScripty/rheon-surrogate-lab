"""Check retained metric/guard consistency, custody and bounded process receipts."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def verify(directory,output):
    if output.exists():raise ValueError('Fresh verification receipt required')
    result=read(directory/'results.json');config=result['config'];execution=result['execution']
    assert result['status']=='completed' and not result['cuda_initialized'] and result['optimizer_updates']==0
    assert execution==read(directory/'reports/execution.json')
    assert execution['status']=='completed' and execution['all_owned_children_reaped']
    assert execution['maximum_active_workers']==3 and all(w['reaped'] and w['returncode']==0 for w in execution['workers'])
    assert execution['minimum_host_available_bytes']>=config['host_reserve_bytes']
    assert all(w['peak_rss_bytes']<=config['worker_rss_limit_bytes'] for w in execution['workers'])
    assert [s['update']for s in result['snapshots']]==config['updates']
    assert result['qualification']['status']=='passed'
    passing=[];aliases=0;states=0
    for snapshot in result['snapshots']:
        measurement=read(directory/'reports'/('update-%03d.json'%snapshot['update']))
        assert measurement['cpu']==snapshot['cpu'] and measurement['threads']==1
        assert measurement['parameter_dtype']=='float32' and not measurement['cuda_initialized']
        assert measurement['optimizer_updates']==measurement['teacher_forwards']==0
        assert measurement['vector_workspace_bound_bytes']<=config['vector_workspace_limit_bytes']
        assert measurement['adam_reconstruction_max_absolute_error']<=config['adam_parameter_absolute_tolerance']
        for name,identity in measurement['sources'].items():assert sha(ROOT/name)==identity,name
        expected={(i,n)for i in range(8)for n in config['passes']}
        for item in measurement['evaluations'].values():
            values=item['metrics'];assert len(values['losses'])==8
            assert {(p['case'],p['passes'])for p in values['pressure']}==expected
            assert all(math.isfinite(p['relative_equation_residual']) for p in values['pressure'])
        def metrics(row):return measurement['evaluations'][row['parameter_sha256']]['metrics']
        def pressure(values):return {(p['case'],p['passes']):p['relative_equation_residual']for p in values['pressure']}
        baseline=metrics(next(r for r in snapshot['rows']if r['policy']=='observed' and r['fraction']==0))
        observed=metrics(next(r for r in snapshot['rows']if r['policy']=='observed' and r['fraction']==1))
        bp,op=pressure(baseline),pressure(observed);weak=config['weak_case'];target=config['target_relative_residual']
        gpu_before,gpu_after=measurement['cuda_observed_before'],measurement['cuda_observed_after']
        gbp,gap=pressure(gpu_before),pressure(gpu_after)
        loss_gap=max(abs(baseline['losses'][weak]-gpu_before['losses'][weak]),abs(observed['losses'][weak]-gpu_after['losses'][weak]))
        residual_gap=max(abs(bp[weak,4]-gbp[weak,4]),abs(op[weak,4]-gap[weak,4]))
        assert len(snapshot['rows'])==15 and len({r['parameter_sha256']for r in snapshot['rows']if r['fraction']==0})==1
        if snapshot['update']==1:
            assert measurement['initialized_reset_control_exact']
            for fraction in config['fractions']:
                assert len({r['parameter_sha256']for r in snapshot['rows']if r['policy']in ('preserved','reset') and r['fraction']==fraction})==1
        for row in snapshot['rows']:
            values=metrics(row);p=pressure(values)
            short_bad=[(i,n)for i in range(8)for n in (1,2)if p[i,n]>bp[i,n]]
            lost=[(i,n)for i in range(8)for n in (4,8)if bp[i,n]<=target and p[i,n]>target]
            robust=baseline['losses'][weak]-values['losses'][weak]>2*loss_gap and bp[weak,4]-p[weak,4]>2*residual_gap
            expected_guard=row['fraction']>0 and not short_bad and not lost and robust and max(p[i,4]for i in range(8))<max(bp[i,4]for i in range(8))
            assert row['weak_four_pass_residual']==p[weak,4] and row['weak_loss']==values['losses'][weak]
            assert [(r['case'],r['passes'])for r in row['short_pass_regressions']]==short_bad
            assert [(r['case'],r['passes'])for r in row['lost_previously_fitted']]==lost
            assert row['guard_passed']==expected_guard and row['weak_improvement_exceeds_empirical_gap']==robust
            if expected_guard:passing.append(dict(update=snapshot['update'],policy=row['policy'],fraction=row['fraction']))
        aliases+=len(snapshot['rows']);states+=len(measurement['evaluations'])
    assert result['candidate_aliases']==aliases and result['unique_evaluations']==states
    assert result['decision']['guard_passing_candidates']==passing
    assert not result['decision']['model_promoted'] and not result['decision']['new_cuda_training_started']
    for name,identity in result['sources'].items():assert sha(ROOT/name)==identity,name
    prior=read(ROOT/config['replay_results']);last=result['snapshots'][-1]['observed_after']
    final=pressure(last)
    assert all(final[r['case'],r['passes']]==r['replay'] for r in prior['final_pressure_comparison'])
    browser=read(directory/'reports/browser.json');assert browser['status']=='passed' and browser['browser_reaped'] and not browser['console_errors']
    assert browser['html_sha256']==sha(directory/'index.html')
    cancelled=read(directory/'reports/cancellation.json')
    assert cancelled['status']=='passed' and len(cancelled['tests'])==2
    for case in cancelled['tests']:
        assert case['controller_reaped'] and case['execution']['status']=='cancelled' and case['execution']['all_owned_children_reaped']
        assert case['execution']['source_sha256']==execution['source_sha256']
    assert sha(Path(__file__).with_name('run.py'))==execution['source_sha256']
    assert read(directory/'reports/workspace-review.json')['all_parameter_hashes_metrics_calibrations_guards_exact']
    for name in ('plan.md','commands.md','execution-ledger.md','next-steps.md'):assert (directory/name).is_file()
    write(output,dict(status='passed',actual_candidate_entries=aliases,unique_measured_parameter_states=states,
        independently_recomputed_guard_passing_candidates=passing,observed_final_cpu_pressure_replay_exact=True,
        all_sources_match=True,all_owned_children_reaped=True,cancellation_boundary_passed=True,
        scientific_scope='Metric/guard/byte/lifecycle consistency on five consumed training snapshots; no external physics conformance, CUDA retraining or rollout claim.',
        results_sha256=sha(directory/'results.json'),html_sha256=sha(directory/'index.html'),source_sha256=sha(__file__)))
    print('Verified measurements, strict guards, source custody and worker lifecycle',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.directory.resolve(),a.output.resolve())
