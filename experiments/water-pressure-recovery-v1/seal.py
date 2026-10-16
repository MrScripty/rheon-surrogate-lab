"""Seal completed consumed measurements after the frozen replacement gates fail."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'runs/water-pressure-recovery-v1'
REPORTS=LAB/'docs/plans/water-pressure-recovery-v1/reports'
sys.path.insert(0,str(LAB/'experiments/live-project-v1'))
from observer import Observer,atomic


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (REPORTS/'summary.json').exists()
    selection=read(REPORTS/'selection.json')
    assert not selection['eligible'], 'Eligible candidates require seed replication and independent validation first'
    assert read(ROOT/'completion-execution.json')['state']=='measurements_completed_fresh_deferred'
    assert read(ROOT/'cpu-continuation.json')['state']=='consumed_comparisons_completed'
    assert read(ROOT/'training-execution.json')['state']=='completed'
    for phase in ('policy','training'):
        root=ROOT/phase
        report=read(root/'development/report.json')
        assert report['state']=='completed' and report['all_qualified']
        assert len(report['records'])==222
        assert read(root/'fixture/report.json')['all_qualified']
        for item in read(root/'admission.json')['sources']:
            assert digest(LAB/item['path'])==item['sha256']
        assert not (root/'validation').exists() and not (root/'fresh-preflight').exists()
    assert not (LAB/'runs/water-pressure-bound-regression-v1/validation').exists()
    timing_path=REPORTS/selection['selected']['phase']/'serial-timing.json'
    timing=read(timing_path)
    assert timing['state']=='completed' and timing['all_qualified']
    commands=[]
    for path in sorted(ROOT.rglob('*-timing.json')):
        value=read(path)
        assert value['child_reaped'] and value['status']=='completed'
        commands.append({'path':str(path.relative_to(LAB)),'sha256':digest(path),
                         **{k:value[k] for k in ('argv','status','elapsed_seconds','started_at_utc','finished_at_utc','child_reaped')}})
    native=[]
    episodes=[]
    for path in sorted(ROOT.rglob('native/receipt.json')):
        value=read(path)
        assert value['native_process']['reaped'] and value['native_process']['returncode']==0
        assert not value['cuda_initialized'] and value['status']=='completed'
        assert not value['native_pressure_refinement']
        native.append({'path':str(path.relative_to(LAB)),'sha256':digest(path),
                       'native_process':value['native_process'],'status':value['status']})
        episodes.append({'path':str(path.relative_to(LAB)),'sha256':digest(path),'receipt':value})
    assert len(native)==462
    managed=[]
    for trial in ('smoke','gentle-010','gentle-025'):
        value=read(ROOT/trial/'managed-full/receipt.json')
        assert value['status']=='completed_and_drained'
        for stage in value['stages']:
            assert stage['state']=='completed' and stage['custody_phase']=='drained'
            managed.append({'trial':trial,'catalog_id':value['catalog_id'],'project_id':value['project_id'],
                            **{k:stage[k] for k in ('name','run_id','state','custody_phase')}})
    assert len(managed)==6 and len({r['run_id'] for r in managed})==6
    assert {r['catalog_id'] for r in managed}=={'a7da4a8de53541a2a0e2aad08005bb71'}
    peaks=[]
    for directory,pattern in ((ROOT,'diagnostics.json'),(REPORTS/'frontier','*.json'),
                              (REPORTS/'trained-frontier-gentle-010','*.json'),
                              (REPORTS/'trained-frontier-gentle-025','*.json')):
        for path in directory.rglob(pattern):
            value=read(path)
            if 'peak_rss_bytes' in value:peaks.append(value['peak_rss_bytes'])
    for path in REPORTS.glob('amplitude-gentle-*.json'):
        peaks.append(read(path)['peak_rss_bytes'])
    first=min(r['started_at_utc'] for r in commands)
    last=max(r['finished_at_utc'] for r in commands)
    summaries={p:read(REPORTS/p/'regression-summary.json') for p in ('policy','training')}
    ai_cases=[]
    for phase,summary in summaries.items():
        names={m['name'] for m in read(LAB/f'experiments/water-pressure-recovery-v1/configs/{phase}-regression.json')['models']}
        ai_cases.extend(c for c in summary['cases'] if c['model'] in names)
    assert len(ai_cases)==296 and all(c['qualified'] and c['complete_window'] for c in ai_cases)
    result={'schema':'rheon-pressure-recovery-completed-v1','state':'completed',
        'candidate_promoted':False,'deployment_changed':False,'eligible_models':[],
        'seed_replication':'deferred: comparative consumed gates failed',
        'fresh_validation':'deferred: comparative consumed gates failed','independent_scenes_consumed':0,
        'full_training_runs':2,'smoke_training_runs':1,'optimizer_updates':10001,
        'training':read(REPORTS/'training-clocks.json'),
        'regression':{p:s['groups'] for p,s in summaries.items()},
        'full_native_controls':148,'full_ai_trajectories':len(ai_cases),'full_ai_steps':sum(len(c['calls']) for c in ai_cases),
        'development_oracle_bound_checks':sum(s['independent_bound_checks'] for s in summaries.values()),
        'timing_diagnostic_model':selection['selected']['name'],
        'serial_timing_report':str(timing_path.relative_to(LAB)),'serial_timing':timing['summaries'],
        'first_measurement_utc':first,'last_measurement_utc':last,
        'measurement_span_seconds':(datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds(),
        'measurement_span_scope':'First timed source test through final timing command; includes overlapping jobs, custody/setup and documentation gaps; not the sum of component times.',
        'maximum_individual_cpu_diagnostic_rss_bytes':max(peaks),
        'raw_root_retained_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        'raw_size_scope':'This experiment root only, excluding the shared historical bank and Tuldok catalog.',
        'commands':commands,'native_children':native,'managed_runs':managed,
        'source_sha256':digest(__file__),'source_invariants_rechecked':True}
    (REPORTS/'episodes.json').write_text(json.dumps(episodes,indent=2,allow_nan=False)+'\n')
    (REPORTS/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    atomic(REPORTS/'current.json',{'state':'completed','candidate_promoted':False,'independent_scenes_consumed':0,
        'eligible_models':[],'summary_sha256':digest(REPORTS/'summary.json'),
        'training_seconds':{n:v['optimizer_seconds'] for n,v in result['training'].items()},
        'policy':summaries['policy']['groups'],'training':summaries['training']['groups']})
    Observer(LAB/'runs/live-project').publish('Pressure recovery · completed',
        'All296 consumed AI trajectories qualified; comparative replacement gates failed; fresh8 unused',296,296,state='completed')
    print(json.dumps({k:result[k] for k in ('state','full_ai_trajectories','development_oracle_bound_checks',
                                          'measurement_span_seconds','maximum_individual_cpu_diagnostic_rss_bytes')}))


if __name__=='__main__':
    main()
