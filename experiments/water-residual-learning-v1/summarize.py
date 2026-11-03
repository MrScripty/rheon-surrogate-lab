"""Build a reviewable comparison index from retained immutable trial reports."""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]
REPORTS=LAB/'docs/plans/water-residual-learning-v1/reports'


def main():
    records=[];stages=[];rows=[]
    for path in sorted(REPORTS.glob('*.json')):
        value=json.loads(path.read_text())
        if value.get('schema') not in ('rheon-residual-experiment-record-v1','rheon-residual-experiment-record-v2'):continue
        records.append({'trial':value['trial'],'report':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        stages.extend({'trial':value['trial'],**{key:stage[key] for key in ('name','run_id','state','custody_phase','catalog_id')}} for stage in value['managed_stages'])
        fitted=value['native_evaluations'].get('fitted-evaluation');validation=value['native_evaluations'].get('validation-evaluation')
        if fitted is None:continue
        history=None
        root=LAB/value['retained_artifacts']
        for history_path in sorted(root.glob('managed-*/stage-*/artifacts/history.json')):
            history=json.loads(history_path.read_text());break
        assert history is not None
        row={'trial':value['trial'],'source_commit':value['method']['source_commit'],'updates':value['cuda_optimizer_updates'],
            'steps_per_prediction':value['method']['configuration'].get('correction',{}).get('steps',1),
            'training_started_utc':datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            'training_finished_utc':datetime.fromtimestamp(history['training_finished_unix'],timezone.utc).isoformat(),
            'updates_per_s':value['optimizer_updates_per_s'],'fitted_relative_residual':fitted['relative_residual']['median'],
            'fitted_qualified':fitted['float32_qualified'],
            'validation_median_relative_residual':validation['relative_residual']['median'] if validation else None,
            'validation_max_relative_residual':validation['relative_residual']['maximum'] if validation else None,
            'validation_qualified':validation['float32_qualified'] if validation else None,
            'validation_cases':validation['cases'] if validation else None,
            'validation_repeated8_median':value['cpu_audits']['validation-audit']['median_repeated_8'],
            'validation_superposition_median':value['cpu_audits']['validation-audit']['median_superposition_error'],
            'trajectory_scenarios':len(value.get('trajectories',{})),
            'trajectory_ai_steps':sum(t['result']['ai_steps'] for t in value.get('trajectories',{}).values()),
            'trajectory_qualified_ai_steps':sum(t['result']['qualified_ai_steps'] for t in value.get('trajectories',{}).values()),
            'trajectory_hard_stops':sum(t['result']['hard_stop'] for t in value.get('trajectories',{}).values()),
            'weights_sha256':value['weights_sha256'],'report':path.name}
        rows.append(row)
    assert len({s['run_id'] for s in stages})==len(stages)
    assert all(s['state']=='completed' and s['custody_phase']=='drained' for s in stages)
    summary={'schema':'rheon-residual-learning-summary-v1','scope':'known-frame fitting and development evaluation; no untouched test or qualified deployment',
        'model_accuracy_qualified':any(r['fitted_qualified']==1 and r['trajectory_scenarios']==3 and r['validation_qualified']==r['validation_cases'] and r['validation_cases'] and not r['trajectory_hard_stops'] and r['trajectory_qualified_ai_steps']==r['trajectory_ai_steps'] and r['trajectory_ai_steps'] for r in rows),
        'relative_residual_target':1e-6,'physical_gates_changed':False,'new_reserved_test_used':False,
        'cuda_optimizer_updates':sum(json.loads((REPORTS/r['report']).read_text())['cuda_optimizer_updates'] for r in records),
        'managed_stages':stages,'records':records,'comparison':rows,
        'baselines':{'matched_historical_no_groupnorm_one_frame':{'fitted_relative_residual':.0003982221868309573,'validation_median_relative_residual':.0828159014298824},
            'prior_1024_parent_different_training_population':{'validation_median_relative_residual':.0385792962090315}},
        'cohort16':'prepared; training remains conditional on known-case physical accuracy'}
    (REPORTS/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if rows:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        (REPORTS/'comparison.csv').write_text(stream.getvalue())
    print(json.dumps({'recorded_trials':len(records),'managed_stages':len(stages),'cuda_updates':summary['cuda_optimizer_updates'],'comparison':rows}),flush=True)


if __name__=='__main__':main()
