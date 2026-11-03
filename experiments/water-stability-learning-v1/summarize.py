"""Derive comparisons from immutable trial records; distinguish inference policies."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]


def row(record,policy=1):
    if policy==1:
        fit=record['native_evaluations'].get('fitted-evaluation')
        validation=record['native_evaluations'].get('validation-evaluation')
    else:
        fit=record.get('iterative_cpu_evaluations',{}).get('fit',{}).get('native_evaluation')
        validation=record.get('iterative_cpu_evaluations',{}).get('validation',{}).get('native_evaluation')
    if fit is None or validation is None:return None
    trajectories=[t for t in record.get('trajectories',{}).values()
                  if t.get('learned_proposals_per_pressure_solve',1)==policy]
    config=record['method']['configuration']
    return {'trial':record['trial'],'source_commit':record['method']['source_commit'],
        'training_updates':record['cuda_optimizer_updates'],
        'analytical_exposures':record.get('analytical_training_feedback',{}).get('additional_examples',0),
        'proposals_per_solve':policy,'network_passes_per_solve':policy*config['correction']['steps'],
        'fitted_relative_residual':fit['relative_residual']['median'],'fitted_qualified':fit['float32_qualified'],
        'validation_median_relative_residual':validation['relative_residual']['median'],
        'validation_max_relative_residual':validation['relative_residual']['maximum'],
        'validation_qualified':validation['float32_qualified'],'validation_cases':validation['cases'],
        'trajectory_runs':len(trajectories),
        'trajectory_ai_steps':sum(t['result']['ai_steps'] for t in trajectories),
        'trajectory_qualified_ai_steps':sum(t['result']['qualified_ai_steps'] for t in trajectories),
        'trajectory_hard_stops':sum(t['result']['hard_stop'] for t in trajectories),
        'weights_sha256':record['weights_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=LAB/'docs/plans/water-stability-learning-v1/reports')
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();records=[];stages=[];rows=[];total_updates=0
    baseline=json.loads(args.baseline.read_text());baseline_row=row(baseline)
    if baseline_row:
        baseline_row['trial']='retained-'+baseline_row['trial'];rows.append(baseline_row)
    for path in sorted(args.reports.glob('*.json')):
        value=json.loads(path.read_text())
        if value.get('schema') not in ('rheon-residual-experiment-record-v1','rheon-residual-experiment-record-v2'):continue
        records.append({'trial':value['trial'],'report':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        total_updates+=value['cuda_optimizer_updates']
        stages.extend({'trial':value['trial'],**{k:s[k] for k in ('name','run_id','state','custody_phase','catalog_id')}} for s in value['managed_stages'])
        for policy in (1,8):
            candidate=row(value,policy)
            if candidate:rows.append(candidate)
    assert len({s['run_id'] for s in stages})==len(stages)
    assert all(s['state']=='completed' and s['custody_phase']=='drained' for s in stages)
    def qualified(r):
        return bool(r['fitted_qualified']==1 and r['validation_qualified']==r['validation_cases']
            and r['validation_cases'] and r['trajectory_runs']>=3 and not r['trajectory_hard_stops']
            and r['trajectory_ai_steps'] and r['trajectory_qualified_ai_steps']==r['trajectory_ai_steps'])
    summary={'schema':'rheon-stability-learning-summary-v1','records':records,'managed_stages':stages,
        'cuda_optimizer_updates':total_updates,'comparison':rows,
        'baseline':{'path':str(args.baseline),'sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            'scope':'retained matched control; its Runs/updates are excluded from this tranche totals'},
        'single_proposal_development_qualified':any(qualified(r) for r in rows if r['proposals_per_solve']==1),
        'repeated_policy_development_qualified':any(qualified(r) for r in rows if r['proposals_per_solve']>1),
        'deployment_qualified':False,'untouched_test_used':False,'physical_gates_changed':False,
        'speedup_claim':False,'scope':'one recorded training frame plus analytical operator patterns; development qualification only'}
    (args.reports/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);(args.reports/'comparison.csv').write_text(stream.getvalue())
    print(json.dumps({'records':len(records),'managed_stages':len(stages),'cuda_updates':total_updates,
        'repeated_policy_development_qualified':summary['repeated_policy_development_qualified']}),flush=True)


if __name__=='__main__':main()
