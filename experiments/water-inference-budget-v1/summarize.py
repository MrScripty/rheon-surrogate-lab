"""Derive a review index from sealed training records and CPU policy evidence."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


def load(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def row(record,proposals=1):
    native=record['native_evaluations'] if proposals==1 else {
        'fitted-evaluation':record['iterative_cpu_evaluations']['fit']['native_evaluation'],
        'validation-evaluation':record['iterative_cpu_evaluations']['validation']['native_evaluation']}
    fit=native['fitted-evaluation'];validation=native['validation-evaluation']
    trajectories=[v for v in record['trajectories'].values() if v.get('learned_proposals_per_pressure_solve',1)==proposals]
    passes=proposals*record['method']['configuration']['correction']['steps']
    return {'trial':record['trial']+(' · fixed repeated reference' if proposals>1 else ''),
        'mean_network_evaluations':passes,'min_network_evaluations':passes,'max_network_evaluations':passes,
        'cuda_updates':record['cuda_optimizer_updates'],'fitted_relative_residual':fit['relative_residual']['median'],
        'fitted_qualified':fit['float32_qualified'],
        'development_median_relative_residual':validation['relative_residual']['median'],
        'development_worst_relative_residual':validation['relative_residual']['maximum'],
        'development_qualified':validation['float32_qualified'],'development_cases':validation['cases'],
        'trajectory_qualified_ai_steps':sum(t['result']['qualified_ai_steps'] for t in trajectories),
        'trajectory_ai_steps':sum(t['result']['ai_steps'] for t in trajectories),
        'trajectory_hard_stops':sum(t['result']['hard_stop'] for t in trajectories),
        'weights_sha256':record['weights_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--controls',type=Path,nargs='+',required=True)
    args=p.parse_args();records=[];rows=[];stages=[];sources=[]
    for path in args.controls:
        v=load(path);rows.append(row(v));sources.append({'path':str(path),'sha256':digest(path),'role':'retained control; updates excluded from new totals'})
        if 'iterative_cpu_evaluations' in v:rows.append(row(v,8))
    for path in sorted(args.reports.glob('*.json')):
        v=load(path)
        if v.get('schema')!='rheon-residual-experiment-record-v2':continue
        records.append(v);rows.append(row(v));sources.append({'path':str(path),'sha256':digest(path),'role':'completed new trial'})
        stages.extend({'trial':v['trial'],**{k:s[k] for k in ('name','run_id','state','custody_phase','catalog_id')}} for s in v['managed_stages'])
    failure=load(args.reports/'single-pass-combined-failure.json')
    stages.append({'trial':failure['trial'],'name':'train','run_id':failure['run_id'],'state':failure['state'],'custody_phase':failure['custody_phase']})
    assert all(s['custody_phase']=='drained' for s in stages)
    assert len({s['run_id'] for s in stages})==len(stages)
    adaptive=load(args.reports/'adaptive-evidence.json');v=adaptive['validation'];passes=adaptive['case_network_passes']
    rows.append({'trial':'stability-modes · adaptive inference','mean_network_evaluations':statistics.mean(passes),
        'min_network_evaluations':min(passes),'max_network_evaluations':max(passes),'cuda_updates':0,
        'fitted_relative_residual':None,'fitted_qualified':None,'development_median_relative_residual':v['relative_residual']['median'],
        'development_worst_relative_residual':v['relative_residual']['maximum'],
        'development_qualified':v['float32_qualified'],'development_cases':v['cases'],
        'trajectory_qualified_ai_steps':adaptive['qualified_ai_steps'],'trajectory_ai_steps':adaptive['total_ai_steps'],
        'trajectory_hard_stops':sum(t['result']['hard_stop'] for t in adaptive['trajectories'].values()),
        'weights_sha256':adaptive['weights_sha256']})
    scaling=load(args.reports/'scaling.json')
    timing_rows=[]
    for scene in scaling['records']:
        if 'same_scene_as' in scene:continue
        for name in ('fixed','adaptive'):
            result=scene.get(name,{})
            timing_rows.append({'scene':scene['label'],'sweep':scene['sweep'],'axis_cells':scene['n'],
                'cells':scene['n']**3,'cell_size_m':scene['cell_size_m'],'policy':name,
                'qualified':result.get('qualified',False),'native_median_seconds':result.get('native_median_seconds'),
                'ai_median_seconds':result.get('ai_median_seconds'),
                'ai_to_native_time_ratio':result.get('ai_to_native_time_ratio'),
                'mean_network_passes':result.get('mean_network_passes'),
                'measured_steps':result.get('timing_samples',0)})
    total=sum(r['cuda_optimizer_updates'] for r in records)
    summary={'schema':'rheon-inference-budget-summary-v1','sources':sources,'comparison':rows,
        'managed_stages':stages,'successful_cuda_optimizer_updates':total,
        'failed_run_completed_updates_lower_bound':failure['last_reported_completed_updates'],
        'all_attempts_cuda_updates_lower_bound':total+failure['last_reported_completed_updates'],
        'failed_run_exact_update_count_available':False,'failed_training':failure['run_id'],
        'scaling':scaling,'single_pass_model_qualified':any(r['max_network_evaluations']==1 and r['fitted_qualified']==1 and r['development_qualified']==r['development_cases'] and r['trajectory_ai_steps']>0 and r['trajectory_qualified_ai_steps']==r['trajectory_ai_steps'] and not r['trajectory_hard_stops'] for r in rows),
        'deployment_changed':False,'untouched_test_used':False,'physical_gates_changed':False,
        'any_qualified_ai_faster_than_native':any(r['qualified'] and r['ai_to_native_time_ratio']<1 for r in timing_rows),
        'timing_scope':'whole independently evolving tracks; same scene inputs; existing distinct native/AI equation profiles; serial CPU core',
        'scope':'bounded development experiments; negative training/scaling outcomes retained; policy speed distinct from trained-model accuracy'}
    (args.reports/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (args.reports/'comparison.csv').write_text(stream.getvalue())
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(timing_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(timing_rows)
    (args.reports/'scaling.csv').write_text(stream.getvalue())
    print(json.dumps({'completed_training_trials':len(records),'managed_stages':len(stages),'successful_cuda_updates':total,'failed_run_updates_lower_bound':failure['last_reported_completed_updates']}))


if __name__=='__main__':main()
