"""Derive the final experiment index without changing immutable trial records."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--timings',type=Path,nargs='+',required=True)
    args=p.parse_args();reports=args.reports;records=[];rows=[];stages=[];updates=0
    baseline=read(reports/'latest-fine-development.json');baseline_long=read(reports/'latest-long-window.json')
    rows.append({'trial':'retained-single-grid-reference','cuda_updates_this_tranche':0,
        'train_simulations':1,'train_states':1,'development_qualified':baseline['native']['float32_qualified'],
        'development_cases':baseline['native']['cases'],'development_mean_evaluations':baseline['network_evaluations']['mean'],
        'development_max_evaluations':baseline['network_evaluations']['maximum'],
        'development_worst_relative_residual':baseline['native']['relative_residual']['maximum'],
        'long_trajectory_qualified_ai_steps':baseline_long['qualified_ai_steps'],'long_trajectory_ai_steps':baseline_long['ai_steps'],'weights_sha256':baseline['weights_sha256']})
    for name in ('trial-multigrid-single-pass.json','trial-multigrid-two-pass.json','trial-cohort16-two-pass.json'):
        path=reports/name;value=read(path);records.append({'path':name,'sha256':digest(path),'trial':value['trial']})
        stages.extend(value['managed_stages']);updates+=value['cuda_optimizer_updates'];dev=value['adaptive_development']
        long_command=next((c['argv'] for c in value['cpu_pipeline']['commands'] if c['name']=='long'),None)
        long_path=Path(long_command[long_command.index('--output')+1])/'report.json' if long_command else None
        trajectory=[r for r in read(long_path)['records'] if r['model']!='native'] if long_path else []
        rows.append({'trial':value['trial'],'cuda_updates_this_tranche':value['cuda_optimizer_updates'],
            'train_simulations':value['training_exposure']['real_simulations'],'train_states':value['training_exposure']['real_states'],
            'development_qualified':dev['native']['float32_qualified'],'development_cases':dev['native']['cases'],
            'development_mean_evaluations':dev['network_evaluations']['mean'],'development_max_evaluations':dev['network_evaluations']['maximum'],
            'development_worst_relative_residual':dev['native']['relative_residual']['maximum'],
            'long_trajectory_qualified_ai_steps':sum(r['result']['qualified_ai_steps'] for r in trajectory) if trajectory else None,
            'long_trajectory_ai_steps':sum(r['result']['ai_steps'] for r in trajectory) if trajectory else None,'weights_sha256':value['weights_sha256']})
    for name in ('managed-smoke.json','cohort-managed-smoke.json'):
        value=read(reports/name);stages.extend(value['stages']);updates+=value['optimizer_updates']
    if len({s['run_id'] for s in stages})!=len(stages) or not all(s['state']=='completed' and s['custody_phase']=='drained' for s in stages):
        raise ValueError('unique completed/drained managed stage inventory required')
    timing=[]
    for path in args.timings:
        value=read(path);items=[]
        for row in value['aggregate']:
            items.append({**row,'ai_to_native_ratio':row['ai_median_seconds']/row['native_median_seconds'] if row['native_median_seconds'] else None})
        timing.append({'path':str(path),'sha256':digest(path),'threads':value['threads_per_worker'],'scope':value['scope'],
            'all_physical_qualified':all(r['qualified'] for r in items),'aggregate':items,
            'native_children':len(value['records']),'all_native_children_reaped':all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in value['records'])})
    result={'schema':'rheon-multigrid-experiment-summary-v1','records':records,'comparison':rows,
        'managed_stages':stages,'managed_stage_count':len(stages),'cuda_optimizer_updates':updates,'timings':timing,
        'precision':'float32 learned weights/output; existing float64 physical arithmetic unchanged',
        'model_equation_target':1e-6,'native_equation_target':1e-8,'maximum_inference_network_evaluations':32,
        'fixed_eight_accuracy':read(reports/'fixed-eight.json'),'reserved_final_population_used':False,'deployment_changed':False,'scope':'finite development experiments; no unconditional convergence or arbitrary-grid accuracy claim'}
    (reports/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (reports/'comparison.csv').write_text(stream.getvalue())
    print(json.dumps({'cuda_optimizer_updates':updates,'managed_stages':len(stages),'comparisons':len(rows),'timing_studies':len(timing)}),flush=True)


if __name__=='__main__':main()
