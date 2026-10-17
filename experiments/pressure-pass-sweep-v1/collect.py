"""Project immutable CPU receipts into the dashboard; preserve raw measurements."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
from material import read,sha


def compact(value):
    return float(format(value,'.8g'))if isinstance(value,float)else value


def collect(material_path,destination):
    material=read(material_path);root=Path(material['root']);run=material_path.parent
    models=[{k:m[k]for k in ['id','name','tags','weights_sha256','original_worst_error','original_inference_median_ms']} for m in material['models']]
    points=[];trajectories=[];sources={};worker_rows=[]
    timing_execution=read(run/'isolated/execution.json')if(run/'isolated/execution.json').exists()else None
    isolated_ready=bool(timing_execution and timing_execution['status']=='completed')
    def source(path):
        name=os.path.relpath(path,root/'docs/plans');sources[name]=sha(path);return name
    for model in models:
        folder=run/model['id'];snapshot=folder/'snapshots.json'
        if snapshot.exists():
            result=read(snapshot)
            isolated=read(run/'isolated'/model['id']/'snapshots.json')if isolated_ready else None
            for n in material['config']['passes']:
                rows=[r for r in result['rows']if r['passes']==n]
                if len(rows)!=128:raise ValueError('Incomplete static population '+model['id'])
                errors=[r['relative_residual']for r in rows];times=[r['inference_s']for r in rows]
                parallel_median=statistics.median(times)*1000
                timing_source=source(snapshot)
                if isolated:
                    timed=[r for r in isolated['rows']if r['passes']==n]
                    if len(timed)!=128 or [(r['trajectory_id'],r['native_frame'],r['relative_residual'])for r in timed]!=[(r['trajectory_id'],r['native_frame'],r['relative_residual'])for r in rows]:
                        raise ValueError('Isolated timing changed input identities or numerical outputs')
                    times=[r['inference_s']for r in timed]
                    timing_source=source(run/'isolated'/model['id']/'snapshots.json')
                points.append(dict(model=model['id'],passes=n,worst_error=max(errors),median_error=statistics.median(errors),
                    inference_median_ms=statistics.median(times)*1000,inference_best_ms=min(times)*1000,
                    inference_worst_ms=max(times)*1000,cases=len(rows),equation_passes=sum(e<=1e-6 for e in errors),
                    mean_actual_calls=statistics.mean(r['network_calls']for r in rows),source=source(snapshot),
                    timing_source=timing_source,concurrent_inference_median_ms=parallel_median))
        for n in material['config']['passes']:
            path=folder/f'passes-{n:02d}/trajectory.json'
            if not path.exists():continue
            trace=read(path)
            # Display rounding only; full precision remains in the linked raw receipts.
            trace['points']=[[p[0],compact(p[1]),compact(p[2]),compact(p[5][0])if p[5]else None,
                              compact(p[5][1])if p[5]else None]for p in trace['points']]
            trace['source']=source(path)
            trajectories.append(trace)
        receipt=folder/'receipt.json'
        if receipt.exists():
            value=read(receipt);worker_rows.append({k:value[k]for k in ['id','status','elapsed_s','peak_rss_bytes','peak_child_rss_bytes','cpu']});source(receipt)
    expected=len(models)*len(material['config']['passes'])
    execution=read(run/'execution.json')if(run/'execution.json').exists()else None
    complete=bool(execution and execution['status']=='completed'and len(points)==expected and len(trajectories)==expected and all(w['status']=='passed'for w in worker_rows)and isolated_ready)
    result=dict(schema='rheon-inference-pass-sweep-chart-v1',status='completed'if complete else'running',
        updated_utc=datetime.now(timezone.utc).isoformat(),models=models,passes=material['config']['passes'],
        points=points,trajectories=trajectories,scene=material['config']['scene'],
        expected_combinations=expected,validation_cases=128,
        selection_chart_sha256=material['chart_sha256'],material_source=source(material_path),
        point_columns=['physical_step','simulation_time_s','pressure_relative_l2','absolute_error_rms_pa_estimate','reference_rms_pa_estimate'],
        timing_scope=('Fresh CPU fixed-pass recurrence plus final equation-error measurement. One checkpoint at a time on CPU 23, one Torch thread; warm-up excluded. Median over all 128 common validation inputs with one measured prefix per case. Parallel measurements are retained separately; timing excludes simulator and JSON IPC.'if isolated_ready else 'Provisional parallel-campaign timing: one Torch thread per checkpoint; three measured prefixes per case after warm-up. Concurrent workers share machine resources. Isolated latency measurement is pending.'),
        timing_mode='isolated'if isolated_ready else'concurrent',timing_execution=timing_execution,
        drift_scope='One independently evolving AI/reference pair per model and fixed pass count, after 15 native warm-up steps. Rejected candidates never advance history; no native fallback or refinement. Pa values are the bounded estimates used by the original drift plot. Native equation/divergence/flux/CFL checks remain enforced, but the deployed adaptive pressure-bound certificate is not applied; completion here is not deployed-policy qualification.',
        workers=worker_rows,execution={k:execution[k]for k in ['status','elapsed_s','maximum_workers','all_owned_children_reaped','cpu_pool']if k in execution}if execution else None,sources=sources)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'chart-data.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],static_points=len(points),trajectories=len(trajectories),expected=expected)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--material',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();collect(args.material.resolve(),args.output.resolve())
