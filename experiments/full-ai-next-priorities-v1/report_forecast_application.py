"""Publish complete-interval application costs without hiding failed controls."""
import argparse
import csv
import math
from pathlib import Path
import statistics
from study_support import ROOT,HERE,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh application report directory required')
    summary=read(args.campaign/'summary.json');execution=read(args.campaign/'execution.json')
    protocol=read(args.protocol);registry=read(args.registry)
    if summary['status']!='completed'or execution['status']!='completed'or len(execution['completed'])!=35:
        raise ValueError('All35application responses required')
    if not all(w['reaped']and w['returncode']==0 for w in execution['workers']):raise ValueError('All application workers must reap')
    if summary['timed_episodes']!=560 or summary['registry_sha256']!=sha(args.registry)or summary['protocol_sha256']!=sha(args.protocol):
        raise ValueError('Frozen application scope changed')
    if summary['source_sha256']!=sha(HERE/'benchmark_forecast_application.py'):raise ValueError('Actual application source changed')
    methods={m['identifier']:m for m in registry['methods']};groups={};sources=[args.campaign/'summary.json',args.campaign/'execution.json',args.protocol,args.registry]
    preparation=[];raw_episodes=0
    for identifier in methods:
        ready_path=args.campaign/(identifier+'-ready.json');ready=read(ready_path);sources.append(ready_path)
        if ready['kind']!='ready'or ready['cases']!=16 or ready['threads']!=1 or ready['affinity']!=[23]or ready['cuda_initialized']:
            raise ValueError('Actual application setup/CPU context differs')
        preparation.append(dict(identifier=identifier,model_setup_after_imports_s=ready['model_setup_s'],
            full_population_warmup_s=ready['warmup_s'],worker_launch_to_ready_s=ready['worker_launch_to_ready_s'],
            peak_worker_rss_bytes=ready['peak_rss_bytes']))
    for response in execution['completed']:
        path=ROOT/response['path']
        if sha(path)!=response['sha256']:raise ValueError('Application response changed')
        value=read(path);sources.append(path);identifier=value['identifier']
        if value['threads']!=1 or value['affinity']!=[23]or value['cuda_initialized']or len(value['rows'])!=16:
            raise ValueError('Declared serial CPU population differs')
        for row in value['rows']:
            raw_path=ROOT/row['receipt']
            if sha(raw_path)!=row['receipt_sha256']:raise ValueError('Raw application episode changed')
            raw=read(raw_path);sources.append(raw_path);raw_episodes+=1
            if raw['warmup']or not raw['offline_references_excluded_from_episode']:
                raise ValueError('Warmup/reference work cannot establish application timing')
            if not all(c['reaped']and c['returncode']==0 and c['graceful_protocol_complete']for c in raw['native_processes']):
                raise ValueError('Every application/native process must complete and reap')
            kind=methods[identifier]['kind'];diagnostics=raw['diagnostics']
            if kind=='forecast':
                if raw['native_numerical_steps_executed']!=0 or raw['fallback']or not raw['native_final_state_bytes_exact']:
                    raise ValueError('Numerical correction or altered native field entered a candidate episode')
                if len(diagnostics['rows'])!=raw['physical_steps']:raise ValueError('Every consumed forecast state requires offline metrics')
                k=methods[identifier]['model']['horizons']
                if raw['completed_physical_interval']and raw['generated_states']!=raw['physical_steps']+raw['event_discarded_states']+raw['endpoint_discarded_states']:
                    raise ValueError('Uncounted generated/discarded forecast work')
                if row['scenario']=='forcing'and raw['event_discarded_states']!=k-1:
                    raise ValueError('Every stale pre-event state must be invalidated')
                model_generated_ms=1000*raw['prediction_seconds']/raw['generated_states']if raw['completed_physical_interval']else None
                model_accepted_ms=1000*raw['prediction_seconds']/raw['physical_steps']if raw['completed_physical_interval']else None
                error=diagnostics['worst'];accuracy=diagnostics['accuracy_pass'];blocks=raw['full_block_seconds']
            else:
                if not raw['native_acceptance_gates_unchanged']or raw['fallback']:raise ValueError('Native control gates/fallback policy changed')
                model_generated_ms=model_accepted_ms=None;blocks=[]
                error=diagnostics['worst_at_diagnostic_checkpoints'];accuracy=diagnostics['diagnostic_accuracy_pass']
            key=(identifier,row['scenario'],'small'if max(row['counts_xyz'])<=16 else'32cubed')
            groups.setdefault(key,[]).append(dict(repeat=value['repeat'],complete=raw['completed_physical_interval'],
                application_s_per_physical_s=raw['application_seconds_per_physical_second'],
                application_attempt_s=raw['application_seconds'],native_setup_s=raw['native_setup_seconds'],
                model_generated_ms=model_generated_ms,model_accepted_ms=model_accepted_ms,
                model_blocks_ms=[v*1000 for v in blocks],error=error,accuracy_pass=accuracy,
                physical_steps=raw['physical_steps'],requested_steps=raw['requested_steps'],
                event_discarded_states=raw['event_discarded_states'],endpoint_discarded_states=raw['endpoint_discarded_states'],
                peak_worker_rss_bytes=raw['peak_worker_rss_including_deferred_validation_bytes'],
                max_native_child_rss_bytes=raw['max_native_child_rss_so_far_bytes']))
    if raw_episodes!=560:raise ValueError('All560timed application episodes required')
    def percentile95(values):return sorted(values)[math.ceil(.95*len(values))-1]
    def balanced(values,key):
        return statistics.median(statistics.median(v[key]for v in values if v['repeat']==r)for r in range(5))
    rows=[]
    for (identifier,scenario,regime),values in sorted(groups.items()):
        expected=30 if regime=='small'else 10
        if len(values)!=expected or {v['repeat']for v in values}!={0,1,2,3,4}:raise ValueError('Complete balanced application group required')
        complete=all(v['complete']for v in values);kind=methods[identifier]['kind'];all_errors=complete and all(v['error']is not None for v in values)
        blocks=[x for v in values for x in v['model_blocks_ms']]
        row=dict(identifier=identifier,kind=kind,scenario=scenario,regime=regime,episodes=len(values),
            completed_episodes=sum(v['complete']for v in values),completed_declared_population=complete,
            median_application_s_per_physical_s=balanced(values,'application_s_per_physical_s')if complete else None,
            p95_application_s_per_physical_s=percentile95([v['application_s_per_physical_s']for v in values])if complete else None,
            maximum_application_s_per_physical_s=max(v['application_s_per_physical_s']for v in values)if complete else None,
            median_attempted_application_s=balanced(values,'application_attempt_s'),
            median_native_setup_s=balanced(values,'native_setup_s'),
            median_model_ms_per_generated_state=balanced(values,'model_generated_ms')if complete and kind=='forecast'else None,
            median_model_ms_per_accepted_state=balanced(values,'model_accepted_ms')if complete and kind=='forecast'else None,
            p95_model_full_block_ms=percentile95(blocks)if blocks else None,maximum_model_full_block_ms=max(blocks)if blocks else None,
            worst_error=max(max(v['error'].values())for v in values)if all_errors else None,
            accuracy_pass=complete and all(v['accuracy_pass']for v in values),
            accuracy_scope='Every consumed forecast state'if kind=='forecast'else'Native diagnostic checkpoints only',
            accepted_steps_min=min(v['physical_steps']for v in values),requested_steps=values[0]['requested_steps'],
            event_discarded_states_per_episode=sorted({v['event_discarded_states']for v in values}),
            endpoint_discarded_states_per_episode=sorted({v['endpoint_discarded_states']for v in values}),
            peak_worker_rss_including_deferred_validation_bytes=max(v['peak_worker_rss_bytes']for v in values),
            maximum_native_child_rss_bytes=max(v['max_native_child_rss_bytes']for v in values))
        prep=next(p for p in preparation if p['identifier']==identifier)
        row['cold_component_amortization_s_per_physical_s']=(row['median_application_s_per_physical_s']+
            (prep['model_setup_after_imports_s']+row['median_native_setup_s'])/(row['requested_steps']/60))if complete else None
        row['cold_amortization_scope']='Arithmetic sum of measured setup and warm episode components; not a direct cold product launch'
        rows.append(row)
    args.output.mkdir(parents=True)
    write(args.output/'results.json',dict(status='completed',rows=rows,worker_preparation=preparation,
        source_evidence={str(p.relative_to(ROOT)):sha(p)for p in sources},reporter_sha256=sha(__file__),
        timed_episodes=560,repeats=5,serial_CPU=True,candidate_native_interface='Structural state consumer; no native physical-acceptance certification',
        numerical_native_controls='Original physical gates unchanged; failed intervals have no completed throughput',
        offline_reference_metrics_excluded_from_application_time=True,reserved_scenes_consumed=False,
        original_goal_complete=False))
    with (args.output/'results.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Full-state application and interaction measurements','',
        'Five balanced serial CPU repeats include complete neural generation, required checks, actual native state '
        'handoff, event reinitialization and discarded outputs. Offline reference comparisons follow the measured '
        'episode. Native numerical controls retain their original pressure gates; the neural state consumer is structural.','',
        '| Method | Scenario | Grid group | Complete episodes | App s / physical s | Model ms / generated state | Worst error | Accuracy scope |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    def number(value):return 'Unavailable'if value is None else f'{value:.6g}'
    for row in rows:
        lines.append(f'| {row["identifier"]} | {row["scenario"]} | {row["regime"]} | {row["completed_episodes"]}/{row["episodes"]} | '
            f'{number(row["median_application_s_per_physical_s"])} | {number(row["median_model_ms_per_generated_state"])} | '
            f'{number(row["worst_error"])} | {row["accuracy_scope"]} |')
    lines.extend(['','Loading/learned-cache preparation, whole-population warmup, native setup, RSS, latency tails '
        'and every discarded state remain separately reported. Cold amortization is an arithmetic sum of measured '
        'components and is labeled accordingly. A rejected or incomplete interval has no completed throughput or '
        'full-interval worst-error claim. Application costs are distinct from the original model-only2ms/state budget.','',
        '[Complete results and source evidence](results.json) · [CSV](results.csv)',''])
    (args.output/'results.md').write_text('\n'.join(lines))
    print('Published560application episodes with complete-interval accounting and separate controls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','protocol','registry','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('campaign','protocol','registry','output'):setattr(args,name,getattr(args,name).resolve())
    run(args)
