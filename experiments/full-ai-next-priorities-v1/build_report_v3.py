"""Build separate scientific frontiers and a local, evidence-linked results report."""
import csv
import argparse
import html
import json
import math
import os
from pathlib import Path
import time
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def gather():
    sources={};points=[];accuracy=[]
    def evidence(path):
        sources[str(path.relative_to(ROOT))]=sha(path)
        return read(path)
    final=BASE/'pressure-final-timing-01/summary.json'
    if final.exists():
        timing=evidence(final)
        for row in timing['rows']:
            points.append(dict(track='A',identifier=row['checkpoint'],regime='historical_small',interval='one_pressure_request',
                median_ms=row['median_ms'],worst_error=row['worst_equation_relative_residual'],
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['median_ms']<=4 and row['worst_equation_relative_residual']<=7e-4,
                timing_source=str(final.relative_to(ROOT))))
    else:
        baseline=BASE/'pressure-baseline-03/summary.json'
        if baseline.exists():
            for row in evidence(baseline)['rows']:
                if row['variant']!='original':continue
                points.append(dict(track='A',identifier=row['checkpoint'],regime='historical_small',interval='one_pressure_request',
                    median_ms=row['median_ms'],worst_error=row['worst_equation_relative_residual'],target_met=False,
                    min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                    timing_source=str(baseline.relative_to(ROOT)),qualification='Baseline repeated timings; new-weight timing pending'))
    for path in sorted((BASE/'pressure-evaluation-01').glob('*/checkpoint-*.json')):
        value=evidence(path);accuracy.append(dict(track='A',identifier=value['identifier']+'@'+str(value['update']),
            worst_error=value['worst_error'],median_error=value['median_error'],p95_error=value['p95_error'],
            weights_sha256=value['weights_sha256'],source=str(path.relative_to(ROOT))))
    for path in sorted((BASE/'pressure-architecture-evaluation-01').glob('*/checkpoint-*.json')):
        value=evidence(path);accuracy.append(dict(track='A',identifier=value['identifier']+'@'+str(value['update']),
            worst_error=value['worst_error'],median_error=value['median_error'],p95_error=value['p95_error'],
            weights_sha256=value['weights_sha256'],source=str(path.relative_to(ROOT))))
    compiled=BASE/'pressure-compiled-timing-01/summary.json'
    if compiled.exists():
        for row in evidence(compiled)['rows']:
            points.append(dict(track='A',identifier=row['checkpoint']+'/'+row['variant'],regime='historical_small',interval='one_pressure_request',
                median_ms=row['median_ms'],worst_error=row['worst_equation_relative_residual'],
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['median_ms']<=4 and row['worst_equation_relative_residual']<=7e-4,
                timing_source=str(compiled.relative_to(ROOT)),
                qualification='Learned full AI model; compiled path separately qualified. Plain calibrated heads regress at128cubed; routed version retains original larger-grid head.'))
    for path in sorted((BASE/'pressure-head-calibration-evaluation-01').glob('*/summary.json')):
        for row in evidence(path)['rows']:
            accuracy.append(dict(track='A',identifier=row['identifier'],regime='historical_small',
                worst_error=row['worst_error'],median_error=row['median_error'],p95_error=row['p95_error'],
                weights_sha256=row['weights_sha256'],source=str(path.relative_to(ROOT))))
    fresh_timing=BASE/'pressure-fresh-timing-01/summary.json'
    if fresh_timing.exists():
        for row in evidence(fresh_timing)['rows']:
            path=BASE/'pressure-fresh-evaluation-01'/row['identifier']/'summary.json'
            evaluation=evidence(path)
            errors=next(r for r in evaluation['prepared_requests']if r['population']==row['population'])
            points.append(dict(track='A',identifier=row['identifier'],regime='fresh_'+row['population'],interval='one_pressure_request',
                median_ms=row['median_ms'],worst_error=errors['worst_equation_error'],
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['population']=='small'and row['median_ms']<=4 and errors['worst_equation_error']<=7e-4,
                timing_source=str(fresh_timing.relative_to(ROOT)),accuracy_source=str(path.relative_to(ROOT)),
                native_accepted_complete_episodes=evaluation['native_accepted_complete_episodes'],
                qualification='Independent Track A reservation; timing4fixed steps/scene, accuracyall64steps/scene; native gates unchanged, no fallback'))
            accuracy.append(dict(track='A',identifier=row['identifier'],regime='fresh_'+row['population'],
                worst_error=errors['worst_equation_error'],median_error=errors['median_equation_error'],p95_error=errors['p95_equation_error'],
                source=str(path.relative_to(ROOT))))
    whole_timing=BASE/'pressure-whole-timing-01/summary.json'
    if whole_timing.exists():
        for row in evidence(whole_timing)['rows']:
            qualification_path=BASE/'pressure-whole-qualification-01'/(row['checkpoint']+'.json')
            qualification=evidence(qualification_path)
            population={'historical_small':'consumed_historical','fresh_small':'consumed_fresh_small',
                        'fresh_32cubed_extension':'consumed_fresh_32cubed_extension'}[row['population']]
            errors=[case for case in qualification['rows']if case['population']==population]
            worst=max(case['compiled_equation_error']for case in errors)
            native_path=BASE/'pressure-fresh-evaluation-01'/row['checkpoint']/'summary.json'
            native=evidence(native_path)
            points.append(dict(track='A',identifier=row['checkpoint']+'/'+row['variant'],regime=row['population'],interval='one_pressure_request',
                median_ms=row['median_ms'],worst_error=worst,min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['population']!='fresh_32cubed_extension'and row['median_ms']<=4 and worst<=7e-4,
                timing_source=str(whole_timing.relative_to(ROOT)),accuracy_source=str(qualification_path.relative_to(ROOT)),
                native_accepted_complete_episodes=native['native_accepted_complete_episodes'],
                qualification='Frozen weights; complete graph or original eager execution; all1152consumed outputs bitwise equivalent. No new independent accuracy claim.'))
    confirmation_timing=BASE/'pressure-confirmation-timing-01/summary.json'
    if confirmation_timing.exists():
        for row in evidence(confirmation_timing)['rows']:
            path=BASE/'pressure-confirmation-evaluation-01'/row['identifier']/'summary.json'
            evaluation=evidence(path)
            errors=next(value for value in evaluation['prepared_requests']if value['population']==row['population'])
            points.append(dict(track='A',identifier=row['identifier']+'/whole_graph',regime='confirmation_'+row['population'],
                interval='one_pressure_request',median_ms=row['median_ms'],worst_error=errors['worst_equation_error'],
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['population']=='small'and row['median_ms']<=4 and errors['worst_equation_error']<=7e-4,
                timing_source=str(confirmation_timing.relative_to(ROOT)),accuracy_source=str(path.relative_to(ROOT)),
                native_accepted_complete_episodes=evaluation['native_accepted_complete_episodes'],
                qualification='New independent reservation after freezing compiled execution and selected weights; bitwise eager equivalence checked on every request'))
            accuracy.append(dict(track='A',identifier=row['identifier'],regime='confirmation_'+row['population'],
                worst_error=errors['worst_equation_error'],median_error=errors['median_equation_error'],p95_error=errors['p95_equation_error'],
                source=str(path.relative_to(ROOT))))
    tail_timing=BASE/'pressure-tail8-timing-01/summary.json'
    tail_results=DOCS/'tail-head-results.json'
    tail_followup=evidence(tail_results)if tail_results.exists()else None
    if tail_timing.exists():
        for row in evidence(tail_timing)['rows']:
            if row['checkpoint']=='a-tail8':
                path=BASE/'pressure-tail8-qualification-01/a-tail8.json'
                qualified=evidence(path)
                population={'historical_small':'consumed_historical','fresh_small':'consumed_confirmation_small',
                    'fresh_32cubed_extension':'consumed_confirmation_32cubed_extension'}[row['population']]
                cases=[case for case in qualified['rows']if case['population']==population]
                key='compiled_equation_error'if row['variant']=='whole_graph'else'original_equation_error'
                worst=max(case[key]for case in cases)
            elif row['population']=='historical_small':
                path=BASE/'pressure-whole-qualification-01'/f"{row['checkpoint']}.json"
                qualified=evidence(path)
                cases=[case for case in qualified['rows']if case['population']=='consumed_historical']
                worst=max(case['original_equation_error']for case in cases)
            else:
                path=BASE/'pressure-confirmation-evaluation-01'/row['checkpoint']/'summary.json'
                evaluated=evidence(path)
                group='small'if row['population']=='fresh_small'else'32cubed_extension'
                matching=[case for case in evaluated['prepared_requests']if case['population']==group]
                if not matching:
                    matching=[case for case in evaluated['prepared_requests']if case['population']!='small']
                if len(matching)!=1:raise ValueError('Ambiguous tail comparison population')
                worst=matching[0]['worst_equation_error']
            regime={'historical_small':'tail_development_historical_small','fresh_small':'tail_development_confirmation_small',
                'fresh_32cubed_extension':'tail_development_confirmation_32cubed_extension'}[row['population']]
            points.append(dict(track='A',identifier=row['checkpoint']+'/'+row['variant'],regime=regime,
                interval='one_pressure_request',median_ms=row['median_ms'],worst_error=worst,
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['population']!='fresh_32cubed_extension'and row['median_ms']<=4 and worst<=7e-4,
                timing_source=str(tail_timing.relative_to(ROOT)),accuracy_source=str(path.relative_to(ROOT)),
                qualification='Paired timing on consumed development populations; all128/768/256accuracy requests. Tail compilation numerically qualified, not bitwise equivalent. No new independent accuracy or native acceptance claim.'))
    precise_path=DOCS/'precise-routed-pressure-results.json'
    precise_followup=evidence(precise_path)if precise_path.exists()else None
    precise_timing=BASE/'precise-routed-pressure-timing-01/summary.json'
    if precise_timing.exists():
        for row in evidence(precise_timing)['rows']:
            path=BASE/('precise-routed-pressure-qualification-01'if row['variant']=='materialized_graph'else'pressure-tail8-qualification-01')/'a-tail8.json'
            qualified=evidence(path)
            population={'historical_small':'consumed_historical','fresh_small':'consumed_confirmation_small',
                'fresh_32cubed_extension':'consumed_confirmation_32cubed_extension'}[row['population']]
            cases=[case for case in qualified['rows']if case['population']==population]
            worst=max(case['compiled_equation_error']for case in cases)
            regime={'historical_small':'precise_development_historical_small','fresh_small':'precise_development_confirmation_small',
                'fresh_32cubed_extension':'precise_development_confirmation_32cubed_extension'}[row['population']]
            points.append(dict(track='A',identifier='a-tail8/'+('precise_routed'if row['variant']=='materialized_graph'else'whole_graph'),
                regime=regime,interval='one_pressure_request',median_ms=row['median_ms'],worst_error=worst,
                min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                target_met=row['population']!='fresh_32cubed_extension'and row['median_ms']<=4 and worst<=7e-4,
                timing_source=str(precise_timing.relative_to(ROOT)),accuracy_source=str(path.relative_to(ROOT)),
                qualification='Qualified learned execution, unchanged weights and gates; consumed development comparison only. Geometry-specific dense learned map at<=2048voxels, compiled CNN above. Cold setup/memory and per-grid limits separate; new independent confirmation outstanding.'))
    independent_precise_path=DOCS/'pressure-confirmation-v2/results.json'
    independent_precise=evidence(independent_precise_path)if independent_precise_path.exists()else None
    if independent_precise is not None:
        for model in independent_precise['rows']:
            path=BASE/'pressure-precise-confirmation-evaluation-01'/model['identifier']/'summary.json'
            evidence(path)
            for row in model['populations']:
                points.append(dict(track='A',identifier=model['identifier']+'/precise_routed',
                    regime='precise_confirmation_'+row['population'],interval='one_pressure_request',
                    median_ms=row['median_ms'],worst_error=row['worst_equation_error'],
                    min_repeat_ms=row['min_repeat_median_ms'],max_repeat_ms=row['max_repeat_median_ms'],
                    target_met=row['target_met'],native_accepted_complete_episodes=model['native_complete_episodes'],
                    timing_source=str((BASE/'pressure-precise-confirmation-timing-01/summary.json').relative_to(ROOT)),
                    accuracy_source=str(path.relative_to(ROOT)),
                    qualification='New independent scenes after model, execution and primary freeze; same optimization for C107 control. Numerical equivalence and native gates retained. Heterogeneous warm median; cold setup and per-grid costs separate.'))
    temporal_timing=BASE/'temporal-timing-02/summary.json'
    temporal_rows=evidence(temporal_timing)['rows']if temporal_timing.exists()else []
    for k in (1,2,3,4):
        path=BASE/f'temporal-development-evaluation-01/b{k}/summary.json'
        if not path.exists():continue
        value=evidence(path)
        for regime in ('small','32cubed'):
            scenes=[r for r in value['scenes']if ('small'if max(r['counts_xyz'])<=16 else'32cubed')==regime]
            timing=next((r for r in temporal_rows if r['horizons']==k and r['variant']=='full'and r['regime']==regime),None)
            for label,index in (('single_block',0),('rollout_12',1),('rollout_48',2),('rollout_240',3)):
                worst={field:max(s['summaries'][index]['worst'][field]for s in scenes)for field in ('pressure','equation','velocity','tracer')}
                complete=all(s['summaries'][index]['complete']for s in scenes)
                row=dict(track='B',identifier=f'K={k}',horizons=k,regime=regime,interval=label,worst_error=max(worst.values()),
                    field_worst=worst,all_finite_complete=complete,accuracy_pass=complete and all(s['summaries'][index]['all_accuracy_criteria_pass']for s in scenes),
                    accuracy_source=str(path.relative_to(ROOT)),weights_sha256=value['model']['weights_sha256'],
                    timing_pending=timing is None,target_met=False)
                accuracy.append(row)
                if timing is not None:
                    point=dict(row,median_ms=timing['median_ms_per_generated_step'],full_block_ms=timing['median_block_ms'],
                        min_repeat_ms=timing['min_repeat_median_ms']/k,max_repeat_ms=timing['max_repeat_median_ms']/k,
                        timing_source=str(temporal_timing.relative_to(ROOT)))
                    point['target_met']=regime=='small'and k>1 and row['accuracy_pass']and point['median_ms']<2
                    points.append(point)
    sharing_progress=[]
    for k in (2,3,4):
        for architecture in ('endpoint','reuse'):
            identifier=f'b{k}-{architecture}';pair=BASE/f'{identifier}-pair-01'
            path=BASE/'temporal-sharing-development-evaluation-01'/identifier/'summary.json'
            row=dict(identifier=identifier,fit_restore_complete=(pair/'receipt.json').exists(),development_complete=path.exists())
            if path.exists():
                value=evidence(path);scenes=[scene for scene in value['scenes']if max(scene['counts_xyz'])<=16]
                row['worst_error']=max(scene['summaries'][0]['worst'][field]for scene in scenes for field in ('pressure','equation','velocity','tracer'))
                row['accuracy_pass']=all(scene['summaries'][0]['all_accuracy_criteria_pass']for scene in scenes)
            sharing_progress.append(row)
    sharing_results_path=DOCS/'temporal-sharing-results.json'
    sharing_results=evidence(sharing_results_path)if sharing_results_path.exists()else None
    if sharing_results is not None:
        for name,digest in sharing_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Sharing evidence changed after comparison')
            sources[name]=digest
        sharing_timing_path=BASE/'temporal-sharing-timing-01/summary.json'
        if sha(sharing_timing_path)!=sharing_results['timing_sha256']:
            raise ValueError('Sharing timing changed after comparison')
        sharing_timing=evidence(sharing_timing_path)
        for value in sharing_results['rows']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            accuracy_path=BASE/'temporal-sharing-development-evaluation-01'/value['identifier']/'summary.json'
            row=dict(track='B',identifier=value['identifier'],horizons=value['horizons'],regime=value['regime'],interval=interval,
                worst_error=value['worst_error'],field_worst=value['worst'],accuracy_pass=value['accuracy_pass'],
                accuracy_source=str(accuracy_path.relative_to(ROOT)),timing_pending=False,target_met=value['combined_target_pass'])
            accuracy.append(row)
            measured=next(r for r in sharing_timing['rows']if r['identifier']==value['identifier']and r['variant']=='full'and r['regime']==value['regime'])
            points.append(dict(row,median_ms=value['median_ms_per_generated_step'],full_block_ms=value['median_full_block_ms'],
                min_repeat_ms=measured['min_repeat_median_ms']/value['horizons'],
                max_repeat_ms=measured['max_repeat_median_ms']/value['horizons'],
                p95_block_ms=measured['p95_block_ms'],maximum_block_ms=measured['maximum_block_ms'],
                timing_source=str(sharing_timing_path.relative_to(ROOT))))
    batches={}
    for name in ('temporal-training-batch-02','architecture-training-batch-01','temporal-development-evaluation-batch-01','post-training-evaluation-batch-02',
                 'head-replication-batch-01','routed-qualification-batch-01','pressure-reserved-batch-01','temporal-mse-batch-01','temporal-mse-report-batch-01',
                 'temporal-global-batch-01','temporal-global-report-batch-01','pressure-whole-execution-batch-01','temporal-pressure-history-batch-01',
                 'pressure-confirmation-batch-01','temporal-component-ceiling-batch-01','tail8-replication-batch-01','causal-request-training-batch-01',
                 'factored-request-training-batch-01','factored-request-training-batch-02','temporal-sharing-batch-01','temporal-sharing-postprocessing-batch-01','temporal-step-scaled-batch-01','temporal-increment-supervision-batch-01','temporal-curl-batch-01','temporal-pressure-expert-batch-01','temporal-pressure-expert-postprocessing-batch-01','temporal-decoder-batch-01','temporal-decoder-batch-02','temporal-coupled-batch-01','temporal-provisional-batch-01','temporal-provisional-timing-batch-01','temporal-provisional-timing-01','temporal-provisional-timing-batch-02','temporal-provisional-timing-02','coupled-CPU-stage-profile-batch-01','point-sampling-batch-01','point-sampling-evaluation-batch-02','temporal-displacement-batch-01','temporal-displacement-postprocessing-batch-01','displacement-component-diagnostic-batch-01','displacement-component-diagnostic-batch-02','displacement-CPU-stage-profile-batch-01','viscosity-calibration-batch-01','neural-viscosity-evaluation-batch-01','neural-viscosity-timing-batch-01','fast-preparation-timing-batch-01','fixed-viscosity-admission-batch-01','fixed-viscosity-training-batch-01','fixed-viscosity-postprocessing-batch-01','float32-map-evaluation-batch-01','float32-map-timing-batch-01','float32-map-publication-batch-01','fixed-viscosity-k2-precision-evaluation-batch-01','fixed-viscosity-k2-precision-timing-batch-01','fixed-viscosity-replication-admission-batch-01','fixed-viscosity-replication-training-batch-01','fixed-viscosity-replication-admission-batch-02','fixed-viscosity-replication-postprocessing-batch-01','fixed-viscosity-replication-execution-batch-01','fixed-viscosity-replication-timing-batch-01','temporal-final-confirmation-batch-01','temporal-final-reserved-evaluation-batch-01','temporal-final-reserved-timing-batch-01','application-domain-change-batch-01','application-domain-change-repair-batch-02'):
        path=BASE/name/'execution.json'
        if path.exists():
            value=evidence(path);batches[name]=dict(status=value['status'],active=(value.get('active')or{}).get('id'))
    followup_path=DOCS/'temporal-loss-followup-results.json'
    followup=evidence(followup_path)if followup_path.exists()else None
    global_path=DOCS/'temporal-global-results.json'
    global_followup=evidence(global_path)if global_path.exists()else None
    component_results=[]
    for name in ('causal-pressure-request-training-result.json','factored-pressure-request-results.json'):
        path=DOCS/name
        if path.exists():component_results.append(dict(path=name,result=evidence(path)))
    sharing_registration=DOCS/'temporal-sharing-registration-01.json'
    sharing=evidence(sharing_registration)if sharing_registration.exists()else None
    conditioning_path=DOCS/'temporal-step-scaled-registration-01.json'
    conditioning=evidence(conditioning_path)if conditioning_path.exists()else None
    conditioning_results_path=DOCS/'temporal-step-scaled-results.json'
    conditioning_results=evidence(conditioning_results_path)if conditioning_results_path.exists()else None
    if conditioning_results is not None:
        for name,digest in conditioning_results['sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Conditioning evidence changed')
            sources[name]=digest
        for value in conditioning_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            identifier='b2-'+value['architecture']+'-step-scaled'
            accuracy.append(dict(track='B',identifier=identifier,horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(value['step_scaled']['worst'].values()),field_worst=value['step_scaled']['worst'],
                accuracy_pass=value['step_scaled']['accuracy_pass'],timing_pending=True,target_met=False,
                accuracy_source=str(conditioning_results_path.relative_to(ROOT))))
    increment_registration_path=DOCS/'temporal-increment-supervision-registration-01.json'
    increment_registration=evidence(increment_registration_path)if increment_registration_path.exists()else None
    increment_results_path=DOCS/'temporal-increment-supervision-results.json'
    increment_results=evidence(increment_results_path)if increment_results_path.exists()else None
    if increment_results is not None:
        for name,digest in increment_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Increment-supervision evidence changed')
            sources[name]=digest
        for value in increment_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            for mode,suffix in (('full_state','mse-fullstate'),('increments','mse-increment')):
                arm=value['arms'][mode]
                accuracy.append(dict(track='B',identifier='b2-endpoint-'+suffix,horizons=2,regime=value['regime'],interval=interval,
                    worst_error=max(arm['worst'].values()),field_worst=arm['worst'],accuracy_pass=arm['accuracy_pass'],
                    timing_pending=True,target_met=False,accuracy_source=str(increment_results_path.relative_to(ROOT))))
    curl_registration_path=DOCS/'temporal-curl-registration-01.json'
    curl_registration=evidence(curl_registration_path)if curl_registration_path.exists()else None
    curl_results_path=DOCS/'temporal-curl-results.json'
    curl_results=evidence(curl_results_path)if curl_results_path.exists()else None
    if curl_results is not None:
        for name,digest in curl_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Curl comparison evidence changed')
            sources[name]=digest
        for value in curl_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            arm=value['curl_increment']
            accuracy.append(dict(track='B',identifier='b2-endpoint-curl-increment',horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(arm['worst'].values()),field_worst=arm['worst'],accuracy_pass=arm['accuracy_pass'],
                timing_pending=True,target_met=False,accuracy_source=str(curl_results_path.relative_to(ROOT))))
    expert_registration_path=DOCS/'temporal-pressure-expert-registration-01.json'
    expert_registration=evidence(expert_registration_path)if expert_registration_path.exists()else None
    expert_results_path=DOCS/'temporal-pressure-expert-results.json'
    expert_results=evidence(expert_results_path)if expert_results_path.exists()else None
    if expert_results is not None:
        for name,digest in expert_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Pressure expert evidence changed')
            sources[name]=digest
        for value in expert_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            arm=value['pressure_expert']
            accuracy.append(dict(track='B',identifier='b2-endpoint-pressure-expert',horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(arm['worst'].values()),field_worst=arm['worst'],accuracy_pass=arm['accuracy_pass'],
                timing_pending=True,target_met=False,accuracy_source=str(expert_results_path.relative_to(ROOT))))
    decoder_registration_path=DOCS/'temporal-decoder-registration-01.json'
    decoder_registration=evidence(decoder_registration_path)if decoder_registration_path.exists()else None
    decoder_results_path=DOCS/'temporal-decoder-results.json'
    decoder_results=evidence(decoder_results_path)if decoder_results_path.exists()else None
    if decoder_results is not None:
        for name,digest in decoder_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Combined decoder evidence changed')
            sources[name]=digest
        for value in decoder_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            arm=value['decoder']
            accuracy.append(dict(track='B',identifier='b2-endpoint-decoder',horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(arm['worst'].values()),field_worst=arm['worst'],accuracy_pass=arm['accuracy_pass'],
                timing_pending=True,target_met=False,accuracy_source=str(decoder_results_path.relative_to(ROOT))))
    coupled_registration_path=DOCS/'temporal-coupled-registration-01.json'
    coupled_registration=evidence(coupled_registration_path)if coupled_registration_path.exists()else None
    coupled_results_path=DOCS/'temporal-coupled-results.json'
    coupled_results=evidence(coupled_results_path)if coupled_results_path.exists()else None
    if coupled_results is not None:
        for name,digest in coupled_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Coupled dynamics evidence changed')
            sources[name]=digest
        for value in coupled_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            arm=value['coupled']
            accuracy.append(dict(track='B',identifier='b2-coupled',horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(arm['worst'].values())if arm['complete']else None,field_worst=arm['worst'],
                all_finite_complete=arm['complete'],worst_scope=arm['worst_scope'],
                observed_prefix_worst=arm['observed_prefix_worst'],accuracy_pass=arm['accuracy_pass'],
                timing_pending=True,target_met=False,accuracy_source=str(coupled_results_path.relative_to(ROOT))))
    coupled_figure_path=BASE/'temporal-coupled-rollout-figure-01.json'
    coupled_figure=evidence(coupled_figure_path)if coupled_figure_path.exists()else None
    if coupled_figure is not None:
        if coupled_results is None or coupled_figure['result_sha256']!=sha(coupled_results_path):raise ValueError('Coupled plot result binding changed')
        if coupled_figure['source_sha256']!=sha(HERE/'plot_coupled_rollout.py'):raise ValueError('Coupled plot source changed')
        for name,digest in {**coupled_figure['source_evidence'],**coupled_figure['artifacts']}.items():
            if sha(ROOT/name)!=digest:raise ValueError('Coupled rollout plot evidence changed')
            sources[name]=digest
    transport_path=DOCS/'temporal-coupled-transport-results.json'
    transport_diagnosis=evidence(transport_path)if transport_path.exists()else None
    if transport_diagnosis is not None:
        for name,digest in transport_diagnosis['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Offline transport diagnostic evidence changed')
            sources[name]=digest
    provisional_registration_path=DOCS/'temporal-provisional-registration-01.json'
    provisional_registration=evidence(provisional_registration_path)if provisional_registration_path.exists()else None
    provisional_results_path=DOCS/'temporal-provisional-results.json'
    provisional_results=evidence(provisional_results_path)if provisional_results_path.exists()else None
    if provisional_results is not None:
        for name,digest in provisional_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Matched continuation evidence changed')
            sources[name]=digest
        for value in provisional_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            for identifier,key in (('b2-provisional-control','control'),('b2-provisional-aux','auxiliary')):
                arm=value[key]
                accuracy.append(dict(track='B',identifier=identifier,horizons=2,regime=value['regime'],interval=interval,
                    worst_error=max(arm['worst'].values())if arm['complete']else None,field_worst=arm['worst'],
                    all_finite_complete=arm['complete'],worst_scope=arm['worst_scope'],observed_prefix_worst=arm['observed_prefix_worst'],
                    accuracy_pass=arm['accuracy_pass'],timing_pending=True,target_met=False,accuracy_source=str(provisional_results_path.relative_to(ROOT))))
    displacement_results_path=DOCS/'temporal-displacement-results.json'
    displacement_results=evidence(displacement_results_path)if displacement_results_path.exists()else None
    if displacement_results is not None:
        for name,digest in displacement_results['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Displacement accuracy evidence changed')
            sources[name]=digest
        for value in displacement_results['comparisons']:
            interval={'single_block':'single_block','own_history12':'rollout_12','own_history48':'rollout_48','own_history240':'rollout_240'}[value['interval']]
            arm=value['displacement']
            accuracy.append(dict(track='B',identifier='b2-displacement',horizons=2,regime=value['regime'],interval=interval,
                worst_error=max(arm['worst'].values())if arm['complete']else None,field_worst=arm['worst'],
                all_finite_complete=arm['complete'],worst_scope=arm['worst_scope'],observed_prefix_worst=arm['observed_prefix_worst'],
                accuracy_pass=arm['accuracy_pass'],timing_pending=True,target_met=False,accuracy_source=str(displacement_results_path.relative_to(ROOT))))
    tracer_controls={}
    for key,name in (('representation','tracer-representation-results.json'),('scalar_replay','tracer-reference-velocity-replay-results.json')):
        path=DOCS/name
        if path.exists():
            value=evidence(path)
            for relative,digest in value['source_evidence'].items():
                if sha(ROOT/relative)!=digest:raise ValueError('Offline tracer evidence changed')
                sources[relative]=digest
            tracer_controls[key]=value
    coupled_cost_path=DOCS/'temporal-provisional-cost-results.json'
    displacement_path=DOCS/'displacement-transport-results.json'
    displacement_controls=evidence(displacement_path)if displacement_path.exists()else None
    if displacement_controls is not None:
        for name,digest in displacement_controls['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Offline displacement evidence changed')
            sources[name]=digest
    point_path=DOCS/'point-sampling-results.json'
    point_component=evidence(point_path)if point_path.exists()else None
    if point_component is not None:
        for name,digest in point_component['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Learned point component evidence changed')
            sources[name]=digest
    displacement_qualification_path=DOCS/'temporal-displacement-qualification-results.json'
    displacement_qualification=evidence(displacement_qualification_path)if displacement_qualification_path.exists()else None
    if displacement_qualification is not None:
        for name,digest in displacement_qualification['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Actual displacement qualification evidence changed')
            sources[name]=digest
    displacement_components_path=DOCS/'displacement-component-results.json'
    displacement_components=evidence(displacement_components_path)if displacement_components_path.exists()else None
    if displacement_components is not None:
        for name,digest in displacement_components['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Offline displacement component evidence changed')
            sources[name]=digest
    coupled_cost=evidence(coupled_cost_path)if coupled_cost_path.exists()else None
    if coupled_cost is not None:
        for name,digest in coupled_cost['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Measured coupled cost evidence changed')
            sources[name]=digest
        for point in coupled_cost['points']:
            points.append(dict(point,source=str(coupled_cost_path.relative_to(ROOT))))
            for row in accuracy:
                if (row.get('identifier'),row.get('regime'),row.get('interval'))==(point['identifier'],point['regime'],point['interval']):
                    if row['worst_error']!=point['worst_error']:raise ValueError('Timed accuracy join changed the measured error')
                    row.update(timing_pending=False,target_met=point['target_met'],timing_source=str(coupled_cost_path.relative_to(ROOT)))
    displacement_cost_path=DOCS/'temporal-displacement-cost-results.json'
    displacement_cost=evidence(displacement_cost_path)if displacement_cost_path.exists()else None
    if displacement_cost is not None:
        if displacement_results is None:raise ValueError('Measured displacement costs require source-bound accuracy')
        for name,digest in displacement_cost['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Measured displacement cost evidence changed')
            sources[name]=digest
        for point in displacement_cost['points']:
            points.append(dict(point,source=str(displacement_cost_path.relative_to(ROOT))))
            for row in accuracy:
                if (row.get('identifier'),row.get('regime'),row.get('interval'))==(point['identifier'],point['regime'],point['interval']):
                    if row['worst_error']!=point['worst_error']:raise ValueError('Displacement timing join changed accuracy')
                    row.update(timing_pending=False,target_met=point['target_met'],timing_source=str(displacement_cost_path.relative_to(ROOT)))
    neural_viscosity={}
    for key,name in (('calibration','viscosity-calibration-results.json'),('accuracy','neural-viscosity-results.json'),('cost','neural-viscosity-cost-results.json'),('fast_cost','fast-preparation-cost-results.json'),('fixed_family','fixed-viscosity-family-results.json'),('fixed_cost','fixed-viscosity-cost-results.json'),('float32_accuracy','float32-learned-map-results.json'),('float32_cost','float32-map-cost-results.json'),('corrected_precision','fixed-viscosity-k2-precision-results.json')):
        path=DOCS/name
        if not path.exists():continue
        value=evidence(path);neural_viscosity[key]=value
        for relative,digest in value['source_evidence'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Neural viscosity evidence changed: '+relative)
            sources[relative]=digest
        if key=='float32_accuracy':
            for entry in value['rows']:
                arm=entry['float32_map']
                accuracy.append(dict(track='B',identifier=value['identifier'],horizons=2,regime=entry['regime'],interval=entry['interval'],
                    worst_error=max(arm['worst'].values())if arm['complete']else None,field_worst=arm['worst'],
                    all_finite_complete=arm['complete'],accuracy_pass=arm['accuracy_pass'],timing_pending=True,target_met=False,
                    accuracy_source=str(path.relative_to(ROOT))))
        if key in ('cost','fast_cost','fixed_cost','float32_cost','corrected_precision'):
            for point in value['points']:
                points.append(dict(point,source=str(path.relative_to(ROOT))))
                origin='fixed-viscosity-k2-precision-results.json'if key=='corrected_precision'else'fixed-viscosity-family-results.json'if key=='fixed_cost'else'float32-learned-map-results.json'if key=='float32_cost'else'neural-viscosity-results.json'
                joined=dict(point,timing_pending=False,accuracy_source=str((DOCS/origin).relative_to(ROOT)),timing_source=str(path.relative_to(ROOT)))
                prior=next((row for row in accuracy if(row.get('identifier'),row.get('regime'),row.get('interval'))==(point['identifier'],point['regime'],point['interval'])),None)
                if prior is None:accuracy.append(joined)
                else:
                    if prior['worst_error']!=point['worst_error']:raise ValueError('New execution timing changed accuracy join')
                    prior.update(joined)
    profile_path=DOCS/'coupled-CPU-stage-profile-results.json'
    coupled_profile=evidence(profile_path)if profile_path.exists()else None
    if coupled_profile is not None:
        for name,digest in coupled_profile['source_evidence'].items():
            if sha(ROOT/name)!=digest:raise ValueError('CPU profile evidence changed')
            sources[name]=digest
    tied_momentum_path=BASE/'tied-momentum-qualification-01.json'
    tied_momentum=None
    if tied_momentum_path.exists():
        value=evidence(tied_momentum_path)
        tied_momentum=dict(status=value['status'],summaries=value['summaries'],
            primitive_maximum_native_relative_difference=value['primitive_maximum_native_relative_difference'],
            candidate_frontier_point=False,future_reference_pressure_only_offline=True)
    application_preparation={}
    for key,relative in (('transport','runs/full-ai-next-priorities-v1/bounded-native-client-qualification-01/receipt.json'),
                         ('mechanics','runs/full-ai-next-priorities-v1/forecast-application-qualification-01/receipt.json'),
                         ('event_reference','runs/full-ai-next-priorities-v1/application-event-reference-01/bank.json')):
        path=ROOT/relative
        if not path.exists():continue
        value=evidence(path)
        application_preparation[key]=dict(status=value['status'],evidence=relative,sha256=sha(path),performance_claim=False,reserved_scenes_consumed=False)
        if key=='transport':application_preparation[key].update(scenes=len(value['rows']),states_compared=sum(r['physical_states_compared']for r in value['rows']))
        if key=='mechanics':application_preparation[key].update(cases=len(value['rows']),trained_horizons=value['trained_horizons'],diagnostic_horizon_views_only=True,independent_other_K_fit_claim=False)
        if key=='event_reference':
            if len(value['trajectories'])!=8 or value['states_per_scene']!=49:raise ValueError('Complete fixed application event references required')
            application_preparation[key].update(scenes=8,states_per_scene=49,compressed_state_bytes=value['compressed_state_bytes'])
            for row in value['trajectories']:
                receipt_path=ROOT/row['receipt'];receipt=read(receipt_path)
                if sha(receipt_path)!=row['receipt_sha256']or not receipt['first14states_and_RHS_bitwise_preserved']or not receipt['event_changes_next_step_velocity']:
                    raise ValueError('Application event/reference alignment differs')
                sources[row['receipt']]=sha(receipt_path)
    replica_progress=[]
    replica_registry=BASE/'fixed-viscosity-replication-inputs-01/models.json'
    if replica_registry.exists():
        evidence(replica_registry)
        for entry in read(replica_registry)['models']:
            identifier=entry['identifier'];pair=BASE/(identifier+'-pair-01')
            row=dict(identifier=identifier,horizons=entry['horizons'],replication_ordinal=entry['replication_ordinal'],
                managed_fit_restore_complete=(pair/'receipt.json').exists(),development_complete=False,
                no_independent_component_pretraining=True,selected_execution_policy_qualification_separate=True)
            if (pair/'receipt.json').exists():
                value=evidence(pair/'receipt.json')
                if value['status']!='completed'or not all(value['artifact_parity'].values()):raise ValueError('Completed replica pair requires exact artifact parity')
                row.update(training_run_id=value['training_run_id'],restore_run_id=value['restore_run_id'])
            evaluation=BASE/'fixed-viscosity-replication-evaluation-01'/identifier/'summary.json'
            if evaluation.exists():
                value=evidence(evaluation)
                if value['status']!='completed'or len(value['scenes'])!=8:raise ValueError('Completed replica summary requires all eight scenes')
                row.update(development_complete=True,
                    all_eight_scene_accuracy_pass=all(s['all_accuracy_criteria_pass']for scene in value['scenes']for s in scene['summaries']),
                    worst_error=max(v for scene in value['scenes']for s in scene['summaries']for v in s['worst'].values()))
            replica_progress.append(row)
    temporal_confirmation={}
    for key,name in (('replica_accuracy','fixed-viscosity-replication-results.json'),
                     ('replica_cost','fixed-viscosity-replication-cost-results.json'),
                     ('reserved_accuracy','temporal-reserved-accuracy-results.json'),
                     ('reserved_cost','temporal-reserved-cost-results.json'),
                     ('application','application-results/results.json'),
                     ('domain_change','application-domain-change-results.json')):
        path=DOCS/name
        if not path.exists():continue
        value=evidence(path);temporal_confirmation[key]=value
        for relative,digest in value['source_evidence'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Temporal confirmation evidence changed: '+relative)
            sources[relative]=digest
        if key=='reserved_accuracy':
            for row in value['rows']:
                accuracy.append(dict(track='B',identifier=row['identifier'],horizons=row['horizons'],
                    regime='reserved_'+row['regime'],interval=row['interval'],
                    worst_error=max(row['worst'].values())if row['complete']else None,field_worst=row['worst'],
                    all_finite_complete=row['complete'],accuracy_pass=row['accuracy_pass'],timing_pending=True,
                    target_met=False,accuracy_source=str(path.relative_to(ROOT))))
        if key in ('replica_cost','reserved_cost'):
            for raw in value['points']:
                point=dict(raw,source=str(path.relative_to(ROOT)))
                if key=='replica_cost':
                    point.update(identifier=raw['identifier']+'/seed-confirmation',model_identifier=raw['identifier'])
                points.append(point)
                joined=dict(point,timing_pending=False,accuracy_source=str(path.relative_to(ROOT)),timing_source=str(path.relative_to(ROOT)))
                prior=next((r for r in accuracy if(r.get('identifier'),r.get('regime'),r.get('interval'))==(point['identifier'],point['regime'],point['interval'])),None)
                if prior is None:accuracy.append(joined)
                else:
                    if prior['worst_error']!=point['worst_error']:raise ValueError('Reserved timing changed actual accuracy join')
                    prior.update(joined)
    return dict(status='interim',goal_complete=False,updated_unix=time.time(),points=points,accuracy=accuracy,
        temporal_confirmation=temporal_confirmation,replica_progress=replica_progress,
        offline_tied_momentum=tied_momentum,
        decoder_registration=decoder_registration,decoder_results=decoder_results,
        coupled_registration=coupled_registration,coupled_results=coupled_results,coupled_figure=coupled_figure,
        offline_transport_diagnosis=transport_diagnosis,provisional_registration=provisional_registration,
        provisional_results=provisional_results,offline_tracer_controls=tracer_controls,
        offline_displacement_controls=displacement_controls,
        learned_point_component=point_component,
        displacement_qualification=displacement_qualification,
        displacement_results=displacement_results,displacement_cost=displacement_cost,
        offline_displacement_components=displacement_components,neural_viscosity=neural_viscosity,application_preparation=application_preparation,
        coupled_cost=coupled_cost,coupled_profile=coupled_profile,
        pressure_expert_registration=expert_registration,pressure_expert_results=expert_results,
        temporal_loss_followup=followup,temporal_global_followup=global_followup,tail_head_followup=tail_followup,
        causal_request_components=component_results,curl_registration=curl_registration,curl_results=curl_results,increment_supervision_registration=increment_registration,increment_supervision_results=increment_results,conditioning_followup=conditioning,conditioning_results=conditioning_results,sharing_comparison=sharing,sharing_results=sharing_results,sharing_progress=sharing_progress,
        precise_routed_followup=precise_followup,independent_precise_confirmation=independent_precise,
        batches=batches,evidence_sha256=sources,source_sha256=sha(__file__),
        excluded_from_candidate_frontiers=['Untrained architecture timings','Analytical low-mode and tail controls',
            'Future-reference-RHS component oracle','Future-reference-pressure prefix controls and native momentum primitive qualification','Future-reference-provisional-velocity controls and native reconstruction qualification','Native-displacement tracer controls and reference-velocity scalar replay','Partial denormal-policy screens and instrumented stage profiles','Pressure-only causal request components','Failed float32 and mixed-storage materialized CNN variants','Numerically corrected or fallback outputs'],
        scopes=dict(A='Prepared pressure request; original128 consumed historical cases',
            B='Actual joint full-state prediction; worst equation/pressure/velocity/tracer error; all output horizons',
            timing='Model encoding, prescribed forcing and physical field reconstruction included; native integration separate',
            selection='Independent pressure points freeze weights/execution before capture; new tail/execution points use explicitly consumed development populations. Temporal reserved scenes remain untouched until temporal finalists qualify.'))


def plots(report):
    os.environ.setdefault('MPLCONFIGDIR',str(BASE/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.fonttype':'none','font.size':9})
    output=DOCS/'figures';output.mkdir(exist_ok=True)
    images=[]
    temporal_colors={name:i for i,name in enumerate(sorted({p['identifier']for p in report['points']if p['track']=='B'}))}
    def frontier(axis,points,xlimit,ylimit,show_target=True):
        unavailable=sum(p.get('worst_error')is None or p.get('all_finite_complete')is False for p in points)
        points=[p for p in points if p.get('worst_error')is not None and p.get('all_finite_complete')is not False]
        if unavailable:
            axis.text(.02,.98,f'{unavailable} incomplete interval(s) excluded; raw records retained',
                va='top',fontsize=7,transform=axis.transAxes)
        if not points:
            axis.text(.5,.5,'Complete-interval accuracy unavailable'if unavailable else'Qualified timing pending',ha='center',va='center',transform=axis.transAxes)
            axis.set_yscale('log');return
        ordered=sorted(points,key=lambda r:(r['median_ms'],r['worst_error']));best=math.inf;envelope=[]
        for index,point in enumerate(ordered):
            color=plt.get_cmap('tab20')(temporal_colors[point['identifier']]%20)if point.get('track')=='B'else plt.get_cmap('tab10')(index%10)
            axis.scatter(point['median_ms'],point['worst_error'],s=35,color=color,zorder=3,label=point['identifier'])
            if 'min_repeat_ms'in point:
                axis.errorbar(point['median_ms'],point['worst_error'],
                    xerr=[[point['median_ms']-point['min_repeat_ms']],[point['max_repeat_ms']-point['median_ms']]],
                    color=color,capsize=3,linewidth=1)
            annotate=(point['worst_error']<best or point['identifier']in('C106','C107'))if len(points)<=8 else point['identifier']in(
                'C106','C107','C107/whole_graph','a-head-calibration-s2-routed/whole_graph',
                'a-one-stage@2400','a-head-calibration-s2/compiled','a-head-calibration-s3/compiled')
            if point.get('track')=='B':annotate=False
            if annotate:
                label=point['identifier'].replace('a-head-calibration-s2-routed/whole_graph','successor/whole').replace('a-head-calibration','head-s1').replace('head-s1-s2','head-s2').replace('head-s1-s3','head-s3')
                offset=(-85,-15)if point['identifier']=='a-head-calibration-s3/compiled'else(12,-16)if point['identifier']=='a-head-calibration-s2/compiled'else(5,-12)if point['identifier']=='a-one-stage@2400'else(-80,-18)if point['identifier']=='a-head-calibration-s2-routed/whole_graph'else(5,5+8*(index%2))
                axis.annotate(label,(point['median_ms'],point['worst_error']),xytext=offset,textcoords='offset points',fontsize=7)
            if point['worst_error']<best:envelope.append(point);best=point['worst_error']
        axis.plot([p['median_ms']for p in envelope],[p['worst_error']for p in envelope],color='#425466',linewidth=1,label='Observed frontier')
        axis.set_yscale('log');axis.axhline(ylimit,color='#218454',linestyle='--',linewidth=1)
        if show_target:axis.axvline(xlimit,color='#218454',linestyle='--',linewidth=1)
        axis.set_xlim(left=0,right=max(xlimit*1.25,max(p.get('max_repeat_ms',p['median_ms'])for p in points)*1.15))
        axis.grid(True,alpha=.2,which='both');axis.set_ylim(bottom=min(ylimit*.6,min(p['worst_error']for p in points)*.7))
    def temporal_legend(fig,axes):
        # One shared legend prevents long model names from colliding across panels.
        unique={}
        for axis in axes:
            handles,labels=axis.get_legend_handles_labels()
            for handle,label in zip(handles,labels):unique.setdefault(label,handle)
        legend_height=.3+.18*math.ceil(len(unique)/2)
        height=6+legend_height
        fig.set_size_inches(14,height)
        fig.subplots_adjust(left=.08,right=.98,top=1-.6/height,
            bottom=(legend_height+.6)/height,wspace=.28)
        fig.legend(list(unique.values()),list(unique),loc='lower center',
            bbox_to_anchor=(.5,.02),ncol=2,fontsize=7,frameon=False,
            columnspacing=2,handletextpad=.6)

    fig,axis=plt.subplots(figsize=(8,4.5),layout='constrained')
    frontier(axis,[p for p in report['points']if p['track']=='A'and p['regime']=='historical_small'],4,7e-4)
    axis.set(title='Full AI pressure: historical comparison',xlabel='Median complete pressure inference (ms)',ylabel='Worst equation relative residual')
    fig.suptitle('Historical target passes require separate fresh and native evaluation',fontsize=10)
    for extension in ('svg','png'):fig.savefig(output/f'pressure-frontier.{extension}',dpi=160)
    plt.close(fig);images.append('pressure-frontier')
    if any(p['track']=='A'and p['regime'].startswith('fresh_')for p in report['points']):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for axis,regime in zip(axes,('fresh_small','fresh_32cubed_extension')):
            values=[p for p in report['points']if p['track']=='A'and p['regime']==regime]
            frontier(axis,values,4,7e-4,show_target=regime=='fresh_small')
            axis.set(title='Fresh small grids'if regime=='fresh_small'else'Fresh 32³ extension',
                xlabel='Median complete pressure inference (ms)',ylabel='Worst equation relative residual')
        fig.suptitle('Frozen learned-head finalists — independent native pressure requests',fontsize=10)
        for extension in('svg','png'):fig.savefig(output/('pressure-fresh-frontier.'+extension),dpi=160)
        plt.close(fig);images.append('pressure-fresh-frontier')
    if any(p['track']=='A'and p['regime'].startswith('confirmation_')for p in report['points']):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for axis,regime in zip(axes,('confirmation_small','confirmation_32cubed_extension')):
            values=[p for p in report['points']if p['track']=='A'and p['regime']==regime]
            frontier(axis,values,4,7e-4,show_target=regime=='confirmation_small')
            axis.set(title='Independent confirmation: small grids'if regime=='confirmation_small'else'Independent confirmation: 32³',
                xlabel='Median complete pressure inference (ms)',ylabel='Worst equation relative residual')
        fig.suptitle('New reserved scenes after complete-graph execution freeze',fontsize=10)
        for extension in('svg','png'):fig.savefig(output/('pressure-confirmation-frontier.'+extension),dpi=160)
        plt.close(fig);images.append('pressure-confirmation-frontier')
    if any(p['regime'].startswith('tail_development_')for p in report['points']):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for axis,regime in zip(axes,('tail_development_historical_small','tail_development_confirmation_small')):
            values=[p for p in report['points']if p['regime']==regime]
            frontier(axis,values,4,7e-4)
            axis.set(title='Historical requests'if regime.endswith('historical_small')else'Most recent consumed small-grid bank',
                xlabel='Median complete pressure inference (ms)',ylabel='Worst equation relative residual')
        fig.suptitle('Learned final block — consumed development comparison, fresh confirmation outstanding',fontsize=10)
        for extension in('svg','png'):fig.savefig(output/('pressure-tail-development-frontier.'+extension),dpi=160)
        plt.close(fig);images.append('pressure-tail-development-frontier')
    if any(p['regime'].startswith('precise_development_')for p in report['points']):
        from matplotlib.lines import Line2D
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for axis,regime in zip(axes,('precise_development_historical_small','precise_development_confirmation_small')):
            values=[p for p in report['points']if p['regime']==regime]
            frontier(axis,values,4,7e-4)
            ordered=sorted(values,key=lambda row:(row['median_ms'],row['worst_error']))
            axis.legend(handles=[Line2D([],[],color=plt.get_cmap('tab10')(index%10),marker='o',linestyle='none',
                label='Precise learned route'if value['identifier'].endswith('precise_routed')else'Compiled CNN')
                for index,value in enumerate(ordered)],loc='lower left',frameon=False,fontsize=8)
            axis.set(title='Historical requests'if regime.endswith('historical_small')else'Most recent consumed small-grid bank',
                xlabel='Median complete warm pressure inference (ms)',ylabel='Worst equation relative residual')
        fig.suptitle('Qualified precise learned execution — cold setup separate, new confirmation outstanding',fontsize=10)
        for extension in('svg','png'):fig.savefig(output/('pressure-precise-development-frontier.'+extension),dpi=160)
        plt.close(fig);images.append('pressure-precise-development-frontier')
    if any(p['regime'].startswith('precise_confirmation_')for p in report['points']):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for axis,regime in zip(axes,('precise_confirmation_small','precise_confirmation_32cubed_extension')):
            values=[p for p in report['points']if p['regime']==regime]
            frontier(axis,values,4,7e-4,show_target=regime=='precise_confirmation_small')
            if regime=='precise_confirmation_small':
                primary=next(value for value in values if value['identifier']=='a-tail8/precise_routed')
                axis.annotate('Fixed primary: a-tail8',(primary['median_ms'],primary['worst_error']),
                    xytext=(-45,-25),textcoords='offset points',fontsize=8)
            axis.set(title='New independent small-grid bank'if regime.endswith('_small')else'New independent 32³ extension',
                xlabel='Median complete warm pressure inference (ms)',ylabel='Worst equation relative residual')
        fig.suptitle('New confirmation — fixed primary, continuation seeds and consistently optimized C107',fontsize=10)
        for extension in('svg','png'):fig.savefig(output/('pressure-precise-confirmation-frontier.'+extension),dpi=160)
        plt.close(fig);images.append('pressure-precise-confirmation-frontier')
    for interval,label in (('single_block','Single-block prediction'),('rollout_240','240-step autonomous rollout')):
        fig,axes=plt.subplots(1,2,figsize=(14,8),layout=None)
        for axis,regime in zip(axes,('small','32cubed')):
            values=[p for p in report['points']if p['track']=='B'and p['regime']==regime and p['interval']==interval]
            frontier(axis,values,2,4e-3,show_target=regime=='small')
            axis.set(title='Small grids — primary latency target'if regime=='small'else'32³ — separate extension',
                xlabel='Median complete block latency / K (ms per step)',ylabel='Worst of equation, pressure, velocity and tracer error')
        temporal_legend(fig,axes)
        fig.suptitle(label+' — all intermediate states evaluated',fontsize=10)
        name='temporal-'+interval+'-frontier'
        for extension in ('svg','png'):fig.savefig(output/(name+'.'+extension),dpi=160)
        plt.close(fig);images.append(name)
    if any(p['regime']=='reserved_small'for p in report['points']):
        for interval,label in (('single_block','Single block'),('rollout_240','240-step own-history rollout')):
            fig,axes=plt.subplots(1,2,figsize=(14,8),layout=None)
            for axis,regime in zip(axes,('reserved_small','reserved_32cubed')):
                values=[p for p in report['points']if p['track']=='B'and p['regime']==regime and p['interval']==interval]
                frontier(axis,values,2,4e-3,show_target=regime=='reserved_small')
                axis.set(title='Original reserved small grids'if regime=='reserved_small'else'Original reserved32³',
                    xlabel='Fresh median complete block / K (ms/state)',ylabel='Worst equation/pressure/velocity/tracer error')
            temporal_legend(fig,axes)
            fig.suptitle(label+' — all frozen models/seeds retained, no primary reselection',fontsize=10)
            name='temporal-reserved-'+interval+'-frontier'
            for extension in ('svg','png'):fig.savefig(output/(name+'.'+extension),dpi=160)
            plt.close(fig);images.append(name)
    return images


def build(final=False):
    report=gather()
    if final:
        readiness=read(DOCS/'completion-audit.json')
        if not readiness['scientific_gates_passed']or readiness['failed']or readiness['auditor_sha256']!=sha(HERE/'audit_completion_v2.py'):
            raise ValueError('Current independent scientific completion audit must pass before final publication')
        for relative,digest in readiness['source_evidence'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Completion evidence changed before finalization: '+relative)
        report.update(status='complete',goal_complete=True,finalization=dict(
            auditor_sha256=sha(HERE/'audit_completion_v2.py'),all_scientific_gates_verified_before_rebuild=True,
            scientific_requirements_verified=[r['id']for r in readiness['requirements']if r['status']=='passed'and r['id']!='P-report'],
            final_report_requirement='P-report'))
    write(DOCS/'frontiers.json',report);images=plots(report)
    if report['coupled_figure']is not None:images.append('temporal-coupled-rollout')
    with (DOCS/'frontiers.csv').open('w',newline='')as stream:
        fields=['track','identifier','regime','interval','median_ms','full_block_ms','worst_error','target_met']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['points'])
    with (DOCS/'accuracy.csv').open('w',newline='')as stream:
        fields=['track','identifier','regime','interval','worst_error','median_error','p95_error','weights_sha256','accuracy_source','source','accuracy_pass','timing_pending','all_finite_complete','worst_scope']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['accuracy'])
    lead=('The fixed pressure primary independently confirms **2.133 ms median CPU inference and 4.6963e-4 worst equation residual**, '
          'with both continuation seeds also passing. This is the warm small-grid benchmark; cold setup, per-grid cost and native rejection remain separate. '
          'Full-state multi-step accuracy and own-history objectives remain open.'if report['independent_precise_confirmation']is not None else
          'The experiment remains active. Learned-head calibration has three seed replications and a frozen Track A selection. '
          'Historical target passes and independent pressure outcomes are reported separately. '
          'The simultaneous-state models still fail the accuracy target; latency alone does not establish a usable model.')
    temporal_note='Full-state temporal objectives remain open.'
    if report['neural_viscosity'].get('fast_cost')is not None:
        value=report['neural_viscosity']['fast_cost']
        point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        temporal_note=(f'The K2 all-neural model reaches {point["full_block_ms"]:.4f} ms per complete block '
            f'({point["median_ms"]:.4f} ms/state), with {point["worst_error"]:.7g} worst error through own240. '
            f'Combined small-grid median target met: {point["target_met"]}. '
            'Repeat medians straddle 4 ms; K3/K4, replicas, reserved scenes and application/event costs remain open.')
        lead=lead.replace('Full-state multi-step accuracy and own-history objectives remain open.',temporal_note)
    if report['neural_viscosity'].get('fixed_cost')is not None:
        family=report['neural_viscosity']['fixed_cost']
        qualified=[p for p in family['points']if p['regime']=='small'and p['interval']=='rollout_240'and p['target_met']]
        temporal_note=('Corrected-family K3/K4 meet all-field/own240 development accuracy and combined small-grid median speed. '
            'K2 precision confirmation, replicated seeds, reserved scenes and application/event measurements remain open. '
            'No internal stage has a separate 2 ms cap.')
        precision=report['neural_viscosity'].get('corrected_precision')
        if precision is not None:
            point=next(p for p in precision['points']if p['policy']=='fast-float32-map'and p['regime']=='small'and p['interval']=='rollout_240')
            temporal_note=(f'Corrected K2 float32-map policy: {point["full_block_ms"]:.4f} ms per complete block '
                f'({point["median_ms"]:.4f} ms/state), worst own240 error {point["worst_error"]:.7g}; combined median pass: {point["target_met"]}. '
                'K3/K4 also pass development accuracy and combined median speed. Replicated seeds, reserved scenes and application/event measurements remain open.')
        lead=(lead.split('Repeat medians straddle')[0]if 'Repeat medians straddle'in lead else lead)
        lead=('The fixed pressure primary independently confirms **2.133 ms median CPU inference and 4.6963e-4 worst equation residual**, '
            'with both continuation seeds passing. Warm small-grid benchmark; cold setup, per-grid cost and native rejection remain separate. '+temporal_note)
    if report['temporal_confirmation'].get('reserved_cost')is not None:
        value=report['temporal_confirmation']['reserved_cost']
        primaries=[p for p in value['points']if p['replication_ordinal']==1 and p['horizons']in(2,3,4)and p['regime']=='reserved_small'and p['interval']=='rollout_240']
        temporal_note=('Original reserved-scene confirmation is complete for every frozen horizon and added seed. '
            f'Primary K2/K3/K4 combined small-grid median passes: {[(p["horizons"],p["target_met"])for p in primaries]}. '
            'All intermediate fields, own240, failures, per-grid/tails,32³and setup costs remain reported separately. '
            'Application measurements are separate from model-only latency.')
        lead=('The fixed pressure primary independently confirms **2.133 ms median CPU inference and 4.6963e-4 worst equation residual**. '
            'Warm small-grid benchmark; cold setup and native acceptance limitations remain separate. '+temporal_note)
    md=['# Full AI study: completed results'if final else'# Full AI study: interim results','',lead,'',
        'Targets: C107 successor at approximately 4 ms and worst equation error ≤7e-4; '
        'K=2,3,4 at <2 ms per generated step, with complete block latency reported and worst equation/field error ≤4e-3.','',
        'See [the original plan](plan.md), [progress notes](execution-progress.md), [raw frontier data](frontiers.json) and [accuracy export](accuracy.csv).','',
        'Analytical teacher controls and untrained prototype timings are excluded from candidate frontiers. '
        'Native pressure acceptance and long-run physical validity are evaluated separately.','',
        '| Queue | Status | Active command |','| --- | --- | --- |']
    md.extend('| '+name+' | '+value['status']+' | '+str(value['active']or'—')+' |'for name,value in report['batches'].items())
    if report['replica_progress']:
        md.extend(['','## Additional temporal-cell seeds','',
            'These seeds share frozen learned component pretraining. Managed restoration and original-policy accuracy '
            'do not supply the separate selected float32/exact-execution qualification or CPU timing.','',
            '| Model | Seed ordinal | Exact managed fit/CPU restore | Eight-scene accuracy complete | Accuracy pass |',
            '| --- | ---: | --- | --- | --- |'])
        for row in report['replica_progress']:
            md.append(f'| {row["identifier"]} | {row["replication_ordinal"]} | {row["managed_fit_restore_complete"]} | {row["development_complete"]} | {row.get("all_eight_scene_accuracy_pass","Pending")} |')
    pressure=[point for point in report['points']if point['track']=='A'and point['regime']=='fresh_small']
    if pressure:
        md.extend(['','## Independent pressure outcomes','',
            'Accuracy uses all 768 small-grid requests; timing uses 48 fixed requests from the same 12 scenes, five balanced repeats. '
            'The 32³ extension is separate. The frozen primary is seed two.','',
            '| Model | Median CPU(ms) | Worst equation error | Native complete episodes |',
            '| --- | ---: | ---: | ---: |'])
        md.extend(f"| {point['identifier']} | {point['median_ms']:.3f} | {point['worst_error']:.6g} | {point['native_accepted_complete_episodes']}/16 |"for point in pressure)
        md.extend(['','All three calibrated seeds pass 7e-4. Original eager medians exceed 4 ms; complete-graph execution comparisons, when present, '
            'retain identical outputs on the already consumed cases and do not claim a new independent accuracy population. '
            'Every native attempt rejects at step one under unchanged gates. No application throughput is claimed.'])
    confirmation=[point for point in report['points']if point['track']=='A'and point['regime']=='confirmation_small']
    if confirmation:
        md.extend(['','## Earlier compiled-head confirmation','',
            'A second bank was reserved and captured after freezing complete-graph execution. All output pressures are bitwise equal '
            'to original eager execution. Accuracy uses all 768 small-grid requests; timing uses 48 fixed requests in five balanced repeats.','',
            '| Model | Median CPU(ms) | Worst equation error | Target met |',
            '| --- | ---: | ---: | --- |'])
        md.extend(f"| {point['identifier']} | {point['median_ms']:.3f} | {point['worst_error']:.6g} | {point['target_met']} |"for point in confirmation)
        md.extend(['','The successor passes 766 of 768 small-grid requests at 7e-4. Two startup requests on 15×6×8 grids exceed the accuracy target. '
            'That preceding model passes median speed but fails the combined target. The later learned-tail confirmation is reported below. Native acceptance remains unmet. '
            'Per-geometry trace preparation costs approximately 0.45–0.53 seconds; total numerical worker setup after module imports is 3.96 seconds, '
            'including three traces. This setup metric excludes Python/Torch import time. The confirmation worker peaks at approximately 0.98 GB RSS; '
            'the earlier many-layout benchmark peaked at approximately 3.86 GB. Warm timing excludes setup; short episodes must account for these costs.'])
    if report['tail_head_followup']is not None:
        md.extend(['','## Learned final block: development comparison','',
            'The new learned eight-mode final block completes its managed 10,000-update fit and exact CPU restore. '
            'It reaches 2.2551e-4 worst historical equation error and 5.5614e-4 on the most recent 768 small-grid requests. '
            'These are consumed development populations. Five balanced serial compiled repeats measure 4.370 ms historically '
            'and 4.802 ms on the recent small-grid mix; its parent measures 4.624 ms in the same run. '
            'That version improves accuracy while missing speed; the subsequent precise-execution results appear below. '
            'Compilation passes 2,176-request numerical qualification, with small floating-point differences from eager. '
            'See [the complete result and custody](tail-head-results.md).','',
            'A separate offline true-future-RHS diagnostic now passes all 384 pressure-component queries. '
            'It is excluded from temporal candidate frontiers. A materialized learned CNN execution pilot fails its initial '
            'numerical qualification and remains excluded while precision is investigated. '
            'See [that bounded execution investigation](materialized-pressure-execution-notes.md).'])
    if report['precise_routed_followup']is not None:
        md.extend(['','## Qualified precise learned execution','',
            'The precise routed path measures 1.909 ms historically and 2.075 ms on the consumed recent small-grid mix, '
            'at 2.2550e-4 and 5.5613e-4 worst equation error respectively. The paired complete CNN measures 4.586 ms on the recent mix. '
            'Full 2,176-request qualification preserves target classifications and keeps output/equation differences within the unchanged tolerances. '
            'Both requested targets pass on that development population; independent confirmation is reported separately below.','',
            'This median spans heterogeneous small grids: 16³ still measures approximately 7.93 ms, and 32³ approximately 83 ms. '
            'Cold setup and geometry-specific cache memory are substantial. The paired 64-geometry worker takes 179.43 seconds to prepare '
            'and peaks at 7.30 GB RSS while retaining both paths; its dense coefficient cache is approximately 419 MiB. '
            'This is not a standalone candidate memory measurement. '
            'For 9×11×11, observed per-geometry preparation is 3.09 seconds, giving approximately 14.94 ms per request if amortized over 240 requests. '
            'Warm target success is not cold application throughput. '
            'See [per-grid timings, cold accounting and evidence](precise-routed-pressure-results.md).','',
            'The causal pressure-request data view and neural objective now pass CPU/CUDA qualification for all four K values and all grid groups. '
            'The completed causal component outcomes are reported below; no full-state temporal advantage is claimed. '
            'See [the bounded next component fit](causal-pressure-request-component-plan.md).'])
    if report['independent_precise_confirmation']is not None:
        md.extend(['','## Independent pressure target confirmed','',
            'The fixed primary independently reaches **2.133 ms median CPU inference and 4.6963e-4 worst equation residual**. '
            'All 768 small-grid requests pass, and both continuation seeds confirm the combined target. '
            'C107 receives the same optimization: 1.642 ms but 1.5620e-3 worst residual. '
            'The primary was fixed before capture and was not replaced after seeing results.','',
            'This is the warm heterogeneous small-grid benchmark. The primary has approximately 7.08 seconds of worker setup after module imports '
            'and peaks near 0.98 GB RSS with three layouts. Larger grids, cold costs and per-grid timings remain separate. '
            'All models reject at the first native step on all 16 scenes under unchanged stricter gates; no application throughput is claimed. '
            'See [all four independent results and runtime-recovery evidence](pressure-confirmation-v2/results.md). '
            'Full-state multi-step accuracy and own-history goals remain open.'])
    if report['causal_request_components']:
        md.extend(['','## Causal pressure components','',
            'Both managed request fits and exact CPU restores are complete. Factoring a learned linear flux response from nonlinear transport '
            'reduces small-grid development median pressure error from 0.62780 to 0.059804 and worst error from 5.2160 to 0.64678. '
            'Cold starts pass, later evolution fails. These pressure-only reference-history queries supply no full-state or own-history claim, '
            'and neither component enters the temporal frontier. '
            'See [all component results](factored-pressure-request-results.md).'])
    if report['sharing_comparison']is not None and report['sharing_results']is None:
        md.extend(['','Both requested full-state sharing paths are implemented and CPU/CUDA-qualified for K=2,3,4. '
            'Six 10,000-update fits, exact CPU restores and development evaluations are registered and queued serially. '
            'One path decodes intermediate states from its predicted endpoint; the other retains first-prediction features for cheap later updates. '
            'Every block runs the spatial encoder once, all physical outputs are supervised and previous pressure is excluded after the shortcut diagnosis. '
            'The complete block budgets remain below4/6/8ms, with no separate internal-stage cap. '
            'See [the frozen comparison](temporal-sharing-registration-01.json).'])
    if report['sharing_comparison']is not None and report['sharing_results']is None:
        md.extend(['','| Sharing model | Fit and CPU restore complete | Development complete | Worst small-grid single-block field/equation error |',
            '| --- | --- | --- | ---: |'])
        for row in report['sharing_progress']:
            error=f"{row['worst_error']:.6g}"if row['development_complete']else'Pending'
            md.append(f"| {row['identifier']} | {row['fit_restore_complete']} | {row['development_complete']} | {error} |")
        md.extend(['','These accuracy-only progress rows do not claim CPU timing or replace the240-step rollout criterion.'])
    if report['sharing_results']is not None:
        md.extend(['','Both requested sharing architectures now have complete managed fits, CPU restores, every-horizon development evaluation, '
            '240-step own-history checks and five balanced CPU timing repeats. '
            'All six pass the small-grid median speed budget at0.85–1.47ms per generated step, but none passes accuracy. '
            'Individual latency tails are slower;32³ remains11.44–19.66ms per step. '
            'See [all six endpoint/reuse comparisons](temporal-sharing-results.md); short-block and long-rollout outcomes remain separate.'])
    if report['conditioning_followup']is not None and report['conditioning_results']is None:
        md.extend(['','A separately registered conditioning follow-up uses fixed aggregate training scales for known-force inputs and '
            'velocity/tracer increment outputs. It changes no physical error criterion, adds no parameters and starts from identical parameter tensors. '
            'The actual constructor passes CPU and CUDA qualification. Two registered K2 fits run serially after the completed original clean comparison. '
            'See [the measured scale diagnosis and next controlled experiment](temporal-step-scaled-plan.md).'])
    if report['conditioning_results']is not None:
        md.extend(['','The two controlled K2 step-unit conditioning fits and development comparisons are complete. '
            'See [their accuracy and control audit](temporal-step-scaled-results.md). CPU speed remains a separate unmeasured gate for those new fits.'])
    if report['increment_supervision_registration']is not None and report['increment_supervision_results']is None:
        md.extend(['','Both conditioning fits improve pressure but leave velocity near persistence and fail sustained full-state accuracy. '
            'A predeclared evidence gate admits the matched K2 MSE supervision comparison: the full-state MSE control and an increment-weighted MSE arm '
            'share the same network, initial tensors, sampling, schedule and physical terms. Both pass actual-constructor CPU/CUDA qualification. '
            'Only their velocity/tracer training-error units differ; evaluation thresholds remain unchanged. '
            'See [the controlled supervision plan](temporal-increment-supervision-plan.md).'])
    if report['increment_supervision_results']is not None:
        md.extend(['','The matched full-state and increment-weighted MSE fits, CPU restores and all-horizon development evaluations are complete. '
            'See [their audited comparison](temporal-increment-supervision-results.md). New CPU latency remains a separate gate.'])
    if report['curl_registration']is not None and report['curl_results']is None:
        md.extend(['','The increment-weighted loss improves training velocity/tracer fits but still fails development and240-step accuracy, while equation errors worsen. '
            'The next matched candidate generates velocity increments from learned edge potentials through a fixed linear curl readout inside the network. '
            'It has no added trainable parameters and preserves the same initial tensors and physical outputs. '
            'Structural impulse/divergence/wall/rank checks and actual two-block CPU/CUDA qualification pass. '
            'This representation is not a velocity post-projection and does not invoke a pressure solve; its fixed arithmetic remains part of model cost. '
            'See [the curl representation trial](temporal-curl-plan.md). Prediction accuracy is unproven.'])
    if report['curl_results']is not None:
        md.extend(['','The generative curl representation fit, CPU restore and all-horizon development/rollout comparison are complete. '
            'See [accuracy, physical diagnostics and matched controls](temporal-curl-results.md). Constraint satisfaction alone does not establish prediction accuracy.'])
    if report['pressure_expert_registration']is not None and report['pressure_expert_results']is None:
        md.extend(['','A causal full-state follow-up reuses the frozen learned pressure expert and managed request initialization. '
            'Both endpoint/intermediate and retained-feature K2/3/4 paths pass actual CPU/CUDA two-block qualification. '
            'One bounded K2 endpoint fit, exact CPU restore and complete development evaluation are admitted. '
            'Inherited training is additional; this is an architecture/initialization comparison, not a single-factor attribution. '
            'No new accuracy or latency result is claimed. See [the implementation and complete block budgets](temporal-pressure-expert-reuse-plan.md).'])
    if report['pressure_expert_results']is not None:
        md.extend(['','The causal full-state pressure-expert fit, exact CPU restore and all-horizon development/own-history comparison are complete. '
            'See [the full-state result and declared controls](temporal-pressure-expert-results.md). '
            'Its new weights have no measured CPU latency; pressure-only component timing is not a complete-block claim.'])
    if report['decoder_registration']is not None and report['decoder_results']is None:
        md.extend(['','A combined decoder representation comparison addresses measured lead-time, pressure-gauge and passive-tracer weaknesses. '
            'All sixK2/3/4 endpoint/reuse paths pass actual CPU/CUDA two-block qualification on both spatial densities. '
            'One boundedK2 endpoint fit and its exact CPU restore/full development evaluation are admitted. '
            'The immediate expert control has identical inherited assets and common fluid/request initial tensors; tracer architecture/initial function differ. '
            'No new prediction accuracy or CPU latency is claimed. See [the registered decoder comparison](temporal-decoder-plan.md).'])
    if report['decoder_results']is not None:
        md.extend(['','The combined lead-time/gauge/passive-tracer decoder fit, exact CPU restore and full development/own-history comparison are complete. '
            'See [the complete errors, controls and predeclared diagnostics](temporal-decoder-results.md). '
            'Structural representation properties alone do not qualify prediction accuracy; new CPU latency remains unmeasured.'])
    if report['coupled_registration']is not None and report['coupled_results']is None:
        md.extend(['','The next coupled full-state model reuses one fluid encoding and performs cheap internal state updates. '
            'Its pressure request and velocity readout share the predicted transport and pretrained neural flux coefficients. '
            'Actual CPU/CUDA two-block constructors qualify for K=1,2,3,4. One bounded K2 fit and complete development evaluation are registered. '
            'The combined budget is below 4 ms for the whole two-state block, with no individual-stage limit. '
            'No fitted accuracy or new CPU latency is claimed. See [the coupled plan](temporal-coupled-dynamics-plan.md).'])
    if report['coupled_results']is not None:
        md.extend(['','The coupled K2 fit, exact CPU restore and eight-scene development evaluation are complete. '
            'See [the full-state result](temporal-coupled-results.md). '
            'Incomplete rollouts retain their raw failures and finite-prefix diagnostics, with no complete-interval worst-error claim. '
            'CPU timing, K3/K4 extensions and reserved evaluation remain separate gates.'])
    if report['offline_transport_diagnosis']is not None:
        md.extend(['','Train-only counterfactuals identify provisional transport as the next fluid limitation. '
            'Exact reference pressure alone barely changes warm velocity error; reference provisional velocity reduces its worst to 2.825e-5. '
            'These future-reference controls are excluded from candidate frontiers. '
            'The unchanged model reproduces its prior fitting metrics bitwise. See [the diagnostic evidence](temporal-coupled-transport-results.md).'])
    if report['provisional_registration']is not None and report['provisional_results']is None:
        md.extend(['','Two matched 4,000-update continuations start from the same completed coupled checkpoint. '
            'Only the training-only provisional auxiliary weight differs; inference and all evaluation gates remain unchanged. '
            'Actual CPU/CUDA constructors and float32-derived native labels qualify. '
            'Fitting and full development/own240 evaluation remain separate from qualification. '
            'See [the frozen objective registration](temporal-provisional-registration-01.json).'])
    if report['provisional_results']is not None:
        md.extend(['','Both matched provisional-supervision continuations, exact CPU restores and full development/own240 evaluations are complete. '
            'See [the audited comparison](temporal-provisional-results.md). '
            'Only the training auxiliary weight differs; no new CPU timing is claimed.'])
    if report['offline_tracer_controls']:
        md.extend(['','Offline tracer controls reproduce all 256 native outputs exactly when native displacements are supplied. '
            'A first-order scalar replay with native velocity history reaches 0.001207 worst tracer error through 128 steps on small training grids. '
            'These are representation/transport controls, with no learned displacement predictor or full-state/240-step pass. '
            'See [the native readout checks](tracer-representation-results.md) and [reference-velocity replay](tracer-reference-velocity-replay-results.md). '
            'Neither joins the candidate frontiers.'])
    if report['coupled_cost']is not None:
        md.extend(['','The parent and both continuations now have complete five-repeat CPU cost measurements. '
            'Small-grid complete blocks take about 5.6 ms, or 2.8 ms per generated step, above the 4 ms/two-state budget. '
            'They also fail accuracy. Larger-grid cost, latency tails, worker preparation and memory remain separately reported. '
            'See [actual CPU costs and their exact accuracy joins](temporal-provisional-cost-results.md).'])
    if report['coupled_profile']is not None:
        md.extend(['','The CPU profile identifies the tracer decoder and field/forcing work as substantial small-grid costs. '
            'A preliminary denormal-policy screen shows no speed benefit and is not adopted. '
            'These instrumented/selected-case diagnostics are excluded from frontiers. See [the stage profile](coupled-CPU-stage-profile-results.md).'])
    if report['offline_displacement_controls']is not None:
        md.extend(['','The training-only face/cell transport labels and fixed convex readout pass all 80 registered cases on each of CPU and CUDA. '
            'Given native displacements, velocity and tracer outputs match the native primitive exactly at float32 precision; field/shift gradients are repeatable. '
            'This qualifies representation and labels, with zero training updates and no learned forecast or latency claim. '
            'See [the transport qualification](displacement-transport-results.md).'])
    if report['learned_point_component']is not None:
        md.extend(['','The 972-coefficient point layer completes 1,000 managed CUDA updates and an exact CPU restore. '
            'Its published float32 coefficients pass 24 native geometry/density checks at 2.51e-8 worst relative error. '
            'This is a learned initialization component, with no full-state accuracy, rollout or latency claim. '
            'See [component custody and native checks](point-sampling-results.md).'])
    if report['displacement_qualification']is not None:
        md.extend(['','The actual displacement model and training objective pass all 64 CPU and 64 CUDA constructor cases across K=1/2/3/4. '
            'Two-block gradients, shared computation, passive tracer and training-teacher isolation qualify with zero updates. '
            'Full-state fitting, development/own240 accuracy and actual CPU timing remain separate gates. '
            'See [the implementation checks](temporal-displacement-qualification-results.md) and [bounded model plan](temporal-displacement-plan.md).'])
    if report['displacement_results']is not None:
        md.extend(['','The displacement K2 fit, exact CPU restore and all eight development scenes are complete. '
            'Every horizon and own-history interval retains its actual accuracy outcome and raw failures. '
            'See [the full-state comparison and controls](temporal-displacement-results.md).'])
    if report['displacement_cost']is not None:
        md.extend(['','The displacement model also has five-repeat full-block and tensor CPU measurements against unchanged coupled baselines. '
            'Accuracy and timing are joined by exact restored model identity. These reference-history block costs do not establish sustained application throughput. '
            'See [actual CPU costs](temporal-displacement-cost-results.md).'])
    if report['offline_displacement_components']is not None:
        md.extend(['','Training-only controls identify the learned viscosity head as the main remaining fluid limitation. '
            'On the high-density 128-step training replay, native viscosity reduces worst equation error from 0.01166 to 0.000566; native displacement alone barely changes fluid error. '
            'The separate profile measures about 1.59 ms per small-grid block in the fixed field readout. '
            'These numerical controls and instrumented costs are excluded from candidate frontiers. '
            'See [the component diagnosis and stage profile](displacement-component-results.md).'])
    if report['neural_viscosity'].get('cost')is not None:
        md.extend(['','The managed 243-coefficient neural viscosity calibration repairs the remaining K2 accuracy failure. '
            'All eight development scenes and every horizon through own240 pass, with worst error 0.0033941. '
            'The fused readout preserves all 3,856 physical outputs byte for byte. Five balanced CPU repeats measure '
            '4.7793 ms per complete small-grid block, or 2.3896 ms per generated step, so speed still misses the 4 ms block budget. '
            '32³ takes 183.0 ms per block; latency tails and setup remain separate. K3/K4, replicas, reserved scenes and application timing remain open. '
            'See [full-state accuracy](neural-viscosity-results.md) and [exact fused execution costs](neural-viscosity-cost-results.md).'])
    if report['neural_viscosity'].get('fast_cost')is not None:
        value=report['neural_viscosity']['fast_cost']
        point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        md.extend(['',f'The next execution variant preserves all 3,856 physical outputs byte for byte while fusing prescribed forcing and removing unused pressure-input conversions. '
            f'Its five-repeat complete small-grid K2 median is {point["full_block_ms"]:.4f} ms ({point["median_ms"]:.4f} ms/state); '
            f'combined accuracy/speed target met: {point["target_met"]}. '
            'Every intermediate output and own240 remain included. Larger grids, tails, replicas and sustained application/event costs remain separate. '
            'See [qualified execution costs](fast-preparation-cost-results.md) and [the corrected K1/2/3/4 family](fixed-viscosity-family-plan.md).'])
    if report['neural_viscosity'].get('fixed_family')is not None:
        value=report['neural_viscosity']['fixed_family']
        md.extend(['','The four corrected-family managed CUDA fits, exact CPU restores and every-output/own240 evaluations complete. '
            f'Accuracy-qualified models across all eight scenes: {value["accuracy_qualified_all_eight_scenes"]}. '
            'The managed viscosity coefficients are frozen from initialization and each model supervises36,000physical states. '
            'Actual optimizer work differs with K; there is no equal-FLOP claim. '
            'See [full-state errors and actual training controls](fixed-viscosity-family-results.md).'])
    if report['neural_viscosity'].get('fixed_cost')is not None:
        value=report['neural_viscosity']['fixed_cost']
        md.extend(['',f'Corrected-family small-grid combined accuracy/median-speed qualifiers: {value["small_grid_combined_qualified"]}. '
            'Five balanced CPU repeats include complete2/3/4-call sequential B1 controls. Every fast execution policy retains '
            'all physical outputs byte for byte; preparation, memory,32³and latency tails remain separate. '
            'See [exact CPU costs and accuracy joins](fixed-viscosity-cost-results.md).'])
    if report['neural_viscosity'].get('float32_accuracy')is not None:
        value=report['neural_viscosity']['float32_accuracy']
        md.extend(['','Float32 arithmetic for the derived learned pressure map receives a new complete evaluation with unchanged trained tensors. '
            f'All-eight-scene accuracy qualification: {value["accuracy_qualified_all_eight_scenes"]}. '
            'Worst small-grid own240 error is0.0033953; no physical-byte equivalence or inherited accuracy is claimed for this rounding change. '
            'Its separately reported paired CPU comparison uses the same composed checkpoint; new-family weights need their own accuracy qualification. '
            'See [actual float32-map accuracy](float32-learned-map-results.md).'])
    if report['neural_viscosity'].get('float32_cost')is not None:
        value=report['neural_viscosity']['float32_cost']
        point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        md.extend(['',f'Float32-map five-repeat median complete K2 latency is {point["full_block_ms"]:.4f} ms '
            f'({point["median_ms"]:.4f} ms/state), with actual own240 worst error {point["worst_error"]:.7g}; '
            f'combined small-grid median target met: {point["target_met"]}. '
            'See [actual map-arithmetic costs and accuracy joins](float32-map-cost-results.md).'])
    for key,label,destination in (
        ('replica_accuracy','Completed managed additional-seed accuracy audit','fixed-viscosity-replication-results.md'),
        ('replica_cost','Actual three-seed CPU costs and matched sequential controls','fixed-viscosity-replication-cost-results.md'),
        ('reserved_accuracy','Original reserved-scene accuracy for every frozen model','temporal-reserved-accuracy-results.md'),
        ('reserved_cost','Fresh reserved CPU costs and all retained outcomes','temporal-reserved-cost-results.md'),
        ('application','Full measured application and interaction episodes','application-results/results.md'),
        ('domain_change','Actual fresh predictions after supported physical-box boundary changes','application-domain-change-results.md')):
        if report['temporal_confirmation'].get(key)is not None:
            md.extend(['',f'[{label}]({destination}).'])
    if report['temporal_confirmation'].get('replica_cost')is not None:
        md.extend(['','The selected K2 deployment uses float32 pressure-map arithmetic while B1uses float64. '
            'Their comparison measures complete deployment choices, including both precision and computation sharing; '
            'it does not isolate the encoder reuse contribution. The K4/B1comparison uses float64 for both.'])
    if report['application_preparation'].get('event_reference')is not None:
        md.extend(['','Application preparation qualifies bounded native transport and exact eight-step restart on the odd grid and32³. '
            'Twelve real neural/native mechanics cases verify state handoff, event invalidation and unsupported-boundary stopping; '
            'their K1/3/4views use the K2checkpoint and do not count as independent horizon fits or accuracy results. '
            'All eight declared development forcing-event references are captured, with their0..13field/RHS prefixes byte-identical '
            'to the original steady bank. These are preparation artifacts, with no application speed or native physical-acceptance claim. '
            'See [the application measurement scope](application-evaluation-plan.md) and [mechanics protocol](forecast-application-qualification-protocol-01.json).'])
    if report['offline_tied_momentum']is not None:
        md.extend(['','New offline native controls distinguish accumulated pressure from a last-pressure approximation. '
            'Using true future pressures on consumed training scenes, the two-step pressure-prefix control reaches4.96e-5worst cold velocity error '
            'versus0.0969using twice the endpoint pressure. These are diagnostic ceilings, not causal predictions or frontier points. '
            'Neural momentum coefficients shared with the pretrained flux layer also match the native primitive within2.70e-8relative. '
            'See [the offline evidence](tied-momentum-results.md) and [next coupled full-state plan](temporal-coupled-dynamics-plan.md).'])
    temporal=[point for point in report['points']if point['track']=='B'and point['regime']=='small'and point['interval']=='single_block'and point['horizons']>1]
    if temporal:
        md.extend(['','## Simultaneous-state primary screen','',
            'Complete inference includes encoding, forcing construction, all horizons and physical reconstruction. '
            'Error is the worst of equation, pressure, velocity and tracer across every development output.','',
            '| Model | K | Full block(ms) | Per generated step(ms) | Worst equation/field error | Target met |',
            '| --- | ---: | ---: | ---: | ---: | --- |'])
        for point in temporal:
            error=f"{point['worst_error']:.6g}"if point['worst_error']is not None else'Unavailable; incomplete interval'
            md.append(f"| {point['identifier']} | {point['horizons']} | {point['full_block_ms']:.3f} | {point['median_ms']:.3f} | {error} | {point['target_met']} |")
    if report['temporal_loss_followup']is not None:
        md.extend(['','The controlled K1 relative-MSE follow-up is complete. Its 24 comparison checks verified exact initialization, '
            'sampling work, architecture and inputs. It still fails accuracy: global worst single-block equation error is 1.06879 '
            'versus 1.03072 for the original objective. '
            'See [all intervals and fitting diagnostics](temporal-loss-followup-results.md). '
            'No new latency or bundled-model advantage is inferred from this negative test.'])
    if report['temporal_global_followup']is not None:
        md.extend(['','The shared-global K1 capacity comparison is also complete. Its managed fit and exact CPU restore retain the original '
            'data, objective, sampled work and training loop. Some velocity and rollout errors improve, but startup pressure still has '
            'approximately 100% relative error. Capacity alone has not established the required fitting accuracy. '
            'See [the full comparison](temporal-global-results.md) and [the user\'s endpoint/reuse architecture options](multi-step-generation-options.md).'])
    for image in images:md.extend(['','![Scientific comparison](figures/'+image+'.png)'])
    md.extend(['','The pressure plot uses the 128 consumed historical requests. The temporal plots use the 8 development scenes, '
        'separating small grids and 32³; the error is the worst across every output horizon and every declared field. '
        'Horizontal bars span the five repeat medians; they are not confidence intervals. '
        'Timing populations and error populations are documented in their raw receipts. '
        'Native and event receipts are linked in the progress notes. Rejected native attempts do not establish simulation throughput.',''])
    (DOCS/'results.md').write_text('\n'.join(md))
    payload=json.dumps(report,allow_nan=False).replace('<','\\u003c')
    body=''.join('<figure><img src="figures/'+name+'.svg" alt="Scientific frontier comparison"></figure>'for name in images)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full AI experiment results</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17283d;background:#f7f9fc}figure{margin:2rem 0;background:white;border:1px solid #d7dfe9}img{width:100%;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #d4dce7;padding:.5rem}th{cursor:pointer}select{font:inherit;padding:.4rem}a{color:#175f9d}.note{padding:1rem;background:#fff2d7}</style>
<h1>Full AI study — interim results</h1><p class="note">Pressure benchmark target independently confirmed: 2.133 ms median and 4.6963e-4 worst residual; both continuation seeds pass too. Warm small-grid mix only; cold cost and native rejection remain separate. __TEMPORAL_STATUS__ No application model has been promoted.</p>
<p>C107 successor: approximately 4 ms and worst equation error ≤7e-4. Multi-step: &lt;2 ms per generated step, with full block latency reported; worst equation and field errors ≤4e-3.</p>
<p><a href="plan.md">Original plan</a> · <a href="execution-progress.md">Progress notes</a> · <a href="requirements-audit.md">Requirement audit</a> · <a href="completion-audit.md">Current-state completion audit</a> · <a href="temporal-loss-followup-results.md">Loss comparison</a> · <a href="temporal-global-results.md">Global capacity comparison</a> · <a href="multi-step-generation-options.md">Endpoint and feature reuse options</a> · <a href="frontiers.csv">Frontier CSV</a> · <a href="accuracy.csv">Accuracy CSV</a> · <a href="frontiers.json">Evidence and hashes</a></p>'''+body+'''
<h2>Measured candidate points</h2><p>Click a column to sort. <select id="track"><option value="">Both tracks</option><option>A</option><option>B</option></select></p><table><thead id="head"></thead><tbody id="rows"></tbody></table>
<h2>Accuracy evaluations awaiting CPU timing</h2><p>Worst errors include every declared output horizon. Incomplete intervals cannot establish a full-interval worst error. These evaluations have no new latency point on the graphs.</p>
<div style="max-height:28rem;overflow:auto"><table><thead><tr><th>Model</th><th>K</th><th>Grid group</th><th>Interval</th><th>Worst equation/field error</th><th>Accuracy passes</th></tr></thead><tbody id="accuracy-rows"></tbody></table></div>
<p><a href="temporal-coupled-dynamics-plan.md">Coupled state-update plan</a> · <a href="temporal-coupled-transport-results.md">Transport diagnosis</a> · <a href="temporal-provisional-registration-01.json">Provisional-supervision comparison</a> · <a href="accuracy.csv">Complete accuracy export</a></p>
<p>Track A measures one prepared pressure request. Track B includes current-state encoding, prescribed forcing, a joint model evaluation and reconstruction of all K states. Numerical controls and analytical teacher diagnostics are excluded. Native integration and physical gates remain separate.</p>
<script id="data" type="application/json">'''+payload+'''</script><script>
const report=JSON.parse(document.getElementById('data').textContent), columns=['track','identifier','regime','interval','median_ms','full_block_ms','worst_error','target_met'];let order='identifier';
function render(){const body=document.getElementById('rows');body.replaceChildren();const filter=document.getElementById('track').value;let rows=report.points.filter(p=>!filter||p.track===filter);rows.sort((a,b)=>typeof a[order]==='number'?a[order]-b[order]:String(a[order]).localeCompare(String(b[order])));for(const row of rows){const tr=document.createElement('tr');for(const key of columns){const td=document.createElement('td'),v=row[key];td.textContent=typeof v==='number'?(key==='worst_error'?v.toExponential(4):v.toFixed(3)):v===undefined?'—':String(v);tr.append(td)}body.append(tr)}}
const tr=document.createElement('tr');for(const key of columns){const th=document.createElement('th');th.textContent=key.replaceAll('_',' ');th.onclick=()=>{order=key;render()};tr.append(th)}document.getElementById('head').append(tr);document.getElementById('track').onchange=render;render();
for(const row of report.accuracy.filter(r=>r.timing_pending)){const tr=document.createElement('tr');const error=row.worst_error===null||row.all_finite_complete===false?'Incomplete interval':Number(row.worst_error).toExponential(5);for(const value of [row.identifier,row.horizons??'—',row.regime,row.interval,error,String(row.accuracy_pass)]){const td=document.createElement('td');td.textContent=String(value);tr.append(td)}document.getElementById('accuracy-rows').append(tr)}
</script></html>'''
    if final:page=page.replace('Full AI study — interim results','Full AI study — completed results')
    (DOCS/'results.html').write_text(page.replace('__TEMPORAL_STATUS__',temporal_note))
    print('Built '+('completed'if final else'interim')+' full-AI report, scientific figures and machine-readable exports',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true',help='Require a current passing scientific audit before publishing final status')
    build(parser.parse_args().final)
