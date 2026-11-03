"""Summarize durable owned controller state without resubmitting any work."""
import time
from study_support import BASE, DOCS, read, write, sha


if __name__=='__main__':
    batches=[]
    for name in ('temporal-training-batch-02','architecture-training-batch-01','temporal-development-evaluation-batch-01',
                 'post-training-evaluation-batch-02','head-replication-batch-01','routed-qualification-batch-01',
                 'pressure-reserved-batch-01','temporal-mse-batch-01','temporal-mse-report-batch-01',
                 'temporal-global-batch-01','temporal-global-report-batch-01','pressure-whole-execution-batch-01','temporal-pressure-history-batch-01',
                 'pressure-confirmation-batch-01','temporal-component-ceiling-batch-01',
                 'tail8-training-batch-01','pressure-tail8-timing-01','tail8-replication-batch-01',
                 'precise-routed-pressure-timing-01','pressure-precise-confirmation-batch-01',
                 'pressure-precise-confirmation-timing-01','causal-request-training-batch-01',
                 'factored-request-training-batch-01','factored-request-training-batch-02','temporal-sharing-batch-01','temporal-sharing-postprocessing-batch-01','temporal-step-scaled-batch-01','temporal-increment-supervision-batch-01','temporal-curl-batch-01','temporal-pressure-expert-batch-01','temporal-pressure-expert-postprocessing-batch-01','temporal-decoder-batch-01','temporal-decoder-batch-02','temporal-coupled-batch-01','temporal-provisional-batch-01','temporal-provisional-timing-batch-01','temporal-provisional-timing-01','temporal-provisional-timing-batch-02','temporal-provisional-timing-02','coupled-CPU-stage-profile-batch-01','point-sampling-batch-01','point-sampling-evaluation-batch-02','temporal-displacement-batch-01','temporal-displacement-postprocessing-batch-01','displacement-component-diagnostic-batch-01','displacement-component-diagnostic-batch-02','displacement-CPU-stage-profile-batch-01','viscosity-calibration-batch-01','neural-viscosity-evaluation-batch-01','neural-viscosity-timing-batch-01','fast-preparation-timing-batch-01','fixed-viscosity-admission-batch-01','fixed-viscosity-training-batch-01','fixed-viscosity-postprocessing-batch-01','float32-map-evaluation-batch-01','float32-map-timing-batch-01','float32-map-publication-batch-01'):
        path=BASE/name/'execution.json'
        if path.exists():
            row=read(path)
            batches.append(dict(batch=name,status=row['status'],active=row.get('active'),
                completed_commands=len(row['completed']),execution_sha256=sha(path)))
    target=dict(temporal='All four corrected primary models fail development accuracy; small-grid latency alone passes',
                temporal_target=.004,temporal_reserved_scenes_consumed=False)
    path=BASE/'pressure-fresh-timing-01/summary.json'
    if path.exists():
        selection=read(DOCS/'pressure-finalist-selection.json')
        timing=next(row for row in read(path)['rows']if row['identifier']==selection['primary_model']and row['population']=='small')
        evidence=read(BASE/'pressure-fresh-evaluation-01'/selection['primary_model']/'summary.json')
        accuracy=next(row for row in evidence['prepared_requests']if row['population']=='small')
        target.update(pressure_primary=selection['primary_model'],pressure_fresh_worst_error=accuracy['worst_equation_error'],
            pressure_fresh_median_cpu_ms=timing['median_ms'],pressure_accuracy_target=.0007,
            pressure_native_accepted_complete_episodes=evidence['native_accepted_complete_episodes'])
    followup=DOCS/'temporal-loss-followup-results.json'
    if followup.exists():
        target['temporal_loss_followup']=read(followup)['next_decision']
    active_global=any('global'in row['batch']and row['status']=='running'for row in batches)
    stage='Shared-global temporal capacity fit/evaluation and pressure execution qualification'if active_global else'Complete pressure-graph execution and temporal pressure-history diagnosis; endpoint/reuse architectures next'
    whole=BASE/'pressure-whole-timing-01/summary.json'
    if whole.exists():
        row=next(row for row in read(whole)['rows']if row['checkpoint']=='a-head-calibration-s2-routed'and row['variant']=='whole_graph'and row['population']=='fresh_small')
        target['whole_pressure_execution']=dict(median_cpu_ms=row['median_ms'],worst_equation_error=row['worst_equation_relative_residual'],
            exact_consumed_output_equivalence=True,new_independent_accuracy_claim=False)
    confirmation=BASE/'pressure-confirmation-timing-01/summary.json'
    if confirmation.exists():
        row=next(row for row in read(confirmation)['rows']if row['identifier']=='a-head-calibration-s2-routed'and row['population']=='small')
        accuracy=next(row for row in read(BASE/'pressure-confirmation-evaluation-01/a-head-calibration-s2-routed/summary.json')['prepared_requests']if row['population']=='small')
        target['independent_compiled_confirmation']=dict(median_cpu_ms=row['median_ms'],worst_equation_error=accuracy['worst_equation_error'],
            cases_passing_7e_4=accuracy['passing_7e_4'],total_small_cases=768,combined_target_met=False)
    tail=BASE/'pressure-tail-screen-01.json'
    if tail.exists():
        target['pressure_tail_development_screen']=read(tail)['summaries']
    if all(row['status']!='running'for row in batches)and confirmation.exists():
        stage='Independent pressure confirmation and temporal diagnostics complete; learned8-mode tail and endpoint/reuse models next'
    tail_evaluation=BASE/'pressure-tail8-evaluation-02/a-tail8/summary.json'
    if tail_evaluation.exists():
        value=read(tail_evaluation)
        target['learned_tail8_development']=dict(status=value['status'],
            populations=value['populations'],new_independent_accuracy_claim=False,
            source_sha256=sha(tail_evaluation))
        stage='Learned eight-mode tail fit and actual prediction evaluation complete; CPU timing, conditional replicas and new independent confirmation next'
    tail_timing=BASE/'pressure-tail8-timing-01/summary.json'
    if tail_timing.exists():
        value=read(tail_timing)
        target['learned_tail8_timing']=dict(rows=value['rows'],source_sha256=sha(tail_timing),
            new_independent_accuracy_claim=False)
        stage='Learned eight-mode tail development accuracy and CPU comparison complete; assess replicas and fresh confirmation'
    tail_oracle=BASE/'temporal-tail-pressure-component-oracle-01.json'
    if tail_oracle.exists():
        value=read(tail_oracle)
        target['temporal_tail_pressure_component_oracle']=dict(summaries=value['summaries'],
            source_sha256=sha(tail_oracle),temporal_candidate=False,future_reference_RHS='Offline diagnostic only')
    diagnosis=BASE/'materialized-pressure-diagnosis-01.json'
    if diagnosis.exists():
        value=read(diagnosis)
        target['materialized_pressure_initial_diagnosis']=dict(requests=len(value['rows']),
            failing_tight_equation_comparison=sum(not row['parity_pass']for row in value['rows']),
            maximum_pressure_relative_difference=max(row['pressure_relative_difference']for row in value['rows']),
            status='Initial float32 construction not qualified',source_sha256=sha(diagnosis))
        stage='Higher-precision learned CNN materialization qualification and causal pressure-request training data preparation'
    mixed=BASE/'mixed-materialized-pressure-qualification-01/a-tail8.json'
    if mixed.exists():
        value=read(mixed)
        target['mixed_materialized_pressure_qualification']=dict(status=value['status'],
            requests=value['tested_requests'],source_sha256=sha(mixed),
            new_independent_accuracy_claim=False)
        stage='Higher-precision materialized pressure qualified; balanced CPU comparison next'
    request_data=BASE/'temporal-request-training-view-01/manifest.json'
    if request_data.exists():
        value=read(request_data)
        target['causal_pressure_request_data']=dict(status=value['status'],parents=len(value['parents']),
            archive_bytes=value['archive_bytes'],source_sha256=sha(request_data),
            development_or_reserved_used=False,temporal_candidate=False)
    precise=BASE/'precise-routed-pressure-qualification-01/a-tail8.json'
    if precise.exists():
        value=read(precise)
        target['precise_routed_pressure_qualification']=dict(status=value['status'],requests=value['tested_requests'],
            source_sha256=sha(precise),new_independent_accuracy_claim=False)
    precise_timing=BASE/'precise-routed-pressure-timing-01/summary.json'
    if precise_timing.exists():
        value=read(precise_timing)
        target['precise_routed_pressure_timing']=dict(rows=value['rows'],source_sha256=sha(precise_timing),
            independent_target_confirmed=False)
        stage='Precise routed pressure passes consumed accuracy and warm timing targets; continuation replicas and new confirmation pending; causal request model CPU/CUDA qualified'
    for device in ('cpu','cuda'):
        qualification=BASE/f'causal-pressure-request-{device}-qualification-01.json'
        if qualification.exists():
            value=read(qualification)
            target['causal_pressure_request_'+device]=dict(status=value['status'],optimizer_updates=value['optimizer_updates'],
                source_sha256=sha(qualification),full_state_temporal_candidate=False)
    precise_confirmation=DOCS/'pressure-confirmation-v2/results.json'
    if precise_confirmation.exists():
        value=read(precise_confirmation)
        target['independent_precise_pressure_confirmation']=dict(primary=value['primary'],
            primary_target_pass=value['primary_independent_target_pass'],
            all_three_seeds_pass=value['all_three_continuation_seeds_independent_target_pass'],
            rows=value['rows'],source_sha256=sha(precise_confirmation),application_model_promoted=False)
        stage='Pressure benchmark target independently confirmed for all three tail seeds; full-state multi-step accuracy and240-step continuation remain open; managed causal request trainer preparation next'
    for key,name in (('causal_request_fit','causal-pressure-request-training-result.json'),
                     ('factored_request_fit','factored-pressure-request-results.json')):
        path=DOCS/name
        if path.exists():
            value=read(path)
            target[key]=dict(status=value['status'],groups=value['groups'],source_sha256=sha(path),
                component_only=True,full_state_temporal_candidate=False)
    sharing=DOCS/'temporal-sharing-registration-01.json'
    if sharing.exists():
        target['full_state_sharing_comparison']=dict(models=read(sharing)['models'],
            registration_sha256=sha(sharing),reserved_scenes_consumed=False)
        stage='Pressure benchmark independently confirmed; both causal request fits complete but inaccurate; six full-state endpoint/reuse K2/3/4 fits and development evaluations queued serially'
    sharing_models=[]
    for k in (2,3,4):
        for architecture in ('endpoint','reuse'):
            identifier=f'b{k}-{architecture}';pair=BASE/f'{identifier}-pair-01'
            evaluation=BASE/'temporal-sharing-development-evaluation-01'/identifier/'summary.json'
            model=dict(identifier=identifier,fit_restore_complete=(pair/'receipt.json').exists(),development_complete=evaluation.exists())
            training=pair/'train/receipt.json'
            if training.exists():
                trained=read(training)
                model['CUDA_training_complete']=trained['status']=='complete'and trained['mode']=='train'
                model['actual_optimizer_updates']=trained['optimizer_updates']
                model['training_receipt_sha256']=sha(training)
            if evaluation.exists():
                value=read(evaluation)
                model['small_grid_worst_single_block']={name:max(scene['summaries'][0]['worst'][name]for scene in value['scenes']if max(scene['counts_xyz'])<=16)for name in ('pressure','equation','velocity','tracer')}
                model['evaluation_sha256']=sha(evaluation)
            sharing_models.append(model)
    target['full_state_sharing_models']=sharing_models
    sharing_results=DOCS/'temporal-sharing-results.json'
    if sharing_results.exists():
        value=read(sharing_results)
        target['full_state_sharing_results']=dict(status=value['status'],finalists=value['small_grid_finalists'],
            all_requested_K_targets_pass=value['all_requested_K_targets_pass'],source_sha256=sha(sharing_results))
        stage='Six explicit full-state endpoint/reuse comparisons completed; assess controlled accuracy and timing results before conditional replicas/reserved evaluation'
    conditioning=DOCS/'temporal-step-scaled-registration-01.json'
    if conditioning.exists():
        target['step_scaled_conditioning_followup']=dict(models=read(conditioning)['models'],
            registration_sha256=sha(conditioning),status='Registered; two K2 fits queued after clean original comparison',
            physical_targets_unchanged=True,reserved_scenes_consumed=False)
    conditioning_batch=next((row for row in batches if row['batch']=='temporal-step-scaled-batch-01'),None)
    if conditioning_batch is not None and conditioning.exists():
        target['step_scaled_conditioning_followup']['status']=conditioning_batch['status']
        target['step_scaled_conditioning_followup']['active_command']=(conditioning_batch.get('active')or{}).get('id')
        if conditioning_batch['status']=='running':
            stage='Original six-model sharing comparison complete: median speed passes, accuracy fails; controlled step-unit conditioning follow-up active'
    conditioning_results=DOCS/'temporal-step-scaled-results.json'
    if conditioning_results.exists():
        value=read(conditioning_results)
        target['step_scaled_conditioning_results']=dict(status=value['status'],
            accuracy_qualified_recipes=value['accuracy_qualified_small_grid_recipes'],
            CPU_latency_claim=value['CPU_latency_claim'],source_sha256=sha(conditioning_results))
        stage='Original endpoint/reuse comparison and two controlled step-unit follow-ups complete; assess accuracy and remaining latency/replication gates'
    increment_registration=DOCS/'temporal-increment-supervision-registration-01.json'
    if increment_registration.exists():
        value=read(increment_registration)
        target['increment_supervision_followup']=dict(models=value['models'],
            registration_sha256=sha(increment_registration),physical_targets_unchanged=True,CPU_latency_claim=False)
        active=next((row for row in batches if row['batch']=='temporal-increment-supervision-batch-01'),None)
        if active is not None:
            target['increment_supervision_followup'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
            if active['status']=='running':stage='Both conditioning fits complete and fail full-state accuracy; matched full-state versus increment-MSE supervision fits active'
    increment_result=DOCS/'temporal-increment-supervision-results.json'
    if increment_result.exists():
        value=read(increment_result)
        target['increment_supervision_result']=dict(status=value['status'],accuracy_qualified_recipes=value['accuracy_qualified_small_grid_recipes'],
            CPU_latency_claim=value['CPU_latency_claim'],source_sha256=sha(increment_result))
        stage='Matched full-state and increment-MSE supervision comparisons complete; assess actual field/rollout accuracy and remaining target gates'
    curl_registration=DOCS/'temporal-curl-registration-01.json'
    if curl_registration.exists():
        target['curl_representation_followup']=dict(model=read(curl_registration)['model'],
            registration_sha256=sha(curl_registration),physical_targets_unchanged=True,CPU_latency_claim=False,
            fixed_linear_curl_readout_inside_model=True,velocity_post_projection=False)
        active=next((row for row in batches if row['batch']=='temporal-curl-batch-01'),None)
        if active is not None:
            target['curl_representation_followup'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
            if active['status']=='running':stage='Matched supervision comparison complete and fails accuracy; generative curl velocity readout trial active'
    curl_result=DOCS/'temporal-curl-results.json'
    if curl_result.exists():
        value=read(curl_result)
        target['curl_representation_result']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_small_grid_recipe'],
            CPU_latency_claim=value['CPU_latency_claim'],source_sha256=sha(curl_result))
        stage='Generative curl comparison complete and fails accuracy; frozen learned pressure-expert port and causal full-state integration next'
    pressure_port=DOCS/'temporal-pressure-core-port-results.json'
    if pressure_port.exists():
        value=read(pressure_port)
        target['frozen_pressure_core_port']=dict(status=value['status'],rows=value['rows'],source_sha256=sha(pressure_port),
            component_only=True,CPU_latency_claim=False,causal_prediction_accuracy_claim=False)
        stage='Frozen learned pressure port passes CPU/CUDA batched gradients and parent parity; causal full-state integration next; temporal accuracy remains unmet'
    expert_registration=DOCS/'temporal-pressure-expert-registration-01.json'
    if expert_registration.exists():
        target['full_state_pressure_expert']=dict(registration_sha256=sha(expert_registration),
            component_only=False,pressure_expert_calls_per_block=1,CPU_latency_claim=False,
            actual_inherited_training_reported=True,reserved_scenes_consumed=False)
        active=next((row for row in batches if row['batch']=='temporal-pressure-expert-batch-01'),None)
        if active is not None:
            target['full_state_pressure_expert'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
            stage='Causal full-state pressure-expert K2 endpoint fit, exact CPU restore and all-horizon/own240 evaluation; all six architectures CPU/CUDA qualified'
    expert_result=DOCS/'temporal-pressure-expert-results.json'
    if expert_result.exists():
        value=read(expert_result)
        target['full_state_pressure_expert_result']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_small_grid_recipe'],
            CPU_latency_claim=False,source_sha256=sha(expert_result))
        stage='Causal full-state expert improves single-block errors but fails accuracy and own240 stability; train-only controls identify lead-time, gauge and passive-tracer decoder priorities'
    decoder_registration=DOCS/'temporal-decoder-registration-01.json'
    if decoder_registration.exists():
        target['combined_decoder_comparison']=dict(registration_sha256=sha(decoder_registration),
            CPU_latency_claim=False,causal_accuracy_claim=False,reserved_scenes_consumed=False)
        active=next((row for row in reversed(batches) if row['batch']in('temporal-decoder-batch-01','temporal-decoder-batch-02')),None)
        if active is not None:
            target['combined_decoder_comparison'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
        stage='Combined lead-time/gauge/passive-tracer decoder K2 fit and complete evaluation; all six paths CPU/CUDA qualified including both spatial densities'
    decoder_result=DOCS/'temporal-decoder-results.json'
    if decoder_result.exists():
        value=read(decoder_result)
        target['combined_decoder_result']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_small_grid_recipe'],
            CPU_latency_claim=False,source_sha256=sha(decoder_result))
        stage='Combined decoder improves velocity/tracer and own240 stability but misses accuracy; qualify coupled learned pressure histories and momentum next'
    tied=BASE/'tied-momentum-qualification-01.json'
    if tied.exists():
        value=read(tied)
        target['offline_tied_momentum_and_pressure_sum']=dict(status=value['status'],summaries=value['summaries'],
            primitive_maximum_native_relative_difference=value['primitive_maximum_native_relative_difference'],source_sha256=sha(tied),
            candidate_frontier_point=False,CPU_latency_claim=False,future_reference_pressure_only_offline=True)
    coupled_registration=DOCS/'temporal-coupled-registration-01.json'
    if coupled_registration.exists():
        target['coupled_dynamics_comparison']=dict(registration_sha256=sha(coupled_registration),
            CPU_latency_claim=False,causal_accuracy_claim=False,reserved_scenes_consumed=False,
            combined_block_budgets_ms={'2':4,'3':6,'4':8},individual_internal_stage_limit=None)
        for device in ('cpu','cuda'):
            path=BASE/f'temporal-coupled-trainer-{device}-qualification-01.json'
            if path.exists():
                value=read(path)
                target['coupled_dynamics_comparison'][device+'_qualification']=dict(status=value['status'],
                    cases=len(value['rows']),optimizer_updates=value['optimizer_updates'],source_sha256=sha(path))
        active=next((row for row in batches if row['batch']=='temporal-coupled-batch-01'),None)
        if active is not None:
            target['coupled_dynamics_comparison'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
            stage='Coupled causal K2 full-state fit, exact CPU restore and complete development/own240 evaluation; CPU/CUDA constructors qualify for K1/2/3/4'
    coupled_result=DOCS/'temporal-coupled-results.json'
    if coupled_result.exists():
        value=read(coupled_result)
        target['coupled_dynamics_result']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_small_grid_recipe'],
            own_history_failures=value['own_history_failures'],CPU_latency_claim=False,source_sha256=sha(coupled_result))
        stage='Coupled full-state fit and full development evaluation complete; assess every-field accuracy and own240 completeness before timing or extensions'
    transport_result=DOCS/'temporal-coupled-transport-results.json'
    if transport_result.exists():
        value=read(transport_result)
        target['offline_provisional_transport_diagnosis']=dict(status=value['status'],summaries=value['summaries'],
            source_sha256=sha(transport_result),candidate_frontier_point=False,CPU_latency_claim=False)
    provisional_registration=DOCS/'temporal-provisional-registration-01.json'
    if provisional_registration.exists():
        value=read(provisional_registration)
        target['provisional_supervision_comparison']=dict(registration_sha256=sha(provisional_registration),models=value['models'],
            updates_per_arm=value['updates_per_arm'],same_checkpoint=True,inference_unchanged=True,CPU_latency_claim=False,reserved_scenes_consumed=False)
        for device in ('cpu','cuda'):
            path=BASE/f'temporal-provisional-trainer-{device}-qualification-01.json'
            if path.exists():
                q=read(path)
                target['provisional_supervision_comparison'][device+'_qualification']=dict(status=q['status'],gradient_cases=len(q['rows']),
                    native_label_cases=len(q['training_label_rows']),maximum_RHS_label_error=q['maximum_training_label_native_RHS_relative_error'],
                    source_sha256=sha(path),optimizer_updates=q['optimizer_updates'])
        active=next((r for r in batches if r['batch']=='temporal-provisional-batch-01'),None)
        if active is not None:
            target['provisional_supervision_comparison'].update(status=active['status'],active_command=(active.get('active')or{}).get('id'))
        stage='Transport diagnosis identifies provisional prediction as fluid limitation; matched4000-update full-state continuations and complete evaluation underway'
    for key,name in (('offline_tracer_representation','tracer-representation-results.json'),
                     ('offline_reference_velocity_scalar_replay','tracer-reference-velocity-replay-results.json')):
        path=DOCS/name
        if path.exists():
            value=read(path)
            target[key]=dict(status=value['status'],summaries=value['summaries'],source_sha256=sha(path),
                candidate_frontier_point=False,CPU_latency_claim=False,full_state_forecast=False)
    provisional_result=DOCS/'temporal-provisional-results.json'
    if provisional_result.exists():
        value=read(provisional_result)
        target['provisional_supervision_result']=dict(status=value['status'],accuracy_qualified_recipes=value['accuracy_qualified_small_grid_recipes'],
            source_sha256=sha(provisional_result),CPU_latency_claim=False,reserved_scenes_consumed=False)
        stage='Matched continuation fits/restores/evaluations complete; assess actual transport-supervision effect and qualified tracer-representation evidence'
    timing=next((r for r in reversed(batches)if r['batch']in('temporal-provisional-timing-batch-02','temporal-provisional-timing-02')),None)
    if timing is not None:
        target['coupled_CPU_cost_measurement']=dict(status=timing['status'],batch=timing['batch'],
            source_bound_accuracy_results='temporal-provisional-results.json',combined_target_claim=False,
            scopes=['Complete two-state model blocks','Prepared tensors'],repeats=5)
        if timing['status']=='running':stage='Matched continuations complete and fail accuracy; logging-wrapper recovery retained; balanced serial CPU model-cost measurements active'
    cost=DOCS/'temporal-provisional-cost-results.json'
    if cost.exists():
        value=read(cost)
        target['coupled_CPU_cost_result']=dict(status=value['status'],rows=value['timing'],source_sha256=sha(cost),combined_target_met=False)
        stage='Both continuations fail accuracy; full CPU blocks also exceed budget; native displacement controls and stage profile identify next transport/readout priorities'
    profile=DOCS/'coupled-CPU-stage-profile-results.json'
    if profile.exists():
        value=read(profile)
        target['coupled_CPU_stage_profile']=dict(status=value['status'],stage_summaries=value['stage_summaries'],
            source_sha256=sha(profile),denormal_variant_adopted=False,candidate_frontier_point=False)
    displacement_registration=DOCS/'temporal-displacement-fit-registration-01.json'
    if displacement_registration.exists():
        target['displacement_full_state']=dict(status='registered',registration_sha256=sha(displacement_registration),
            full_state_accuracy_claim=False,CPU_latency_claim=False,reserved_scenes_consumed=False)
        for device in ('cpu','cuda'):
            path=BASE/f'temporal-displacement-{device}-qualification-01.json'
            if path.exists():
                q=read(path)
                target['displacement_full_state'][device+'_qualification']=dict(status=q['status'],cases=len(q.get('rows',[])),source_sha256=sha(path))
        active=next((r for r in batches if r['batch']=='temporal-displacement-batch-01'),None)
        if active is not None:
            target['displacement_full_state'].update(batch_status=active['status'],active_command=(active.get('active')or{}).get('id'))
        stage='Actual displacement constructors pass; bounded managed K2 fitting/restoration/evaluation underway'
    displacement_result=DOCS/'temporal-displacement-results.json'
    if displacement_result.exists():
        value=read(displacement_result)
        target['displacement_full_state_result']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_small_grid_recipe'],source_sha256=sha(displacement_result))
        stage='Displacement full-state accuracy and own240 results complete; assess actual CPU costs before selection'
    displacement_cost=DOCS/'temporal-displacement-cost-results.json'
    if displacement_cost.exists():
        value=read(displacement_cost)
        target['displacement_CPU_cost']=dict(status=value['status'],rows=value['timing'],source_sha256=sha(displacement_cost))
        stage='Displacement accuracy and actual CPU costs complete; assess combined gates and remaining K/replication requirements'
    displacement_components=DOCS/'displacement-component-results.json'
    if displacement_components.exists():
        value=read(displacement_components)
        target['offline_displacement_components']=dict(status=value['status'],source_sha256=sha(displacement_components),
            candidate_frontier_point=False,optimizer_updates=0,CPU_latency_claim=False)
        stage='Displacement accuracy and speed still fail; train-only controls identify neural viscosity, and actual CPU profile prioritizes fixed-readout fusion'
    neural_cost=DOCS/'neural-viscosity-cost-results.json'
    if neural_cost.exists():
        value=read(neural_cost)
        point=next(p for p in value['points']if p['identifier']=='b2-displacement-viscosity-fused'and p['regime']=='small'and p['interval']=='rollout_240')
        target['neural_viscosity_K2']=dict(accuracy_pass=point['accuracy_pass'],worst_error=point['worst_error'],
            median_block_ms=point['full_block_ms'],median_ms_per_generated_step=point['median_ms'],combined_target_met=point['target_met'],
            exact_fused_physical_outputs=3856,source_sha256=sha(neural_cost),K3_K4_and_replicas_pending=True,reserved_scenes_consumed=False)
        target['temporal']='Repaired all-neural K2 passes every field/horizon and own240 on all8development scenes; complete CPU block4.7793ms still exceeds4ms; corrected-family K3/K4 remain open'
        stage='First all-neural K2 accuracy pass; exact fused readout lowers full CPU block to4.7793ms; further execution savings and remaining horizon/replication gates next'
    fast_cost=DOCS/'fast-preparation-cost-results.json'
    if fast_cost.exists():
        value=read(fast_cost)
        point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        target['fast_neural_K2']=dict(accuracy_pass=point['accuracy_pass'],worst_error=point['worst_error'],
            median_block_ms=point['full_block_ms'],median_ms_per_generated_step=point['median_ms'],combined_target_met=point['target_met'],
            source_sha256=sha(fast_cost),exact_physical_outputs=3856,reserved_scenes_consumed=False)
        target['temporal']='K2 complete small-grid combined target met'if point['target_met']else'K2 accuracy passes, complete CPU speed still misses'
        stage='Qualified forcing and codec execution complete; corrected-family K1/2/3/4 fits, replicas and application/reserved gates remain'
    for device in ('cpu','cuda'):
        path=BASE/f'fixed-viscosity-{device}-qualification-01.json'
        if path.exists():
            value=read(path);target['fixed_viscosity_'+device]=dict(status=value['status'],source_sha256=sha(path),optimizer_updates=0)
    active=next((r for r in batches if r['batch']=='fixed-viscosity-admission-batch-01'),None)
    if active is not None:
        target['fixed_viscosity_admission']=dict(status=active['status'],active=active.get('active'))
        stage='Corrected-family actual CPU/CUDA qualification and finite managed K1/2/3/4 training/evaluation sequence '+active['status']
    family_models=[]
    for k in (1,2,3,4):
        identifier=f'b{k}-fixed-viscosity';pair=BASE/(identifier+'-pair-01')
        row=dict(identifier=identifier,horizons=k,fit_restore_complete=(pair/'receipt.json').exists())
        if (pair/'active.json').exists():row['managed_run_ids']=read(pair/'active.json')
        result_path=BASE/'fixed-viscosity-evaluation-01'/identifier/'summary.json'
        row['development_complete']=result_path.exists()
        if result_path.exists():
            value=read(result_path);row['source_sha256']=sha(result_path)
            row['all8_accuracy_pass']=all(t['all_accuracy_criteria_pass']for scene in value['scenes']for t in scene['summaries'])
        family_models.append(row)
    target['fixed_viscosity_models']=family_models
    for name in ('fixed-viscosity-k2-precision-evaluation-batch-01','fixed-viscosity-k2-precision-timing-batch-01',
                 'fixed-viscosity-k2-precision-publication-batch-01','fixed-viscosity-replication-admission-batch-01',
                 'fixed-viscosity-replication-admission-batch-02','fixed-viscosity-replication-training-batch-01',
                 'fixed-viscosity-replication-postprocessing-batch-01','fixed-viscosity-replication-execution-batch-01',
                 'fixed-viscosity-replication-timing-batch-01','temporal-final-confirmation-batch-01',
                 'temporal-final-reserved-evaluation-batch-01','temporal-final-reserved-timing-batch-01',
                 'application-domain-change-batch-01'):
        path=BASE/name/'execution.json'
        if path.exists():
            row=read(path)
            batches.append(dict(batch=name,status=row['status'],active=row.get('active'),
                completed_commands=len(row['completed']),execution_sha256=sha(path)))
    family=DOCS/'fixed-viscosity-family-results.json'
    if family.exists():
        value=read(family);target['fixed_viscosity_accuracy']=dict(status=value['status'],qualified=value['accuracy_qualified_all_eight_scenes'],source_sha256=sha(family))
    costs=DOCS/'fixed-viscosity-cost-results.json'
    if costs.exists():
        value=read(costs);target['fixed_viscosity_CPU_cost']=dict(status=value['status'],qualified=value['small_grid_combined_qualified'],source_sha256=sha(costs))
        target['temporal']='All four corrected primary models pass all8development scenes and own240accuracy; K3/K4meet combined small-grid median speed, K2precision comparison precedes finalists, replicas and reserved/application gates'
        stage='Corrected-family four fits, exact CPU restores, accuracy/execution qualification and actual CPU costs complete; assess finalists, replicated control and reserved/application gates'
    float32_path=DOCS/'float32-learned-map-results.json'
    if float32_path.exists():
        value=read(float32_path);target['float32_learned_map']=dict(status=value['status'],accuracy_qualified=value['accuracy_qualified_all_eight_scenes'],
            source_sha256=sha(float32_path),CPU_latency_claim=False,physical_bytes_exact_claim=False)
    float32_cost=DOCS/'float32-map-cost-results.json'
    if float32_cost.exists():
        value=read(float32_cost);point=next(p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240')
        target['float32_map_CPU_cost']=dict(status=value['status'],median_block_ms=point['full_block_ms'],worst_error=point['worst_error'],combined_target_met=point['target_met'],source_sha256=sha(float32_cost))
    corrected_precision=DOCS/'fixed-viscosity-k2-precision-results.json'
    if corrected_precision.exists():
        value=read(corrected_precision)
        target['corrected_K2_precision']=dict(status=value['status'],points=[p for p in value['points']if p['regime']=='small'and p['interval']=='rollout_240'],source_sha256=sha(corrected_precision))
        target['temporal']='All requested K2/3/4 meet all8development-scene accuracy/own240 and combined small-grid median budgets; seed confirmation, reserved scenes and sustained application/event costs remain open'
    selection_path=DOCS/'fixed-viscosity-replication-selection-01.json'
    if selection_path.exists():
        selection=read(selection_path)
        target['replication_selection']=dict(bundled_horizons=selection['recipe_horizons'],
            policies=selection['execution_policy_by_horizon'],one_step_control_included=True,
            primary_seed_reselected=False,source_sha256=sha(selection_path))
        replication=next((row for row in batches if row['batch']=='fixed-viscosity-replication-admission-batch-02'),None)
        if replication is not None:
            stage='Selected K2/K4and B1additional-seed constructor qualification, managed fits, exact CPU restores and full evaluations: '+replication['status']
    replica_registry=BASE/'fixed-viscosity-replication-inputs-01/models.json'
    if replica_registry.exists():
        replica_models=[]
        for entry in read(replica_registry)['models']:
            identifier=entry['identifier'];pair=BASE/(identifier+'-pair-01')
            row=dict(identifier=identifier,horizons=entry['horizons'],replication_ordinal=entry['replication_ordinal'],
                managed_fit_restore_complete=(pair/'receipt.json').exists(),development_complete=False)
            if (pair/'receipt.json').exists():
                value=read(pair/'receipt.json');row.update(artifact_parity=all(value['artifact_parity'].values()),
                    training_run_id=value['training_run_id'],restore_run_id=value['restore_run_id'],
                    pair_receipt_sha256=sha(pair/'receipt.json'))
            if (pair/'active.json').exists():row['managed_run_ids']=read(pair/'active.json')
            evaluation=BASE/'fixed-viscosity-replication-evaluation-01'/identifier/'summary.json'
            if evaluation.exists():
                value=read(evaluation);row.update(development_complete=value['status']=='completed'and len(value['scenes'])==8,
                    all_eight_scene_accuracy_pass=all(s['all_accuracy_criteria_pass']for scene in value['scenes']for s in scene['summaries']),
                    evaluation_sha256=sha(evaluation),selected_execution_policy_qualification_separate=True)
            replica_models.append(row)
        target['replica_models']=replica_models
    pending=next((row for row in batches if row['batch']=='float32-map-timing-batch-01'and row['status']=='running'),None)
    for key,name in (('replica_accuracy','fixed-viscosity-replication-results.json'),
                     ('replica_cost','fixed-viscosity-replication-cost-results.json'),
                     ('reserved_accuracy','temporal-reserved-accuracy-results.json'),
                     ('reserved_cost','temporal-reserved-cost-results.json'),
                     ('application','application-results/results.json'),
                     ('domain_change','application-domain-change-results.json')):
        path=DOCS/name
        if path.exists():target[key]=dict(status=read(path)['status'],source_sha256=sha(path),evidence=str(path.relative_to(BASE.parent.parent)))
    if pending is not None:
        target['CPU_timing_order']='Existing four-fit family, exact execution proofs and CPU comparison first; float32 comparison next; defer new numerical jobs until both timing campaigns reap'
    application=DOCS/'application-preparation-results.json'
    if application.exists():
        value=read(application)
        target['application_preparation']=dict(status=value['status'],source_sha256=sha(application),
            transport_bitwise_states=value['bitwise_states_compared'],mechanics_cases=value['actual_neural_native_mechanics_cases'],
            event_reference_scenes=value['event_reference_scenes'],independent_other_K_fit_claim=False,
            application_speed_claim=False,reserved_scenes_consumed=False)
    write(BASE/'work-state.json',dict(status='active',stage=stage,
        original_goal_complete=False,requirements=29,updated_unix=time.time(),batches=batches,
        live_workers=[row for row in batches if row['status']=='running'],target_status=target,
        original_registration_sha256=sha(DOCS/'registration.json'),source_sha256=sha(__file__),
        no_automatic_retry=True,model_promotion=False))
    print('Refreshed durable study state',flush=True)
