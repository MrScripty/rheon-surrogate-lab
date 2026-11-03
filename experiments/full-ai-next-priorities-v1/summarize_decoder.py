"""Audit the full-state pressure-expert follow-up without single-factor claims."""
import numpy as np
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-decoder-results.json'
    if output.exists():raise ValueError('Fresh completed full-state result required')
    candidate=BASE/'b2-endpoint-decoder-pair-01';control=BASE/'b2-endpoint-pressure-expert-pair-01'
    pair=read(candidate/'receipt.json');project=ROOT/pair['source_project'];assembly=read(project/'assembly.json')
    if pair['status']!='completed'or sha(project/'assembly.json')!=pair['source_assembly_sha256']:
        raise ValueError('Complete immutable managed pair required')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured expert source changed')
    trained=read(candidate/'train/receipt.json');restored=read(candidate/'restore/receipt.json')
    history=read(candidate/'restore/history.json');old_history=read(control/'restore/history.json')
    old_project=ROOT/read(control/'receipt.json')['source_project'];old_config=read(old_project/'config.json')
    config=assembly['config'];initial=history['initial_inventory'];final=history['final_inventory']
    checks=dict(actual_10000_updates=trained['optimizer_updates']==10000,exact_CPU_artifact_parity=all(pair['artifact_parity'].values()),
        zero_update_CPU_restore=restored['optimizer_updates']==0 and restored['full_cpu_probe_bitwise_verified'],
        CPU_cuda_initialized_false=not restored['execution']['cuda_initialized'],
        same_full_state_data=read(old_project/'assembly.json')['static_inputs']==assembly['static_inputs'],
        same_sample_stream=history['sample_order_sha256']==old_history['sample_order_sha256'],
        same_counted_block_state_work=history['optimizer_work']==old_history['optimizer_work'],
        frozen_coefficients_unchanged=all(initial[name]==final[name]==spec for name,spec in trained['frozen_parameter_inventory'].items()),
        exact_source_closure=trained['closure']==restored['closure'],
        actual_one_expert_per_block=trained['pressure_expert_calls_per_block']==1)
    for key in ('horizons','width','architecture','supervision_mode','updates','single_block_updates','learning_rate','schedule','seed',
                'sampling_seed','feedback_protocol','physical_context','step_scale_statistics_sha256','original_normalization_sha256'):
        checks['config_'+key]=config[key]==old_config[key]
    for name in ('temporal_objective.py','temporal_objective_increment.py','temporal_feedback.py','temporal_data.py',
                 'temporal_codec.py','temporal_features_v2.py','step-scales.json'):
        checks['source_'+name]=sha(old_project/name)==sha(project/name)
    provenance=read(project/'expert-assets/request-initialization.json')
    with np.load(candidate/'restore/initial-weights.npz',allow_pickle=False)as actual,np.load(project/'expert-assets/request-weights.npz',allow_pickle=False)as parent:
        checks['managed_request_initialization_exact']=all(np.array_equal(actual['request.'+name].view('<u4'),parent[name].view('<u4'))for name in parent.files)
    common=set(initial)&set(old_history['initial_inventory'])
    checks['shared_seeded_fluid_request_tensors']=all(initial[name]==old_history['initial_inventory'][name]for name in common)
    checks['identical_inherited_expert_assets']=config['expert_asset_sha256']==old_config['expert_asset_sha256']
    trainer=(project/'temporal_train_decoder.py').read_text().replace('from bundle_model_decoder import BundleNet','from bundle_model_pressure_expert import BundleNet')
    checks['training_loop_differs_only_in_model_import']=trainer==(old_project/'temporal_train_pressure_expert.py').read_text()
    if not all(checks.values()):
        write(BASE/'temporal-decoder-control-audit-failure-01.json',dict(status='failed',checks=checks))
        raise ValueError('Declared full-state follow-up controls failed')
    paths=[BASE/'temporal-pressure-expert-evaluation-01/b2-endpoint-pressure-expert/summary.json',
           BASE/'temporal-decoder-evaluation-01/b2-endpoint-decoder/summary.json']
    evaluations=[read(path)for path in paths]
    if any(value['status']!='completed'or len(value['scenes'])!=8 for value in evaluations):raise ValueError('All eight development scenes required')
    if len({value['bank_sha256']for value in evaluations})!=1 or len({value['metrics_sha256']for value in evaluations})!=1:
        raise ValueError('Comparison evaluation population or metrics differ')
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            arms=[]
            for value in evaluations:
                scenes=[scene for scene in value['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                worst={field:max(scene['summaries'][index]['worst'][field]for scene in scenes)for field in ('pressure','equation','velocity','tracer')}
                arms.append(dict(worst=worst,accuracy_pass=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes),
                    complete=all(scene['summaries'][index]['complete']for scene in scenes)))
            comparisons.append(dict(regime=regime,interval=interval,expert_control=arms[0],decoder=arms[1]))
    diagnosis=BASE/'temporal-decoder-fitting-diagnosis-01.json'
    factorization_path=BASE/'temporal-decoder-factorization-diagnosis-01.json'
    factorization=read(factorization_path)
    if factorization['status']!='completed'or not factorization['training_queries_only']or factorization['optimizer_updates']!=0:
        raise ValueError('Complete unchanged-candidate train-only diagnostics required')
    evidence={str(path.relative_to(ROOT)):sha(path)for path in (*paths,diagnosis,candidate/'receipt.json',
        candidate/'restore/receipt.json',candidate/'restore/history.json',project/'assembly.json',factorization_path)}
    passed=all(row['decoder']['accuracy_pass']and row['decoder']['complete']for row in comparisons if row['regime']=='small')
    write(output,dict(status='completed',comparisons=comparisons,fitting=read(diagnosis)['summaries'],
        accuracy_qualified_small_grid_recipe=passed,CPU_latency_claim=False,control_audit=dict(status='passed',checks=checks),
        model=dict(parameters=trained['parameters'],trainable_parameters=trained['trainable_parameters'],
            parent_pressure_learned_parameters=trained['parent_pressure_learned_parameters'],pressure_expert_calls_per_block=1),
        inherited_request_training=provenance,inherited_training_equal_immediate_control=True,inherited_training_additional_relative_original_generic_models=True,
        offline_factorization_diagnosis=dict(summaries=factorization['summaries'],lead_time_readout_summaries=factorization['lead_time_readout_summaries'],
            worst_zero_current_tracer_predicted_absolute_rms=max(row['zero_current_tracer_predicted_absolute_rms']for row in factorization['rows']),
            worst_pressure_gauge_shift_velocity_relative_change=max(row['pressure_gauge_shift_velocity_relative_change']for row in factorization['rows']),
            candidate_frontier_point=False),
        tracer_initialization_and_decoder_representation_differ=True,single_factor_attribution=False,
        source_evidence=evidence,source_sha256=sha(__file__),registration_sha256=sha(DOCS/'temporal-decoder-registration-01.json'),
        candidate_numerical_pressure_solve=False,candidate_velocity_post_projection=False,
        reserved_scenes_consumed=False,model_promotion=False,original_goal_complete=False))
    lines=['# Lead-time, gauge and passive-tracer decoder comparison','',
        'One bounded K2 endpoint fit changes pressure gauge/lead-time conditioning and the passive-tracer readout. '
        'The control has identical inherited pressure/request assets, shared seeded fluid/request tensors, data, objective, sampled stream, schedule and feedback. '
        'Tracer parameters and its initial function differ; arithmetic work is not FLOP-matched. This combined representation comparison does not isolate a single cause.','',
        '| Grids | Interval | Field | Expert control | Decoder |','| --- | --- | --- | ---: | ---: |']
    for row in comparisons:
        for field in ('pressure','equation','velocity','tracer'):
            lines.append(f"| {row['regime']} | {row['interval']} | {field} | {row['expert_control']['worst'][field]:.6g} | {row['decoder']['worst'][field]:.6g} |")
    lines.extend(['','Small-grid full-state accuracy-qualified: '+str(passed)+'.',
        'The exact CPU restore and declared comparison-control audit pass. Frozen learned coefficients remain unchanged. '
        'CPU latency remains unmeasured for these new weights, and previous component/model timings cannot establish a complete-block claim. '
        'Reserved scenes remain unopened. The original full-state and own-history objectives remain active.','',
        'Predeclared train-only controls separately replace anchor pressure with native truth, shift its arbitrary gauge, '
        'and probe zero current tracer. These controls make zero updates, never enter candidate inference and are excluded from frontiers. '
        'Their complete metrics and pressure-readout lead-time alignments are included in the machine-readable result.','',
        '[All errors, provenance and controls](temporal-decoder-results.json)',''])
    (DOCS/'temporal-decoder-results.md').write_text('\n'.join(lines))
    print('Completed combined decoder comparison audit; accuracy qualified:',passed,flush=True)


if __name__=='__main__':run()
