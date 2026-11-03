"""Audit the coupled full-state fit, retaining incomplete rollout failures."""
import numpy as np
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


FIELDS=('pressure','equation','velocity','tracer')


def aggregate(evaluation,regime,index):
    scenes=[s for s in evaluation['scenes']if(max(s['counts_xyz'])<=16)==(regime=='small')]
    summaries=[s['summaries'][index]for s in scenes]
    complete=all(s['complete']for s in summaries)
    observed={field:max(s['worst'][field]for s in summaries)for field in FIELDS}
    return dict(complete=complete,accuracy_pass=complete and all(s['all_accuracy_criteria_pass']for s in summaries),
        worst=observed if complete else None,observed_prefix_worst=observed,
        worst_scope='Complete declared interval'if complete else'Observed finite prefix only',
        outputs=sum(s['outputs']for s in summaries),scenes=len(scenes))


def run():
    output=DOCS/'temporal-coupled-results.json'
    if output.exists():raise ValueError('Fresh completed coupled result required')
    candidate=BASE/'b2-coupled-pair-01';control=BASE/'b2-endpoint-decoder-pair-01'
    pair=read(candidate/'receipt.json');project=ROOT/pair['source_project'];assembly=read(project/'assembly.json')
    if pair['status']!='completed'or sha(project/'assembly.json')!=pair['source_assembly_sha256']:
        raise ValueError('Complete immutable managed pair required')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured source changed')
    trained=read(candidate/'train/receipt.json');restored=read(candidate/'restore/receipt.json')
    history=read(candidate/'restore/history.json');old_history=read(control/'restore/history.json')
    old_project=ROOT/read(control/'receipt.json')['source_project'];old_assembly=read(old_project/'assembly.json')
    config=assembly['config'];old_config=old_assembly['config']
    initial=history['initial_inventory'];final=history['final_inventory'];work=history['optimizer_work']
    checks=dict(actual_10000_updates=trained['optimizer_updates']==10000,
        exact_CPU_artifact_parity=all(pair['artifact_parity'].values()),
        zero_update_CPU_restore=restored['optimizer_updates']==0 and restored['full_cpu_probe_bitwise_verified'],
        CPU_cuda_initialized_false=not restored['execution']['cuda_initialized'],
        same_full_state_data=old_assembly['static_inputs']==assembly['static_inputs'],
        same_sample_stream=history['sample_order_sha256']==old_history['sample_order_sha256'],
        frozen_coefficients_unchanged=all(initial[name]==final[name]==spec for name,spec in trained['frozen_parameter_inventory'].items()),
        exact_source_closure=trained['closure']==restored['closure'],
        actual_K_pressure_readouts_per_block=trained['pressure_expert_calls_per_block']==config['horizons'],
        counted_pressure_readouts=work['pressure_readout_calls']==work['network_forwards']*config['horizons'],
        counted_shared_encoder=work['encoder_calls']==work['network_forwards'],
        counted_internal_updates=work['latent_updates']==work['network_forwards']*config['horizons'],
        counted_internal_history=work['internal_history_updates']==work['network_forwards']*(config['horizons']-1),
        registered_coupled_width=config['width']==read(DOCS/'temporal-coupled-registration-01.json')['width'])
    for key in ('network_forwards','encoder_cells','predicted_physical_states','encoder_calls'):
        checks['same_target_work_'+key]=work[key]==old_history['optimizer_work'][key]
    for key in ('horizons','supervision_mode','updates','single_block_updates','learning_rate','schedule','seed',
                'sampling_seed','feedback_protocol','physical_context','step_scale_statistics_sha256','original_normalization_sha256'):
        checks['config_'+key]=config[key]==old_config[key]
    for name in ('temporal_objective.py','temporal_objective_increment.py','temporal_objective_mse.py','temporal_feedback.py',
                 'temporal_data.py','temporal_codec.py','temporal_features_v2.py','step-scales.json'):
        checks['source_'+name]=sha(old_project/name)==sha(project/name)
    for path in sorted((project/'expert-assets/pressure').rglob('*')):
        if path.is_file():
            relative=path.relative_to(project/'expert-assets')
            checks['inherited_pressure_'+str(relative)]=sha(path)==sha(old_project/'expert-assets'/relative)
    with np.load(candidate/'restore/initial-weights.npz',allow_pickle=False)as actual,np.load(project/'expert-assets/request-weights.npz',allow_pickle=False)as parent:
        checks['frozen_calibrated_flux_exact']=np.array_equal(actual['flux.weight'].view('<u4'),parent['flux.weight'].view('<u4'))
    if not all(checks.values()):
        write(BASE/'temporal-coupled-control-audit-failure-01.json',dict(status='failed',checks=checks))
        raise ValueError('Declared coupled comparison controls failed')
    paths=[BASE/'temporal-decoder-evaluation-01/b2-endpoint-decoder/summary.json',
        BASE/'temporal-coupled-evaluation-01/b2-coupled/summary.json']
    evaluations=[read(path)for path in paths]
    if any(v['status']!='completed'or len(v['scenes'])!=8 for v in evaluations):raise ValueError('All eight development scenes required')
    if len({v['bank_sha256']for v in evaluations})!=1 or len({v['metrics_sha256']for v in evaluations})!=1:
        raise ValueError('Evaluation population or criteria changed')
    protocol=read(BASE/'temporal-coupled-evaluator-qualification-01/receipt.json')
    if protocol['status']!='passed'or protocol['evaluator_sha256']!=evaluations[1]['source_sha256']:
        raise ValueError('Actual failure-aware evaluator differs from qualified source')
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            comparisons.append(dict(regime=regime,interval=interval,decoder_control=aggregate(evaluations[0],regime,index),
                coupled=aggregate(evaluations[1],regime,index)))
    diagnosis=BASE/'temporal-coupled-fitting-diagnosis-01.json';fitting=read(diagnosis)
    if fitting['status']!='completed'or fitting['optimizer_updates']!=0 or fitting['reserved_scenes_consumed']:
        raise ValueError('Complete unchanged-model fixed-window diagnostics required')
    passed=all(row['coupled']['accuracy_pass']for row in comparisons if row['regime']=='small')
    evidence_paths=[*paths,diagnosis,candidate/'receipt.json',candidate/'train/receipt.json',candidate/'restore/receipt.json',
        candidate/'restore/history.json',project/'assembly.json',DOCS/'temporal-coupled-admission-01.json',
        DOCS/'temporal-coupled-registration-01.json',DOCS/'temporal-coupled-evaluation-protocol-01.json',
        BASE/'temporal-coupled-evaluator-qualification-01/receipt.json',
        BASE/'temporal-coupled-control-audit-failure-01.json',BASE/'temporal-coupled-control-audit-failure-01.source.py',
        BASE/'temporal-coupled-control-audit-failure-01.note.json']
    evidence={str(p.relative_to(ROOT)):sha(p)for p in evidence_paths}
    failures=[dict(identifier=s['identifier'],first_unavailable_step=s.get('first_unavailable_step'),
        failures=s.get('inference_failures',[]))for s in evaluations[1]['scenes']if s.get('first_unavailable_step')is not None]
    write(output,dict(status='completed',comparisons=comparisons,fitting=fitting['summaries'],
        accuracy_qualified_small_grid_recipe=passed,CPU_latency_claim=False,control_audit=checks,
        parameters=trained['parameters'],trainable_parameters=trained['trainable_parameters'],
        optimizer_work=work,own_history_failures=failures,source_evidence=evidence,
        architecture_widths=dict(decoder_control=old_config['width'],coupled=config['width']),
        coupled_representation='Shared learned transport/flux/pressure/momentum with entering-velocity neural tracer readout',
        full_interval_worst_requires_complete_outputs=True,single_factor_attribution=False,matched_FLOPs_claim=False,
        inherited_training='Frozen pressure expert and1000-update neural flux calibration; nonlinear request weights unused',
        reserved_scenes_consumed=False,model_promotion=False,original_goal_complete=False,
        next_decision='Measure complete CPU block latency and qualify K3/K4 follow-ups'if passed else'Inspect fixed-window fitting and raw rollouts before another registered dynamics change',
        source_sha256=sha(__file__)))
    lines=['# Coupled full-state generation comparison','',
        'The K2 coupled fit completes 10,000 updates and an exact zero-update CPU restore. '
        'The data, sample stream, supervision and counted target work match the decoder control. '
        f'Architecture widths differ ({old_config["width"]} for the decoder and {config["width"]} for the coupled model), '
        'as do internal arithmetic and parameter count; this does not isolate a single cause.','',
        '| Regime | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- | --- |']
    for row in comparisons:
        arm=row['coupled'];text=' / '.join(f'{arm["worst"][f]:.7g}'for f in FIELDS)if arm['complete']else'Unavailable; incomplete rollout'
        lines.append(f'| {row["regime"]} | {row["interval"]} | {arm["complete"]} | {text} | {arm["accuracy_pass"]} |')
    lines.extend(['','All eight development scenes are evaluated. Any failed block returns no fabricated states; '
        'its observed finite-prefix maximum is retained separately and cannot establish a complete-interval worst error. '
        'See the machine-readable result and raw scene records.','',
        f'The model has {trained["parameters"]:,} parameters, of which {trained["trainable_parameters"]:,} are trainable. '
        'It encodes once per block and counts every internal latent, neural pressure, flux, momentum and tracer readout. '
        'Dense learned pressure maps have separate setup and memory costs and use the original component on 32³.','',
        'No new CPU latency is claimed. Complete-block timing and all requested horizons remain separate qualification gates. '
        'The eight reserved temporal scenes remain unopened.','',
        'The first reporting audit incorrectly required equal architecture widths. It stopped before writing results. '
        'The corrected audit checks the coupled width against its unchanged preregistration and reports the difference. '
        'Both attempts are retained; no fit, restore or evaluation was repeated.','',
        '![Every-step own-history errors against the previous decoder](figures/temporal-coupled-rollout.png)','',
        '[Results and source hashes](temporal-coupled-results.json) · [Registered comparison](temporal-coupled-registration-01.json) · '
        '[Previous decoder](temporal-decoder-results.md) · [Original plan](plan.md)',''])
    (DOCS/'temporal-coupled-results.md').write_text('\n'.join(lines))
    print('Completed coupled full-state comparison; accuracy qualified:',passed,flush=True)


if __name__=='__main__':run()
