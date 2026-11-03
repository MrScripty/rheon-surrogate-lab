"""Audit learned displacement custody and every-output/own240 accuracy."""
import numpy as np
from summarize_coupled import aggregate,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run():
    output=DOCS/'temporal-displacement-results.json'
    if output.exists():raise ValueError('Fresh complete displacement result required')
    candidate=BASE/'b2-displacement-pair-01';control=BASE/'b2-coupled-pair-01'
    pair=read(candidate/'receipt.json');project=ROOT/pair['source_project'];assembly=read(project/'assembly.json')
    prior_project=ROOT/read(control/'receipt.json')['source_project'];prior=read(prior_project/'assembly.json')
    trained=read(candidate/'train/receipt.json');restored=read(candidate/'restore/receipt.json')
    history=read(candidate/'restore/history.json');prior_history=read(control/'restore/history.json')
    registration_path=DOCS/'temporal-displacement-fit-registration-01.json';registration=read(registration_path)
    config=assembly['config'];initial=history['initial_inventory'];final=history['final_inventory'];work=history['optimizer_work']
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured full-state source changed')
    for name,digest in registration['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered source changed')
    checks=dict(completed_managed_pair=pair['status']=='completed',actual_10000_updates=trained['optimizer_updates']==10000,
        exact_CPU_artifact_parity=all(pair['artifact_parity'].values()),zero_update_CPU_restore=restored['optimizer_updates']==0,
        actual_CPU_probe_bitwise=restored['full_cpu_probe_bitwise_verified'],CPU_cuda_initialized_false=not restored['execution']['cuda_initialized'],
        source_assembly_unchanged=sha(project/'assembly.json')==pair['source_assembly_sha256'],
        source_config_registered=sha(project/'config.json')==registration['config_sha256'],
        exact_training_restore_closure=trained['closure']==restored['closure'],
        restore_origin_matches=restored['originating_training_run']['run_id']==pair['training_run_id'],
        same_full_state_data=assembly['static_inputs']==prior['static_inputs'],same_sample_stream=history['sample_order_sha256']==prior_history['sample_order_sha256'],
        frozen_coefficients_unchanged=all(initial[name]==final[name]==spec for name,spec in trained['frozen_parameter_inventory'].items()),
        counted_shared_encoder=work['encoder_calls']==work['network_forwards'],
        counted_every_physical_state=work['predicted_physical_states']==work['network_forwards']*2,
        counted_internal_updates=work['latent_updates']==work['network_forwards']*2,
        counted_internal_history=work['internal_history_updates']==work['network_forwards'],
        counted_fused_velocity_projection=work['velocity_feature_calls']==work['network_forwards']*2,
        counted_displacement_readouts=work['displacement_readout_calls']==work['network_forwards']*2,
        counted_pressure_readouts=work['pressure_readout_calls']==work['network_forwards']*2,
        counted_flux_readouts=work['flux_readout_calls']==work['network_forwards']*2,
        counted_momentum_readouts=work['momentum_readout_calls']==work['network_forwards']*2,
        registered_auxiliary_objective=config['displacement_auxiliary_weight']==1.,registered_architecture=config['architecture']=='displacement')
    for label,key in (('train','training_run_id'),('restore','restore_run_id')):
        terminal=read(candidate/f'terminal-{pair[key]}.json')
        checks[label+'_completed_drained']=terminal['state']=='completed'and terminal['custody_phase']=='drained'
    for key in ('network_forwards','encoder_cells','predicted_physical_states','encoder_calls'):
        checks['same_target_work_'+key]=work[key]==prior_history['optimizer_work'][key]
    for key in ('horizons','updates','single_block_updates','sampling_seed','feedback_protocol','physical_context',
        'physical_scales','step_scale_statistics_sha256','original_normalization_sha256','width'):
        checks['same_'+key]=config[key]==prior['config'][key]
    for name in ('temporal_objective.py','temporal_objective_increment.py','temporal_objective_mse.py','temporal_feedback.py',
        'temporal_data.py','temporal_codec.py','temporal_features_v2.py','step-scales.json'):
        checks['source_'+name]=sha(project/name)==sha(prior_project/name)
    for path in sorted((project/'expert-assets/pressure').rglob('*')):
        if path.is_file():checks['inherited_pressure_'+str(path.relative_to(project/'expert-assets'))]=sha(path)==sha(prior_project/'expert-assets'/path.relative_to(project/'expert-assets'))
    with np.load(candidate/'restore/initial-weights.npz',allow_pickle=False)as actual,np.load(project/'expert-assets/point-weights.npz',allow_pickle=False)as point:
        checks['calibrated_points_exact']=np.array_equal(actual['displacement.point_weight'].view('<u4'),point['projection.weight'].view('<u4'))
    if not all(checks.values()):
        write(BASE/'temporal-displacement-control-audit-failure-01.json',dict(status='failed',checks=checks))
        raise ValueError('Declared full-state comparison controls failed')
    paths=[BASE/'temporal-coupled-evaluation-01/b2-coupled/summary.json',BASE/'temporal-displacement-evaluation-01/b2-displacement/summary.json']
    evaluations=[read(p)for p in paths]
    if any(e['status']!='completed'or len(e['scenes'])!=8 for e in evaluations):raise ValueError('All eight development scenes required')
    if len({e['bank_sha256']for e in evaluations})!=1 or len({e['metrics_sha256']for e in evaluations})!=1:
        raise ValueError('Evaluation population or error criteria changed')
    value=evaluations[1];identity=value['model']
    if identity['weights_sha256']!=sha(candidate/'restore/weights.npz')or identity['inference_source_sha256']!=sha(HERE/'bundle_model_displacement.py'):
        raise ValueError('Evaluated weights or actual inference identity differs')
    if value['source_sha256']!=registration['evaluation_implementation_sha256']or identity['architecture']!='displacement':
        raise ValueError('Qualified evaluation core or actual model differs')
    loader_note=read(DOCS/'temporal-displacement-evaluation-loader-note-01.json')
    if value['loader_sha256']!=loader_note['legacy_loader_sha256']or sha(HERE/'displacement_evaluation_support.py')!=loader_note['actual_loader_sha256']:
        raise ValueError('Explicit inherited/actual loader provenance differs')
    raw=[]
    for scene in value['scenes']:
        path=paths[1].parent/(scene['identifier']+'.json');record=read(path);raw.append(path)
        single=[r for r in record['rows']if r['mode']=='single_block']
        if len(single)!=121*2 or {(r['start'],r['horizon'])for r in single}!={(s,h)for s in range(121)for h in (1,2)}:
            raise ValueError('Every single-block output required')
        own=[r for r in record['rows']if r['mode']=='own_history']
        if scene['first_unavailable_step']is None and [r['step']for r in own]!=list(range(1,241)):
            raise ValueError('Complete own-history output sequence required')
    comparisons=[]
    for regime in ('small','32cubed'):
        for i,label in enumerate(('single_block','own_history12','own_history48','own_history240')):
            comparisons.append(dict(regime=regime,interval=label,coupled_control=aggregate(evaluations[0],regime,i),displacement=aggregate(value,regime,i)))
    fitting_path=BASE/'temporal-displacement-fitting-diagnosis-01.json';fitting=read(fitting_path)
    if fitting['status']!='completed'or fitting['optimizer_updates']!=0 or fitting['model']['weights_sha256']!=identity['weights_sha256']:
        raise ValueError('Complete unchanged-weight fitting diagnosis required')
    passed=all(c['displacement']['accuracy_pass']for c in comparisons if c['regime']=='small')
    evidence_paths=[*paths,*raw,candidate/'receipt.json',candidate/'train/receipt.json',candidate/'restore/receipt.json',candidate/'restore/history.json',
        candidate/'restore/weights.npz',project/'assembly.json',registration_path,DOCS/'temporal-displacement-evaluation-protocol-01.json',
        DOCS/'temporal-displacement-evaluation-loader-note-01.json',DOCS/'temporal-displacement-qualification-results.json',fitting_path]
    write(output,dict(status='completed',comparisons=comparisons,control_audit=checks,parameters=trained['parameters'],
        trainable_parameters=trained['trainable_parameters'],optimizer_work=work,accuracy_qualified_small_grid_recipe=passed,
        fitting=fitting['summaries'],source_evidence={str(p.relative_to(ROOT)):sha(p)for p in evidence_paths},reporter_sha256=sha(__file__),
        changes_vs_coupled=['displacement representation','curvature auxiliary supervision','fresh seed','learning-rate schedule'],
        single_factor_attribution=False,matched_FLOPs_claim=False,CPU_latency_claim=False,
        actual_loader_sha256=loader_note['actual_loader_sha256'],raw_core_legacy_loader_stamp_preserved=True,
        full_interval_worst_requires_complete_outputs=True,source_normalization_and_criteria_unchanged=True,
        reserved_scenes_consumed=False,original_goal_complete=False,
        next_decision='Measure actual CPU cost, then extend/replicate qualifying full-state configurations'if passed else'Use errors and fitting diagnosis to select the next bounded representation/objective comparison'))
    lines=['# Learned displacement full-state results','',
        'The K2 model completes 10,000 managed updates, an exact CPU restore and every development output. '
        'Data, sampling and physical target work match the coupled control. Representation, auxiliary objective, '
        'initialization seed and learning-rate schedule differ, so the comparison does not isolate one cause.','',
        '| Regime | Interval | Complete | Worst P / equation / velocity / tracer | Accuracy pass |',
        '| --- | --- | --- | --- | --- |']
    for c in comparisons:
        arm=c['displacement'];text=' / '.join(f'{arm["worst"][f]:.7g}'for f in FIELDS)if arm['complete']else'Unavailable; incomplete rollout'
        lines.append(f'| {c["regime"]} | {c["interval"]} | {arm["complete"]} | {text} | {arm["accuracy_pass"]} |')
    lines.extend(['','Incomplete trajectories retain their raw finite-prefix errors and failures, without a complete-interval pass. '
        'CPU latency is not established by this accuracy report. The eight reserved scenes remain unopened.','',
        'The unchanged evaluation core runs with the registered displacement loader. Its raw legacy loader stamp '
        'is preserved separately; actual loader, model source and restored weights are bound in the evidence.','',
        '[Results, controls and source evidence](temporal-displacement-results.json) · '
        '[Implemented model](temporal-displacement-plan.md) · [Original objectives](plan.md)',''])
    (DOCS/'temporal-displacement-results.md').write_text('\n'.join(lines))
    print('Published displacement full-state comparison; small-grid accuracy qualified:',passed,flush=True)


if __name__=='__main__':run()
