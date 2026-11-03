"""Audit a generative velocity representation against its direct-increment control."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-curl-results.json'
    if output.exists():raise ValueError('Fresh completed representation comparison required')
    control=BASE/'b2-endpoint-mse-increment-pair-01';candidate=BASE/'b2-endpoint-curl-increment-pair-01'
    a=read(control/'receipt.json');b=read(candidate/'receipt.json')
    old_project=ROOT/a['source_project'];new_project=ROOT/b['source_project']
    old=read(control/'restore/history.json');new=read(candidate/'restore/history.json')
    ac=read(old_project/'config.json');bc=read(new_project/'config.json')
    checks=dict(exact_CPU_artifact_parity=all(b['artifact_parity'].values()),parameter_count=a['parameters']==b['parameters']==163068,
        seeded_initial_tensors=old['initial_inventory']==new['initial_inventory'],sampling=old['sample_order_sha256']==new['sample_order_sha256'],
        counted_work=old['optimizer_work']==new['optimizer_work'],
        static_inputs=read(old_project/'assembly.json')['static_inputs']==read(new_project/'assembly.json')['static_inputs'])
    for key in ('horizons','width','architecture','supervision_mode','updates','single_block_updates','learning_rate','schedule','seed',
                'sampling_seed','feedback_protocol','physical_context','step_scale_statistics_sha256','original_normalization_sha256'):
        checks['config_'+key]=ac[key]==bc[key]
    for name in ('bundle_model_step_scaled.py','learned_global.py','temporal_features_v2.py','temporal_data.py','temporal_codec.py',
                 'temporal_objective.py','temporal_objective_increment.py','temporal_feedback.py','step-scales.json'):
        checks['source_'+name]=sha(old_project/name)==sha(new_project/name)
    trainer=(new_project/'temporal_train_curl.py').read_text().replace('from bundle_model_curl import BundleNet','from bundle_model_step_scaled import BundleNet')
    checks['training_loop_differs_only_in_model_import']=trainer==(old_project/'temporal_train_increment_supervision.py').read_text()
    if not all(checks.values()):
        write(BASE/'temporal-curl-control-audit-failure-01.json',dict(status='failed',checks=checks))
        raise ValueError('Representation comparison controls differ')
    paths=[BASE/'temporal-increment-supervision-development-evaluation-01/b2-endpoint-mse-increment/summary.json',
           BASE/'temporal-curl-development-evaluation-01/b2-endpoint-curl-increment/summary.json']
    evaluations=[read(path)for path in paths]
    if len({value['bank_sha256']for value in evaluations})!=1 or len({value['metrics_sha256']for value in evaluations})!=1:
        raise ValueError('Representation comparison evaluation changed')
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            values=[]
            for value in evaluations:
                scenes=[scene for scene in value['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                worst={field:max(scene['summaries'][index]['worst'][field]for scene in scenes)for field in ('pressure','equation','velocity','tracer')}
                values.append(dict(worst=worst,accuracy_pass=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes)))
            comparisons.append(dict(regime=regime,interval=interval,direct_increment=values[0],curl_increment=values[1]))
    physics=[]
    for family,path in zip(('direct_increment','curl_increment'),paths):
        scene_paths=[path.parent/(scene['identifier']+'.json')for scene in read(path)['scenes']]
        rows=[row for scene in scene_paths for row in read(scene)['rows']]
        for mode in ('single_block','own_history'):
            selected=[row for row in rows if row['mode']==mode]
            physics.append(dict(family=family,mode=mode,outputs=len(selected),
                worst_predicted_divergence_rms=max(row['metrics']['predicted_physics']['divergence_rms']for row in selected),
                maximum_wall_normal_velocity=max(row['metrics']['predicted_physics']['wall_normal_velocity_max']for row in selected)))
    diagnosis=BASE/'temporal-curl-fitting-diagnosis-b2-endpoint-curl-increment-01.json'
    passed=all(row['curl_increment']['accuracy_pass']for row in comparisons if row['regime']=='small')
    write(output,dict(status='completed',comparisons=comparisons,physics_diagnostics=physics,fitting=read(diagnosis)['summaries'],
        accuracy_qualified_small_grid_recipe=passed,CPU_latency_claim=False,control_audit=dict(status='passed',checks=checks),
        source_evidence={str(path.relative_to(ROOT)):sha(path)for path in (*paths,diagnosis)},source_sha256=sha(__file__),
        registration_sha256=sha(DOCS/'temporal-curl-registration-01.json'),fixed_linear_curl_readout_inside_model=True,
        candidate_numerical_pressure_solve=False,candidate_velocity_post_projection=False,
        reserved_scenes_consumed=False,model_promotion=False,original_goal_complete=False))
    lines=['# Generative curl velocity readout comparison','',
        'Both models share initial tensors, training data, sampled windows, optimizer schedule, increment-MSE loss and two-block own-history curriculum. '
        'The candidate interprets three learned output channels as edge potentials and reconstructs velocity increments with a fixed linear curl readout. '
        'It adds no trainable parameters. Extra readout arithmetic is counted as model work; FLOPs are not matched. '
        'Pressure, tracer and all intermediate states remain required.','',
        '| Grids | Interval | Field | Direct increment | Curl increment |','| --- | --- | --- | ---: | ---: |']
    for row in comparisons:
        for field in ('pressure','equation','velocity','tracer'):
            lines.append(f"| {row['regime']} | {row['interval']} | {field} | {row['direct_increment']['worst'][field]:.6g} | {row['curl_increment']['worst'][field]:.6g} |")
    lines.extend(['','Small-grid full-state accuracy-qualified: '+str(passed)+'.',
        'The representation constrains increments, not pre-existing input divergence. Float32 interchange adds roundoff. '
        'Constraint satisfaction alone is not prediction accuracy. CPU latency remains unmeasured for the new weights; '
        'no reserved scenes were consumed and no model is promoted.','',
        '[Complete metrics, physical diagnostics and control audit](temporal-curl-results.json)',''])
    (DOCS/'temporal-curl-results.md').write_text('\n'.join(lines))
    print('Completed curl representation comparison; small-grid full-state accuracy qualified:',passed,flush=True)
