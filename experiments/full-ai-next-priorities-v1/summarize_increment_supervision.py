"""Compare matched MSE arms and the previous log control on unchanged errors."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-increment-supervision-results.json'
    if output.exists():raise ValueError('Fresh completed supervision comparison required')
    control=BASE/'b2-endpoint-step-scaled-pair-01';previous=read(control/'restore/history.json')
    control_project=ROOT/read(control/'receipt.json')['source_project'];config=read(control_project/'config.json')
    audits=[];evaluations=[];sources={};fitting=[]
    identities=[('log_control',control,BASE/'temporal-step-scaled-development-evaluation-01/b2-endpoint-step-scaled/summary.json')]
    for mode,suffix in (('full_state','mse-fullstate'),('increments','mse-increment')):
        identifier='b2-endpoint-'+suffix;pair=BASE/f'{identifier}-pair-01'
        receipt=read(pair/'receipt.json');project=ROOT/receipt['source_project']
        current=read(pair/'restore/history.json');candidate_config=read(project/'config.json')
        checks=dict(exact_CPU_artifact_parity=all(receipt['artifact_parity'].values()),
            parameter_count=receipt['parameters']==163068,
            initial_tensors=current['initial_inventory']==previous['initial_inventory'],
            sampling=current['sample_order_sha256']==previous['sample_order_sha256'],
            counted_work=current['optimizer_work']==previous['optimizer_work'],
            static_inputs=read(project/'assembly.json')['static_inputs']==read(control_project/'assembly.json')['static_inputs'])
        for key in ('horizons','width','architecture','updates','single_block_updates','learning_rate','schedule','seed','sampling_seed',
                    'feedback_protocol','physical_context','step_scale_statistics_sha256','original_normalization_sha256'):
            checks['config_'+key]=candidate_config[key]==config[key]
        for name in ('bundle_model_step_scaled.py','learned_global.py','temporal_features_v2.py','temporal_codec.py',
                     'temporal_data.py','temporal_objective.py','temporal_feedback.py','step-scales.json'):
            checks['source_'+name]=sha(project/name)==sha(control_project/name)
        trainer=(project/'temporal_train_increment_supervision.py').read_text()
        trainer=trainer.replace('import argparse\nfrom functools import partial','import argparse')
        trainer=trainer.replace('from temporal_objective_increment import loss as supervised_loss','from temporal_objective import loss')
        start=trainer.index('def construct_objective(');end=trainer.index('def run(args):',start)
        trainer=trainer[:start]+trainer[end:]
        trainer=trainer.replace('    loss=construct_objective(config,network)\n','')
        checks['training_loop_differs_only_in_objective_construction']=trainer==(control_project/'temporal_train_step_scaled.py').read_text()
        checks['registered_objective_mode']=candidate_config['supervision_mode']==mode
        if not all(checks.values()):
            write(BASE/f'{identifier}-comparison-audit-failure-01.json',dict(status='failed',checks=checks))
            raise ValueError('Supervision comparison controls failed')
        audits.append(dict(identifier=identifier,checks=checks,pair_receipt_sha256=sha(pair/'receipt.json')))
        path=BASE/'temporal-increment-supervision-development-evaluation-01'/identifier/'summary.json'
        identities.append((mode,pair,path))
        diagnosis=BASE/f'temporal-increment-supervision-fitting-diagnosis-{identifier}-01.json'
        fitting.append(dict(mode=mode,summaries=read(diagnosis)['summaries']))
        sources[str(diagnosis.relative_to(ROOT))]=sha(diagnosis)
    for mode,pair,path in identities:
        value=read(path);evaluations.append((mode,value));sources[str(path.relative_to(ROOT))]=sha(path)
    if len({value['bank_sha256']for _,value in evaluations})!=1 or len({value['metrics_sha256']for _,value in evaluations})!=1:
        raise ValueError('Evaluation data or metrics differ between supervision arms')
    comparisons=[]
    for regime in ('small','32cubed'):
        for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
            values={}
            for mode,value in evaluations:
                scenes=[scene for scene in value['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                worst={field:max(scene['summaries'][index]['worst'][field]for scene in scenes)for field in ('pressure','equation','velocity','tracer')}
                values[mode]=dict(worst=worst,accuracy_pass=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes))
            comparisons.append(dict(regime=regime,interval=interval,arms=values))
    qualified=[mode for mode in ('full_state','increments')if all(row['arms'][mode]['accuracy_pass']for row in comparisons if row['regime']=='small')]
    write(output,dict(status='completed',comparisons=comparisons,fitting=fitting,control_audits=audits,
        accuracy_qualified_small_grid_recipes=qualified,CPU_latency_claim=False,source_evidence=sources,
        registration_sha256=sha(DOCS/'temporal-increment-supervision-registration-01.json'),source_sha256=sha(__file__),
        reserved_scenes_consumed=False,optimizer_updates=0,original_goal_complete=False,model_promotion=False))
    lines=['# Matched full-state and increment-weighted MSE supervision','',
        'All three recipes use the same step-scaled K2 endpoint network, initial tensors, data, sampled windows, schedule and own-history curriculum. '
        'The two MSE arms differ only in velocity/tracer field-error normalization; both use the same pressure, equation and physical terms. '
        'The earlier log objective is shown separately. Evaluation targets and error floors are unchanged.','',
        '| Grids | Interval | Field | Log control | Full-state MSE | Increment MSE |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for row in comparisons:
        for field in ('pressure','equation','velocity','tracer'):
            values=' | '.join(f"{row['arms'][mode]['worst'][field]:.6g}"for mode in ('log_control','full_state','increments'))
            lines.append(f"| {row['regime']} | {row['interval']} | {field} | {values} |")
    lines.extend(['','Small-grid accuracy-qualified new recipes: '+(', '.join(qualified)or'none')+'.',
        'Full CPU latency remains unmeasured for these new fits. Every intermediate state and240-step own-history remain required; '
        'no reserved scenes were consumed and no model is promoted.','',
        '[All evidence, fitting diagnostics and control audits](temporal-increment-supervision-results.json)',''])
    (DOCS/'temporal-increment-supervision-results.md').write_text('\n'.join(lines))
    print('Completed matched supervision comparison; accuracy-qualified recipes:',qualified,flush=True)
