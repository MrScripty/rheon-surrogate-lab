"""Audit and summarize paired conditioning fits without claiming unmeasured speed."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=DOCS/'temporal-step-scaled-results.json'
    if output.exists():raise ValueError('Fresh conditioning comparison required')
    comparisons=[];audits=[];fitting=[];sources={}
    for architecture in ('endpoint','reuse'):
        original=f'b2-{architecture}';candidate=original+'-step-scaled'
        pair=BASE/f'{candidate}-pair-01';control=BASE/f'{original}-pair-01'
        new_receipt=read(pair/'receipt.json');old_receipt=read(control/'receipt.json')
        new_project=ROOT/new_receipt['source_project'];old_project=ROOT/old_receipt['source_project']
        current=read(pair/'restore/history.json');previous=read(control/'restore/history.json')
        config=read(new_project/'config.json');old_config=read(old_project/'config.json')
        checks=dict(all_artifacts_bitwise=all(new_receipt['artifact_parity'].values()),
            parameters_unchanged=new_receipt['parameters']==old_receipt['parameters'],
            initialization_tensors_identical=current['initial_inventory']==previous['initial_inventory'],
            sample_order=current['sample_order_sha256']==previous['sample_order_sha256'],
            counted_training_work=current['optimizer_work']==previous['optimizer_work'],
            physical_normalization=sha(pair/'restore/normalization.json')==sha(control/'restore/normalization.json'),
            static_inputs=read(new_project/'assembly.json')['static_inputs']==read(old_project/'assembly.json')['static_inputs'])
        for key in ('horizons','width','architecture','updates','single_block_updates','learning_rate','schedule','seed','sampling_seed','feedback_protocol','physical_context'):
            checks['config_'+key]=config[key]==old_config[key]
        for name in ('temporal_features_v2.py','temporal_codec.py','temporal_data.py','temporal_objective.py','temporal_feedback.py','learned_global.py'):
            checks['source_'+name]=sha(new_project/name)==sha(old_project/name)
        if not all(checks.values()):
            write(BASE/f'{candidate}-comparison-audit-failure-01.json',dict(status='failed',checks=checks))
            raise ValueError('Conditioning comparison controls failed; inspect retained evidence')
        audits.append(dict(identifier=candidate,checks=checks,pair_receipt_sha256=sha(pair/'receipt.json')))
        paths=[BASE/'temporal-sharing-development-evaluation-01'/original/'summary.json',
               BASE/'temporal-step-scaled-development-evaluation-01'/candidate/'summary.json']
        evaluations=[read(path)for path in paths]
        for path in paths:sources[str(path.relative_to(ROOT))]=sha(path)
        for regime in ('small','32cubed'):
            for index,interval in enumerate(('single_block','own_history12','own_history48','own_history240')):
                values=[]
                for value in evaluations:
                    scenes=[scene for scene in value['scenes']if(max(scene['counts_xyz'])<=16)==(regime=='small')]
                    worst={field:max(scene['summaries'][index]['worst'][field]for scene in scenes)for field in ('pressure','equation','velocity','tracer')}
                    values.append(dict(worst=worst,accuracy_pass=all(scene['summaries'][index]['all_accuracy_criteria_pass']for scene in scenes)))
                comparisons.append(dict(architecture=architecture,horizons=2,regime=regime,interval=interval,
                    original=values[0],step_scaled=values[1]))
        diagnosis=BASE/f'temporal-step-scaled-fitting-diagnosis-{candidate}-01.json'
        fitting.append(dict(identifier=candidate,summaries=read(diagnosis)['summaries']))
        sources[str(diagnosis.relative_to(ROOT))]=sha(diagnosis)
    qualified=[architecture for architecture in ('endpoint','reuse')if all(row['step_scaled']['accuracy_pass']for row in comparisons
               if row['architecture']==architecture and row['regime']=='small')]
    write(output,dict(status='completed',comparisons=comparisons,fitting=fitting,control_audits=audits,
        accuracy_qualified_small_grid_recipes=qualified,CPU_latency_claim=False,
        registration_sha256=sha(DOCS/'temporal-step-scaled-registration-01.json'),sources=sources,
        source_sha256=sha(__file__),reserved_scenes_consumed=False,optimizer_updates=0,
        original_goal_complete=False,model_promotion=False))
    lines=['# Temporal step-unit conditioning comparison','',
        'Only fixed aggregate train-only forcing and predicted-increment units change. Paired fits verify identical initial parameter tensors, '
        'parameter counts, data, physical error normalization, sample stream, objective, curriculum and counted training work. '
        'All values are worst relative errors across all declared scenes and horizons. No reserved scenes were consumed.','',
        '| Architecture | Grids | Interval | Field | Original | Step-scaled |','| --- | --- | --- | --- | ---: | ---: |']
    for row in comparisons:
        for field in ('pressure','equation','velocity','tracer'):
            lines.append(f"| {row['architecture']} | {row['regime']} | {row['interval']} | {field} | {row['original']['worst'][field]:.6g} | {row['step_scaled']['worst'][field]:.6g} |")
    lines.extend(['','Small-grid accuracy-qualified recipes: '+(', '.join(qualified)or'none')+'.',
        'CPU latency remains unmeasured for these new fits; accuracy alone does not establish the combined target. '
        'Fixed-window train/development and cold/warm fitting diagnostics remain in the evidence.',
        '', '[Full comparison and audit evidence](temporal-step-scaled-results.json)',''])
    (DOCS/'temporal-step-scaled-results.md').write_text('\n'.join(lines))
    print('Completed controlled step-unit comparisons; accuracy-qualified recipes:',qualified,flush=True)
