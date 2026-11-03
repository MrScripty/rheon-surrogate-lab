"""Verify the registered training controls for every endpoint/reuse comparison."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=BASE/'temporal-sharing-comparison-audit-01.json'
    if output.exists():raise ValueError('Fresh sharing comparison audit required')
    registration=read(DOCS/'temporal-sharing-registration-01.json');rows=[]
    for k in (2,3,4):
        control=BASE/f'b{k}-pair-02';control_receipt=read(control/'receipt.json')
        control_project=ROOT/control_receipt['source_project']
        original=read(control/'restore/history.json');config=read(control_project/'config.json')
        initial_encoders=[]
        for architecture in ('endpoint','reuse'):
            identifier=f'b{k}-{architecture}';pair=BASE/f'{identifier}-pair-01'
            receipt=read(pair/'receipt.json');project=ROOT/receipt['source_project']
            candidate=read(pair/'restore/history.json');current=read(project/'config.json')
            checks=dict(pair_complete=receipt['status']=='completed',
                exact_CPU_restore=all(receipt['artifact_parity'].values()),
                registered_parameter_count=receipt['parameters']==next(r['parameters']for r in registration['models']
                    if r['horizons']==k and r['architecture']==architecture),
                sample_order=original['sample_order_sha256']==candidate['sample_order_sha256'],
                normalization=sha(control/'restore/normalization.json')==sha(pair/'restore/normalization.json'),
                static_inputs=read(control_project/'assembly.json')['static_inputs']==read(project/'assembly.json')['static_inputs'])
            for key in ('horizons','updates','single_block_updates','learning_rate','schedule','seed','sampling_seed','feedback_protocol','physical_context'):
                checks['config_'+key]=config[key]==current[key]
            for name in ('temporal_features.py','temporal_features_v2.py','temporal_codec.py','temporal_data.py','temporal_objective.py','temporal_feedback.py'):
                checks['source_'+name]=sha(control_project/name)==sha(project/name)
            work=candidate['optimizer_work']
            checks['matched_original_work_counters']=all(work[key]==value for key,value in original['optimizer_work'].items())
            checks['one_encoder_per_block']=work['encoder_calls']==work['network_forwards']==18000
            checks['all_physical_outputs_counted']=work['predicted_physical_states']==18000*k
            checks['latent_updates_counted']=work['latent_updates']==(18000*(k-1)if architecture=='reuse'else 0)
            checks['endpoint_outputs_counted']=work['endpoint_predictions']==(18000 if architecture=='endpoint'else 0)
            checks['intermediate_outputs_counted']=work['intermediate_states']==(18000*(k-1)if architecture=='endpoint'else 0)
            encoder={name:value for name,value in candidate['initial_inventory'].items()
                if name.startswith(('fine.','forcing.','local.','global_blocks.'))}
            initial_encoders.append(encoder)
            rows.append(dict(identifier=identifier,checks=checks,optimizer_work=work,
                pair_receipt_sha256=sha(pair/'receipt.json'),status='passed'if all(checks.values())else'failed'))
        if initial_encoders[0]!=initial_encoders[1]:raise ValueError('Paired models did not initialize the same shared encoder')
    passed=all(row['status']=='passed'for row in rows)
    write(output,dict(status='passed'if passed else'failed',rows=rows,
        registration_sha256=sha(DOCS/'temporal-sharing-registration-01.json'),source_sha256=sha(__file__),
        same_encoder_initialization_within_each_K=True,FLOP_matched=False,
        deliberate_changes='Fresh pressure, no pressure input; predicted endpoint decoder or retained-feature updates',
        original_model_initial_function_equivalent=False,optimizer_updates=0,reserved_scenes_consumed=False))
    if not passed:raise ValueError('Sharing comparison controls failed; inspect retained audit')
    print('Audited all six sharing fits: common data, sampling, loss, own-history work and paired encoder initialization',flush=True)
