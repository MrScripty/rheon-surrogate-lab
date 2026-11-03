"""Verify matched data and training work for the shared-global capacity comparison."""
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


if __name__=='__main__':
    output=BASE/'temporal-global-comparison-audit-01.json'
    if output.exists():
        raise ValueError('Fresh capacity comparison audit required')
    control=BASE/'b1-pair-02'
    candidate=BASE/'b1-global-pair-01'
    old=read(control/'restore/history.json')
    new=read(candidate/'restore/history.json')
    old_project=ROOT/read(control/'receipt.json')['source_project']
    new_project=ROOT/read(candidate/'receipt.json')['source_project']
    a=read(old_project/'config.json')
    b=read(new_project/'config.json')
    checks=dict(sample_order=old['sample_order_sha256']==new['sample_order_sha256'],
        counted_network_forwards_and_processed_cells=old['optimizer_work']==new['optimizer_work'],
        normalization=sha(control/'restore/normalization.json')==sha(candidate/'restore/normalization.json'),
        static_inputs=read(old_project/'assembly.json')['static_inputs']==read(new_project/'assembly.json')['static_inputs'])
    for key in('horizons','updates','single_block_updates','learning_rate','schedule','seed','sampling_seed','feedback_protocol','physical_context'):
        checks['config_'+key]=a[key]==b[key]
    for name in('temporal_features.py','temporal_features_v2.py','temporal_codec.py','temporal_data.py','temporal_objective.py','temporal_feedback.py'):
        checks['source_'+name]=sha(old_project/name)==sha(new_project/name)
    trainer=(new_project/'temporal_train_global.py').read_text().replace('from bundle_model_global import BundleNet','from bundle_model_v5 import BundleNet')
    checks['training_loop_differs_only_in_model_import']=trainer==(old_project/'temporal_train_v2.py').read_text()
    qualification=BASE/'temporal-global-cpu-qualification-01.json'
    q=read(qualification)
    checks['initial_output_function_bitwise_equivalent']=q['status']=='passed'and all(row['initialization_function_bitwise_matches_original']for row in q['rows'])
    checks['global_width_and_parameter_count']=b['width']==12 and read(candidate/'restore/receipt.json')['parameters']==170728
    write(output,dict(status='passed'if all(checks.values())else'failed',checks=checks,
        control_pair_sha256=sha(control/'receipt.json'),candidate_pair_sha256=sha(candidate/'receipt.json'),
        initial_function_qualification_sha256=sha(qualification),registration_sha256=sha(DOCS/'temporal-global-registration.json'),
        deliberate_change='Neural architecture/capacity; initialized parameter tensors and FLOPs differ, initial output function matches',
        FLOP_matched=False,optimizer_updates=0,source_sha256=sha(__file__)))
    if not all(checks.values()):
        raise ValueError('Global capacity comparison controls differ from registration')
    print('Global-capacity comparison audit passed: same inputs, sampled work, objective and training loop',flush=True)
