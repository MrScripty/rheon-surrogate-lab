"""Check the loss comparison retained initialization, sampled work and data exactly."""
from study_support import ROOT, BASE, DOCS, read, write, sha


if __name__ == '__main__':
    output = BASE / 'temporal-mse-comparison-audit-01.json'
    if output.exists():
        raise ValueError('Fresh comparison audit required')
    control = BASE / 'b1-pair-02'
    candidate = BASE / 'b1-relative-mse-pair-01'
    old = read(control / 'restore/history.json')
    new = read(candidate / 'restore/history.json')
    checks = dict(initial_inventory=old['initial_inventory'] == new['initial_inventory'],
                  sample_order=old['sample_order_sha256'] == new['sample_order_sha256'],
                  optimizer_work=old['optimizer_work'] == new['optimizer_work'],
                  normalization=sha(control/'restore/normalization.json') == sha(candidate/'restore/normalization.json'))
    old_project = ROOT / read(control/'receipt.json')['source_project']
    new_project = ROOT / read(candidate/'receipt.json')['source_project']
    a = read(old_project/'config.json')
    b = read(new_project/'config.json')
    for key in ('horizons','width','updates','single_block_updates','learning_rate','schedule','seed','sampling_seed','feedback_protocol','physical_context'):
        checks['config_' + key] = a[key] == b[key]
    checks['all_static_inputs'] = read(old_project/'assembly.json')['static_inputs'] == read(new_project/'assembly.json')['static_inputs']
    for name in ('bundle_model.py','bundle_model_v5.py','adaptive_pool.py','temporal_features.py','temporal_features_v2.py',
                 'temporal_codec.py','temporal_data.py','temporal_feedback.py','temporal_train_v2.py'):
        checks['source_' + name] = sha(old_project/name) == sha(new_project/name)
    write(output, dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
          registration_sha256=sha(DOCS/'temporal-loss-followup-registration.json'),
          control_pair_sha256=sha(control/'receipt.json'), candidate_pair_sha256=sha(candidate/'receipt.json'),
          optimizer_updates=0, source_sha256=sha(__file__)))
    if not all(checks.values()):
        raise ValueError('Loss comparison changed another registered factor')
    print('Controlled temporal comparison: exact initialization, sampled work, static inputs and model verified', flush=True)
