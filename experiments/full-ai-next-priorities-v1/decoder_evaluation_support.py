"""Load captured full-state expert inference and its exact managed CPU restore."""
from study_support import ROOT,HERE,read,sha
from temporal_train_decoder import construct_network,load_weights,inventory


def load_model(pair):
    receipt=read(pair/'receipt.json');project=ROOT/receipt['source_project']
    if sha(project/'assembly.json')!=receipt['source_assembly_sha256']:raise ValueError('Expert Project changed')
    assembly=read(project/'assembly.json')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured expert source/asset changed')
    for name in ('bundle_model_decoder.py','bundle_model_pressure_expert.py','frozen_temporal_pressure_core.py','factored_request_model.py',
                 'temporal_train_decoder.py','temporal_features_v2.py','temporal_feedback.py'):
        if sha(project/name)!=sha(HERE/name):raise ValueError('Evaluator differs from captured full-state inference')
    restored=read(pair/'restore/receipt.json')
    if restored['mode']!='restore'or restored['optimizer_updates']!=0 or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Actual exact zero-update CPU restore required')
    weights=pair/'restore/weights.npz'
    if sha(weights)!=restored['outputs']['weights.npz']['sha256']:raise ValueError('Restored expert weights changed')
    config=assembly['config']
    network=construct_network(config,'cpu',project/'step-scales.json',project/'expert-assets').eval()
    initial=inventory(network)
    for name,spec in restored['frozen_parameter_inventory'].items():
        if initial[name]!=spec:raise ValueError('Frozen expert differs from prepared initialization')
    load_weights(network,weights,restored['final_inventory'])
    final=inventory(network)
    if any(final[name]!=spec for name,spec in restored['frozen_parameter_inventory'].items()):
        raise ValueError('Restored frozen expert differs from its unchanged learned maps')
    return network,read(pair/'restore/normalization.json'),dict(pair=str(pair.relative_to(ROOT)),
        pair_receipt_sha256=sha(pair/'receipt.json'),weights_sha256=sha(weights),horizons=config['horizons'],
        parameters=restored['parameters'],trainable_parameters=restored['trainable_parameters'],
        family='full-state-'+config['architecture']+'-decoder-v1',architecture=config['architecture'],
        supervision_mode=config['supervision_mode'],inference_source_sha256=sha(HERE/'bundle_model_decoder.py'),
        pressure_core_source_sha256=sha(HERE/'frozen_temporal_pressure_core.py'),pressure_expert_calls_per_block=1)
