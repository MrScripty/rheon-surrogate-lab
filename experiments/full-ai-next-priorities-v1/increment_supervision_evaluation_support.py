"""Load only a verified managed shared-global model and its exact CPU restore."""
from study_support import ROOT,HERE,read,sha
from temporal_train_increment_supervision import construct_network,load_weights


def load_model(pair):
    receipt=read(pair/'receipt.json')
    project=ROOT/receipt['source_project']
    if sha(project/'assembly.json')!=receipt['source_assembly_sha256']:
        raise ValueError('Global training Project changed')
    assembly=read(project/'assembly.json')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Captured global training source changed')
    for name in ('bundle_model_step_scaled.py','temporal_train_increment_supervision.py','temporal_objective_increment.py','learned_global.py','temporal_features_v2.py','temporal_feedback.py'):
        if sha(project/name)!=sha(HERE/name):
            raise ValueError('Global evaluator differs from actual captured inference')
    restored=read(pair/'restore/receipt.json')
    if restored['mode']!='restore'or restored['optimizer_updates']!=0 or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Actual zero-update CPU restore required')
    weights=pair/'restore/weights.npz'
    if sha(weights)!=restored['outputs']['weights.npz']['sha256']:
        raise ValueError('Restored global weights changed')
    config=assembly['config']
    network=construct_network(config,'cpu',project/'step-scales.json').eval()
    load_weights(network,weights,restored['final_inventory'])
    return network,read(pair/'restore/normalization.json'),dict(pair=str(pair.relative_to(ROOT)),
        pair_receipt_sha256=sha(pair/'receipt.json'),weights_sha256=sha(weights),horizons=config['horizons'],
        parameters=restored['parameters'],family='full-state-'+config['architecture']+'-step-scaled-v1',architecture=config['architecture'],supervision_mode=config['supervision_mode'],inference_source_sha256=sha(HERE/'bundle_model_step_scaled.py'),
        global_operator_sha256=sha(HERE/'learned_global.py'))
