"""Load the full neural composition with closed parent and coefficient provenance."""
from displacement_evaluation_support import load_model as load_parent
from managed_fit_artifacts import load_weights,inventory
from study_support import ROOT,HERE,read,sha


def load_model(directory):
    directory=directory.resolve();receipt=read(directory/'receipt.json')
    if receipt['status']!='assembled_from_exact_managed_CPU_parents'or receipt['optimizer_updates']!=0:
        raise ValueError('Registered zero-update neural assembly required')
    parent=ROOT/receipt['parent_pair'];component=ROOT/receipt['component_pair']
    for label,path in (('parent',parent),('component',component)):
        if sha(path/'receipt.json')!=receipt['parents'][label]['pair_receipt_sha256']or sha(path/'restore/weights.npz')!=receipt['parents'][label]['weights_sha256']:
            raise ValueError('Managed composition parent changed')
    registration=ROOT/'docs/plans/full-ai-next-priorities-v1/neural-viscosity-composition-registration-01.json'
    if sha(registration)!=receipt['registration_sha256']:raise ValueError('Composition registration changed')
    reg=read(registration)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered composition source changed')
    network,norm,identity=load_parent(parent);initial=inventory(network)
    if sha(directory/'weights.npz')!=receipt['weights_sha256']or sha(directory/'normalization.json')!=receipt['normalization_sha256']:
        raise ValueError('Neural composition weights/normalization changed')
    if sha(directory/'normalization.json')!=sha(parent/'restore/normalization.json'):raise ValueError('Physical normalization changed')
    load_weights(network,directory/'weights.npz',receipt['inventory'])
    changed=[n for n,spec in inventory(network).items()if initial[n]!=spec]
    if changed!=['viscous_head.weight']:raise ValueError('Only learned viscosity may differ from the parent')
    return network,norm,dict(identity,identifier=receipt['identifier'],assembly=str(directory.relative_to(ROOT)),
        assembly_receipt_sha256=sha(directory/'receipt.json'),weights_sha256=receipt['weights_sha256'],
        family='full-state-neural-displacement-calibrated-viscosity-v1',
        composition_loader_sha256=sha(__file__),component_training_run=receipt['parents']['component']['training_run_id'],
        assembly_optimizer_updates=0,numerical_viscosity_in_inference=False)
