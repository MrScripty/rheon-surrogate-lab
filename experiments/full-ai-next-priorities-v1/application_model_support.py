"""Build a declared CPU deployment from exact managed neural artifacts.

This constructs the requested arithmetic policy. Accuracy/execution evidence
must be checked separately before a measured or reserved deployment is admitted.
"""
import torch
from fixed_viscosity_evaluation_support import load_model as load_family
from viscosity_composition_support import load_model as load_composition
from fast_preparation_model_support import wrap as fast_wrap
from float32_learned_pressure_map import Float32LearnedPressureMap
from temporal_train_displacement import inventory
from study_support import ROOT,HERE,BASE,read,sha


def load_model(spec):
    directory=(ROOT/spec['directory']).resolve()
    if not directory.is_relative_to(BASE.resolve()):raise ValueError('Declared study neural artifact directory required')
    if sha(directory/'receipt.json')!=spec['receipt_sha256']:raise ValueError('Declared neural artifact receipt changed')
    receipt=read(directory/'receipt.json')
    if spec['family']=='fixed-viscosity':
        weights=directory/'restore/weights.npz';normalization=directory/'restore/normalization.json'
        network,norm,identity=load_family(directory);project=ROOT/receipt['source_project']
    elif spec['family']=='neural-composition':
        weights=directory/'weights.npz';normalization=directory/'normalization.json'
        network,norm,identity=load_composition(directory)
        parent=ROOT/receipt['parent_pair'];project=ROOT/read(parent/'receipt.json')['source_project']
    else:raise ValueError('Unsupported unqualified full-state artifact family')
    if network.horizons!=spec['horizons']or identity['horizons']!=spec['horizons']:
        raise ValueError('Actual trained horizon must match the declared deployment')
    if sha(weights)!=spec['weights_sha256']or sha(normalization)!=spec['normalization_sha256']:
        raise ValueError('Exact neural weights/normalization changed')
    before=inventory(network);assets=project/'expert-assets'
    network,norm,identity=fast_wrap(network,norm,identity,assets)
    if spec['policy']=='fast-float64':
        pressure_source=HERE/'full_learned_pressure_map.py';physical_equivalence='Requires exact execution proof against its checkpoint'
    elif spec['policy']=='fast-float32-map':
        core=Float32LearnedPressureMap(assets/'pressure',assets/'gate-bounds.json').eval()
        core.load_state_dict(network.pressure_core.state_dict(),strict=True);network.pressure_core=core
        pressure_source=HERE/'float32_learned_pressure_map.py';physical_equivalence='Rounding changes; requires its own complete actual accuracy evaluation'
    else:raise ValueError('Unknown deployment arithmetic; no implicit fallback')
    if inventory(network)!=before or any(p.device.type!='cpu'for p in network.parameters())or torch.cuda.is_initialized():
        raise ValueError('Deployment must preserve every neural tensor on CPU without CUDA initialization')
    return network,norm,dict(identity,identifier=spec['identifier'],horizons=network.horizons,
        application_policy=spec['policy'],application_loader_sha256=sha(__file__),
        pressure_execution_source_sha256=sha(pressure_source),trained_horizons=network.horizons,
        original_neural_tensor_inventory_unchanged=True,normalization_sha256=sha(normalization),
        artifact_receipt_sha256=sha(directory/'receipt.json'),physical_equivalence=physical_equivalence,
        accuracy_claim=False,optimizer_updates=0,numerical_pressure_solver=False)
