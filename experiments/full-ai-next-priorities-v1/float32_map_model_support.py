"""Use unchanged managed weights with an explicitly approximate neural execution."""
from pathlib import Path
from viscosity_composition_support import load_model as load_composition
from fast_preparation_model_support import wrap as fast_wrap
from float32_learned_pressure_map import Float32LearnedPressureMap
from temporal_train_displacement import inventory
from study_support import ROOT,HERE,read,sha


def wrap(network,norm,identity,assets):
    before=inventory(network);variant,norm,identity=fast_wrap(network,norm,identity,assets)
    assets=Path(assets)
    core=Float32LearnedPressureMap(assets/'pressure',assets/'gate-bounds.json').eval()
    core.load_state_dict(variant.pressure_core.state_dict(),strict=True)
    variant.pressure_core=core
    if inventory(variant)!=before:raise ValueError('All original neural tensors must remain byte-identical')
    return variant,norm,dict(identity,identifier='b2-displacement-viscosity-fast-f32map',
        execution_variant='fast-preparation-float32-learned-map-v1',exact_physical_output_inheritance_claim=False,
        execution_map_source_sha256=sha(HERE/'float32_learned_pressure_map.py'),execution_loader_sha256=sha(__file__),
        learned_map_conversion=core.conversion,no_analytic_coefficients=True,optimizer_updates=0,
        memory_caveat='Original float64 preparation maps remain allocated; measure actual RSS rather than claiming total memory halved')


def load_model(directory):
    original,norm,identity=load_composition(directory)
    parent=ROOT/read(directory/'receipt.json')['parent_pair'];project=ROOT/read(parent/'receipt.json')['source_project']
    return wrap(original,norm,identity,project/'expert-assets')
