"""Same pressure network with a declared numerical stage before each pass."""
import json
from pathlib import Path
from historical_physical import *
import historical_physical
import coarse

METHOD = json.loads(Path(__file__).with_name('hybrid-method.json').read_text())['kind']
if METHOD not in ['numerical-coarse','sparse-anchors']:
    raise ValueError('Closed numerical training method required')


def predict_pressure(network,sample,normalization,owner,correction,intermediates=None):
    base = sample['base']
    initial = sample['native_input_residual']
    pressure = base.clone()
    for stage in range(correction['steps']):
        remaining = initial if stage == 0 else initial-owner.full_operator(
            pressure-base,sample['cell_size_m'],sample['rho'])
        numerical = (coarse.numerical if METHOD == 'numerical-coarse' else coarse.sparse)(
            remaining,sample['cell_size_m'],sample['rho'])
        numerical_base = pressure+numerical
        fine_remaining = remaining-owner.full_operator(numerical,sample['cell_size_m'],sample['rho'])
        proposed = historical_physical.predict_pressure(network,
            {**sample,'base':numerical_base,'native_input_residual':fine_remaining},
            normalization,owner,{**correction,'steps':1})
        pressure = (numerical_base+coarse.pin_neural_correction(proposed-numerical_base)
                    if METHOD == 'sparse-anchors' else proposed)
        if intermediates is not None:
            intermediates.append(pressure)
    return pressure
