"""Offline CPU checkpoint owner and explicit single-network recurrence.

Captured models own their feature preparation/reconstruction. This research
policy controls only recurrence and never modifies model globals or weights.
"""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from material import read,sha

_CACHE=None


def load(material,identifier):
    global _CACHE
    root=Path(material['root']);entry=next(m for m in material['models'] if m['id']==identifier)
    if _CACHE is not None:
        if _CACHE.identifier!=identifier:raise ValueError('One checkpoint per numerical process required')
        return _CACHE
    for name,expected in entry['identities'].items():
        if sha(root/name)!=expected:raise ValueError('Changed checkpoint material '+name)
    project=root/entry['project'];artifacts=root/entry['restore']
    sys.path.insert(0,str(project))
    import torch
    import model
    import data
    config=read(project/'config.json');witness=read(project/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(project/name)!=expected:raise ValueError('Changed captured numerical source '+name)
    model.configure('cpu');torch.set_num_threads(1)
    if config.get('conditioning') is not None:model.select(config)
    receipt=read(artifacts/'receipt.json');history=read(artifacts/'history.json')
    if (receipt['mode']!='restore' or receipt['optimizer_updates']!=0
        or receipt['execution']['cuda_initialized'] or receipt['architecture']!=model.ARCHITECTURE
        or receipt['closure']['source_witness_sha256']!=sha(project/'source-witness.json')
        or receipt['closure']['experiment_sha256']!=sha(project/'config.json')):
        raise ValueError('Captured CPU restore contract differs')
    for name in ['weights.npz','normalization.json','history.json']:
        if sha(artifacts/name)!=receipt['outputs'][name]['sha256']:raise ValueError('Restore output changed '+name)
    network=model.PressureCorrectionNet().eval()
    inventory=history['checkpoints'][f"checkpoint-{config['training']['updates']:03d}.npz"]['inventory']
    data.helpers().load_weights(network,artifacts/'weights.npz',inventory)
    if torch.cuda.is_initialized() or any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in network.parameters()):
        raise ValueError('Unchanged CPU float32 neural parameters required')
    _CACHE=SimpleNamespace(identifier=identifier,torch=torch,np=data.np,model=model,data=data,network=network,
        normalization=read(artifacts/'normalization.json'),config=config,entry=entry)
    return _CACHE


def step(state,sample,current):
    model=state.model
    remaining=sample['native_input_residual']-model.full_operator(current-sample['base'],sample['cell_size_m'],sample['rho'])
    updated={**sample,'base':current,'native_input_residual':remaining}
    if state.config.get('conditioning')=='physical-residual':
        physical=importlib.import_module('physical_residual')
        return physical.predict_pressure(state.network,updated,state.normalization,model.owner,
                                         {**model.CORRECTION,'steps':1})
    return model.predict_pressure(state.network,updated,state.normalization)


def fixed(state,sample,passes):
    pressure=sample['base']
    for _ in range(passes):pressure=step(state,sample,pressure)
    return pressure
