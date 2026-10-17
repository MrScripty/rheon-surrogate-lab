"""Research-only fixed-pass adapter over current Rheon request validation.

Does not change or replace a deployed manifest, stopping policy or checkpoint.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import inference
from material import read,sha


def load(path):
    spec=read(path);material=read(spec['material']);root=Path(material['root'])
    if spec['schema']!='rheon-pressure-pass-sweep-adapter-v1' or spec['passes']not in material['config']['passes']:
        raise ValueError('Explicit frozen research pass policy required')
    state=inference.load(material,spec['id'])
    owner=root/material['config']['source_lab']/'experiments/dependency-migration-v1/model_adapter.py'
    loader=importlib.util.spec_from_file_location('sweep_current_request_contract',owner)
    validator=importlib.util.module_from_spec(loader);sys.modules[loader.name]=validator;loader.loader.exec_module(validator)
    return SimpleNamespace(**vars(state),validator=validator,manifest=material['support'],
        passes=spec['passes'],policy={'max_network_calls':spec['passes'],'stopping':None})


def predict(request,state,trace):
    sample=state.validator.sample_from_request(request,state)
    calls=[];hook=state.network.register_forward_hook(lambda *_:calls.append(1))
    try:
        with state.torch.no_grad():pressure=inference.fixed(state,sample,state.passes)
    finally:hook.remove()
    values=pressure.reshape(-1).numpy()
    if not bool(state.torch.isfinite(pressure).all()) or values[0]!=0:
        raise ValueError('Nonfinite or noncanonical research prediction')
    trace.append({'proposals':len(calls),'requested_network_passes':state.passes})
    return {'kind':'candidate','id':request['id'],'role':'full_pressure','pressure':values.tolist(),'device':'cpu','optimizer_updates':0}


def failure(error,request_id=None):
    return {'kind':'error','id':request_id,'code':'invalid','message':str(error)[:2000]}
