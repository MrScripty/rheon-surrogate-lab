"""Deploy an exact retained CPU fit into the bounded research trajectory only."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def load(path):
    import json
    spec = json.loads(Path(path).read_text())
    root = Path(spec['root'])
    sys.path[:0] = [str(root/'experiments/pressure-pass-sweep-v1')]
    from material import read,sha
    if (spec['schema'] != 'rheon-pressure-methods-fit-trajectory-v1'
            or spec['passes'] not in range(1,17)
            or sha(__file__) != spec['adapter_sha256']
            or sha(spec['evaluation']) != spec['evaluation_sha256']):
        raise ValueError('Exact evaluated fit trajectory contract required')
    result = read(spec['evaluation'])
    if result['status'] != 'completed' or result['optimizer_updates'] != 0:
        raise ValueError('Completed independent fit evaluation required')
    original = read(spec['material'])
    import inference
    state = inference.load(dict(original,models=[result['entry']]),result['entry']['id'])
    validator_source = root/original['config']['source_lab']/'experiments/dependency-migration-v1/model_adapter.py'
    loader = importlib.util.spec_from_file_location('methods_fit_request_validator',validator_source)
    validator = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(validator)
    return SimpleNamespace(**vars(state),validator=validator,manifest=original['support'],
        specification=spec,policy=dict(max_network_calls=spec['passes'],stopping=None))


def predict(request,state,trace):
    import inference
    sample = state.validator.sample_from_request(request,state)
    calls = []
    hook = state.network.register_forward_hook(lambda *_:calls.append(1))
    try:
        with state.torch.no_grad():
            pressure = inference.fixed(state,sample,state.specification['passes'])
    finally:
        hook.remove()
    if not bool(state.torch.isfinite(pressure).all()) or float(pressure.flatten()[0]) != 0:
        raise ValueError('Finite full gauge-fixed pressure required')
    trace.append(dict(proposals=len(calls),kind='retained-trained-method',
        inference_source='Captured trained Project and actual fresh CPU weights',
        numerical_corrections=state.specification['passes'] if 'numerical_training_method' in state.model.ARCHITECTURE else 0))
    return dict(kind='candidate',id=request['id'],role='full_pressure',pressure=pressure.flatten().tolist(),device='cpu',optimizer_updates=0)


def failure(error,request_id=None):
    return dict(kind='error',id=request_id,code='invalid',message=str(error)[:512])
