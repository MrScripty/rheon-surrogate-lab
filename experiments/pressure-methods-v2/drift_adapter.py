"""Deployable periodic and equation-triggered policies; no trajectory oracle input."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

def load(path):
    spec=json.loads(Path(path).read_text());root=Path(spec['root'])
    sys.path[:0]=[str(root/'experiments/pressure-pass-sweep-v1'),str(root/'experiments/pressure-methods-v1')]
    from material import read,sha
    if spec['schema']!='rheon-pressure-methods-v2-drift' or sha(__file__)!=spec['adapter_sha256']:
        raise ValueError('Bound v2 drift adapter required')
    if spec['kind'] not in ['neural','periodic','triggered','spectral'] or not 1<=spec['passes']<=16:
        raise ValueError('Declared drift policy required')
    original=read(spec['material']);material=original
    if spec.get('evaluation'):
        if sha(spec['evaluation'])!=spec['evaluation_sha256']:raise ValueError('Evaluation identity changed')
        entry=read(spec['evaluation'])['entry'];material=dict(original,models=[entry]);identifier=entry['id']
    else:identifier='C53'
    import inference
    state=inference.load(material,identifier)
    source=root/original['config']['source_lab']/'experiments/dependency-migration-v1/model_adapter.py'
    loader=importlib.util.spec_from_file_location('v2_drift_validator',source)
    validator=importlib.util.module_from_spec(loader);loader.loader.exec_module(validator)
    return SimpleNamespace(**vars(state),validator=validator,manifest=original['support'],
        specification=spec,policy=dict(max_network_calls=spec['passes'],stopping=None))

def predict(request,state,trace):
    import inference,numerics
    torch=state.torch;spec=state.specification
    sample=state.validator.sample_from_request(request,state)
    sample['rhs']=sample['native_input_residual']+state.model.full_operator(
        sample['base'],sample['cell_size_m'],sample['rho'])
    calls=[];hook=state.network.register_forward_hook(lambda *_:calls.append(1))
    numerical=0;triggered=False;residuals=[]
    def remaining(p):
        return sample['native_input_residual']-state.model.full_operator(p-sample['base'],sample['cell_size_m'],sample['rho'])
    denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
    def qualified(r):
        norm=float(torch.linalg.vector_norm(r))
        predicted=float(sample['dt'][0]*r.abs().max()/sample['cell_size_m'][0]**3)
        return norm<=max(1e-12,denominator*1e-6) and predicted<=1e-7
    scheduled=spec['kind']=='spectral' or (spec['kind']=='periodic' and
        (request['id']-spec['warmup_steps']-1)%spec['interval']==0)
    try:
        with torch.no_grad():
            pressure=sample['base']
            if not scheduled:
                for _ in range(spec['passes']):
                    if spec['kind']=='triggered' and qualified(remaining(pressure)):break
                    pressure=inference.step(state,sample,pressure)
                    residuals.append(float(torch.linalg.vector_norm(remaining(pressure)))/denominator)
            r=remaining(pressure)
            triggered=spec['kind']=='triggered' and not qualified(r)
            if scheduled or triggered:
                pressure=pressure+numerics.correction(r,sample['cell_size_m'],sample['rho'],None);numerical=1
    finally:hook.remove()
    if not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0])!=0:
        raise ValueError('Finite cell-zero-gauge pressure required')
    trace.append(dict(proposals=len(calls),kind=spec['kind'],numerical_corrections=numerical,
        scheduled=scheduled,residual_triggered=triggered,relative_residuals=residuals,
        trigger_inputs='Current pressure equation and candidate only; no reference trajectory'))
    return dict(kind='candidate',id=request['id'],role='full_pressure',pressure=pressure.flatten().tolist(),
        device='cpu',optimizer_updates=0)

def failure(error,request_id=None):
    return dict(kind='error',id=request_id,code='invalid',message=str(error)[:1000])
