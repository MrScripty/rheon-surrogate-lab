"""Research pressure policies with explicit numerical work; no fallback on failure."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load(path):
    spec = json.loads(Path(path).read_text())
    root = Path(spec['root'])
    sys.path[:0] = [str(root/'experiments/pressure-methods-v1'), str(root/'experiments/pressure-pass-sweep-v1')]
    from material import read, sha
    import numerics
    material = read(spec['material'])
    if (spec['schema'] != 'rheon-pressure-methods-trajectory-v1'
            or sha(root/'experiments/pressure-methods-v1/numerics.py') != spec['numerics_sha256']
            or sha(__file__) != spec['adapter_sha256']
            or spec['kind'] not in ['full-numerical', 'neural', 'before', 'after', 'periodic']):
        raise ValueError('Frozen trajectory policy/source required')
    if spec['kind'] == 'full-numerical':
        import numpy as np
        import torch
        torch.set_num_threads(1)
        state = SimpleNamespace(np=np, torch=torch, model=None, network=torch.nn.Identity().eval(), normalization={})
    else:
        import inference
        if spec['passes'] not in [1, 2, 4, 8] or spec['checkpoint'] not in ['C53', 'C44', 'C37']:
            raise ValueError('Declared frozen checkpoint and pass budget required')
        state = inference.load(material, spec['checkpoint'])
    if spec['kind'] == 'periodic' and spec['interval'] not in [4, 16, 64]:
        raise ValueError('Declared periodic interval required')
    source = root/material['config']['source_lab']/'experiments/dependency-migration-v1/model_adapter.py'
    loader = importlib.util.spec_from_file_location('methods_trajectory_request_validator', source)
    validator = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(validator)
    return SimpleNamespace(**vars(state), validator=validator, manifest=material['support'],
        specification=spec, policy=dict(max_network_calls=spec['passes'], stopping=None))


def predict(request, state, trace):
    import numerics
    sample = state.validator.sample_from_request(request, state)
    spec = state.specification
    kind = spec['kind']
    numeric_step = kind == 'full-numerical' or (kind == 'periodic' and
        (request['id']-spec['warmup_steps']-1) % spec['interval'] == 0)
    calls = []
    hook = state.network.register_forward_hook(lambda *_: calls.append(1))
    try:
        with state.torch.no_grad():
            pressure = sample['base']
            if numeric_step:
                pressure = pressure+numerics.correction(sample['native_input_residual'], sample['cell_size_m'], sample['rho'], None)
            else:
                import inference
                if kind == 'before':
                    pressure = pressure+numerics.correction(sample['native_input_residual'], sample['cell_size_m'], sample['rho'], 3)
                for _ in range(spec['passes']):
                    pressure = inference.step(state, sample, pressure)
                if kind == 'after':
                    remaining = sample['native_input_residual']-state.model.full_operator(
                        pressure-sample['base'], sample['cell_size_m'], sample['rho'])
                    pressure = pressure+numerics.correction(remaining, sample['cell_size_m'], sample['rho'], 3)
    finally:
        hook.remove()
    if not bool(state.torch.isfinite(pressure).all()) or float(pressure.flatten()[0]) != 0.:
        raise ValueError('Invalid research full pressure')
    trace.append(dict(proposals=len(calls), kind=kind, full_numerical_step=numeric_step,
                      numerical_corrections=int(numeric_step or kind in ['before', 'after'])))
    return dict(kind='candidate', id=request['id'], role='full_pressure',
                pressure=pressure.flatten().tolist(), device='cpu', optimizer_updates=0)


def failure(error, request_id=None):
    return dict(kind='error', id=request_id, code='invalid', message=str(error)[:2000])
