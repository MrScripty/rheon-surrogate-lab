"""Pretraining conversion proof with retained base weights and active coarse map.

Synthetic coarse coefficients exercise deployment; they are not learned weights
or scientific model results. Export and verify run in separate CPU processes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('export', 'verify', 'negative'))
    for name in ('project', 'artifacts', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--package', type=Path)
    p.add_argument('--expected-error')
    a = p.parse_args()
    started = time.perf_counter()
    import numpy as np
    import torch
    if a.phase == 'export':
        a.work.mkdir(parents=True, exist_ok=False)
        sys.path.insert(0, str(a.project))
        import model
        model.configure('cpu')
        config = json.loads((a.project/'config.json').read_text())
        model.select(config)
        network = model.PressureCorrectionNet().eval()
        with np.load(a.artifacts/'weights.npz', allow_pickle=False) as source:
            missing, unexpected = network.load_state_dict({k:torch.from_numpy(source[k].copy()) for k in source.files}, strict=False)
        if missing != ['spatial_coarse.weight'] or unexpected:
            raise ValueError('conversion source mapping differs')
        with torch.no_grad():
            network.spatial_coarse.weight.copy_(torch.sin(torch.arange(4096).reshape(64,64)) * 1e-4)
        outputs = {}
        normalization = json.loads((a.artifacts/'normalization.json').read_text())
        for index, counts in enumerate(((16,16,16), (33,11,9), (25,16,23))):
            z,y,x = torch.meshgrid(*(torch.arange(n,dtype=torch.float64)+.5 for n in counts[::-1]), indexing='ij')
            exact = (torch.cos(torch.pi*x/counts[0]) + .25*torch.cos(2*torch.pi*y/counts[1]))[None,None]
            exact -= exact.flatten()[0].clone()
            sample = {key:torch.tensor([value],dtype=torch.float64) for key,value in
                      [('cell_size_m',.125), ('rho',1000.), ('dt',1/60)]}
            sample['base'] = torch.zeros_like(exact)
            sample['native_input_residual'] = model.full_operator(exact, sample['cell_size_m'], sample['rho'])
            inputs = model.build_inputs(sample['native_input_residual'],sample['base'],sample['cell_size_m'],sample['rho'],sample['dt'],normalization)
            with torch.no_grad():
                outputs[f'raw_{index}'] = network(inputs).numpy().copy()
                outputs[f'pressure_{index}'] = model.predict_pressure(network,sample,normalization).numpy().copy()
            for key,value in sample.items():
                outputs[f'{index}_{key}'] = value.numpy().copy()
            outputs[f'input_{index}'] = inputs.numpy().copy()
        with (a.work/'expected.npz').open('xb') as stream:
            np.savez(stream,**outputs)
        import data
        inventory = data.helpers().save_weights(network,a.work/'weights.npz')
        (a.work/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        (a.work/'normalization.json').write_bytes((a.artifacts/'normalization.json').read_bytes())
        print(json.dumps({'phase':'export','parameters':sum(p.numel() for p in network.parameters()),'elapsed_s':time.perf_counter()-started}))
        return
    sys.path.insert(0, str(a.package))
    import model_adapter
    if a.phase == 'negative':
        try:
            model_adapter.load(a.package/'manifest.json')
        except model_adapter.ProtocolError as error:
            if error.code != 'invalid' or str(error) != a.expected_error:
                raise
            result = {'status':'passed','expected_code':error.code,'expected_message':str(error),'elapsed_s':time.perf_counter()-started}
            print(json.dumps(result))
            return
        raise AssertionError('mutated numerical mapping was accepted')
    state = model_adapter.load(a.package/'manifest.json')
    rows = []
    with np.load(a.work/'expected.npz',allow_pickle=False) as arrays, torch.no_grad():
        for index in range(3):
            sample = {key:torch.from_numpy(arrays[f'{index}_{key}'].copy()) for key in
                      ('base','native_input_residual','cell_size_m','rho','dt')}
            raw = state.network(torch.from_numpy(arrays[f'input_{index}'].copy()))
            pressure = state.model.predict_pressure(state.network,sample,state.normalization)
            if not np.array_equal(raw.numpy(),arrays[f'raw_{index}']) or not np.array_equal(pressure.numpy(),arrays[f'pressure_{index}']):
                raise AssertionError('packaged raw or recurrent pressure differs from captured trainer')
            rows.append({'counts_xyz':list(sample['base'].shape[:1:-1]),'exact_raw':True,'exact_two_call_pressure':True})
    result = {'status':'passed','rows':rows,'weights_sha256':digest(a.package/'weights.npz'),
        'candidate_parameters':sum(p.numel() for p in state.network.parameters()),
        'coarse_nonzero':bool(torch.count_nonzero(state.network.spatial_coarse.weight)),
        'cuda_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'elapsed_s':time.perf_counter()-started,
        'scope':'Exact CPU conversion with synthetic nonzero coarse coefficients and retained control weights; no scientific accuracy claim.'}
    with (a.work/'conversion-proof.json').open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
