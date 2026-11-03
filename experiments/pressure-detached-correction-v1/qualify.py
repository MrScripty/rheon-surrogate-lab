"""Verify scalar parity, detached gradients, depth balance and historical numerical parity."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write


def check(project, output):
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('fundamental_qualification', ROOT / 'experiments/pressure-fundamental-bank-fit-v1/qualify.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    bank_output = output.with_suffix('.bank.json')
    owner.check(project, bank_output)
    import torch
    import data
    import model
    import fine_tuning
    import fit_bank
    import original_fine_tuning
    import physical_residual

    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ['dataset', 'dataset_extra']), read(inputs['splits']))
    cache = {(i, j): data.sample(parent, j) for i, parent in enumerate(parents) for j in range(len(parent['frames']))}
    normalization = model.fit_normalization(parents, data.fit_normalization)
    initial = ROOT / next(r['path'] for r in read(project / 'project.json')['data'] if r['id'] == 'initial_weights')
    network = model.PressureCorrectionNet().eval()
    fine_tuning.initialize(network, initial, data.helpers())
    fit_bank.bind(cache, 1600)
    before = data.helpers().state_inventory(network)
    assert all(torch.equal(torch.bincount(torch.as_tensor(row), minlength=4), torch.tensor([50]*4)) for row in fit_bank.DEPTHS)
    assert (fit_bank.DEPTHS == fit_bank.depth_order(8, 200, fine_tuning.SPEC['rollout_feedback']['depth_seed'])).all()
    manifest = read(ROOT / 'docs/plans/flow-reasoning-assessment/initializations.json')
    entry = next(e for e in manifest['initializations'] if e['weights_sha256'] == sha(initial))
    historical_project = ROOT / entry['project']
    for name in ['original_model.py', 'portable_model.py', 'frozen_gn.py']:
        assert sha(historical_project / name) == sha(project / name), name
    numerical_spec = importlib.util.spec_from_file_location('historical_physical_residual', historical_project / 'physical_residual.py')
    historical = importlib.util.module_from_spec(numerical_spec)
    numerical_spec.loader.exec_module(historical)
    rows = []
    for index, sample in enumerate(fit_bank.SAMPLES):
        import analytical
        criterion = lambda p: analytical.loss(p, sample, model.full_operator)
        pressures = []
        original_fine_tuning.loss(model.baseline, network, sample, normalization, criterion, intermediates=pressures)
        for depth in range(4):
            attached, pa, ca = fine_tuning.auxiliary(model.baseline, network, sample, normalization, pressures, criterion, depth, 'A')
            detached, pd, cd = fine_tuning.auxiliary(model.baseline, network, sample, normalization, pressures, criterion, depth, 'D')
            assert torch.equal(pa, pd) and torch.equal(attached, detached)
            assert torch.equal(pa, pressures[depth])
            assert not cd.requires_grad
            ga = torch.autograd.grad(attached.sum(), tuple(network.parameters()), retain_graph=True)
            gd = torch.autograd.grad(detached.sum(), tuple(network.parameters()), retain_graph=True)
            assert all(bool(torch.isfinite(g).all()) for g in (*ga, *gd))
            assert any(bool((g != 0).any()) for g in gd)
            if depth == 0:
                assert all(torch.equal(a, d) for a, d in zip(ga, gd, strict=True))
            else:
                prefix_a = torch.autograd.grad(attached.sum(), ca, retain_graph=True)[0]
                prefix_d = torch.autograd.grad(detached.sum(), ca, allow_unused=True, retain_graph=True)[0]
                assert bool((prefix_a != 0).any()) and prefix_d is None
            rows.append(dict(case=index, depth=depth, exact_pressure_parity=True, exact_scalar_parity=True,
                             detached_prefix=True, finite_nonzero_new_correction_gradient=True))
        with torch.no_grad():
            for passes in (1, 2, 4, 8):
                options = dict(activation='linear', steps=passes)
                old = historical.predict_pressure(network, sample, normalization, model.owner, options)
                new = physical_residual.predict_pressure(network, sample, normalization, model.owner, options)
                assert torch.equal(old, new), (index, passes)
        del pressures, attached, detached, pa, pd, ca, cd, ga, gd
    assert before == data.helpers().state_inventory(network)
    assert not torch.cuda.is_initialized()
    result = read(bank_output)
    result.update(detached_correction=dict(initialization=entry['id'], cases=rows,
        historical_helper_pressure_parity=True, historical_common_network_sources_identical=True,
        historical_parity_passes=[1,2,4,8], depth_exposures_per_shape=[50]*4,
        depth_order_sha256=fine_tuning.STATS['rollout_feedback']['depth_order_sha256'],
        depth_rng_independent=True, optimizer_updates=0),
        detached_qualification_source_sha256=sha(__file__), total_elapsed_s=time.perf_counter()-started)
    write(output, result)
    print('Qualified detached correction', entry['id'], fine_tuning.SPEC['rollout_feedback']['mode'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    check(args.project.resolve(), args.output.resolve())
