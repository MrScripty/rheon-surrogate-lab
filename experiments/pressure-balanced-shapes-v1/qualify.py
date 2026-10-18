"""Compare a real aggregate cycle with frozen per-example gradient/Adam oracles."""
import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write
from qualify import qualify


def check(config_path, project, output):
    started = time.perf_counter()
    if output.exists():
        raise ValueError('Fresh qualification receipt required')
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    config = read(config_path)
    control = ROOT / config['control_project']
    left, right = read(project / 'config.json'), read(control / 'config.json')
    for document in (left, right):
        document.pop('id')
        for name in ('updates', 'checkpoints', 'schedule'):
            document['training'].pop(name)
    assert left == right, 'Only update/checkpoint/schedule identity may differ'
    a, b = read(project / 'fine-tuning.json'), read(control / 'fine-tuning.json')
    a.pop('arm'); b.pop('arm')
    assert a == b, 'Unchanged four-pass feedback and fitting bank required'
    witness, previous = read(project / 'source-witness.json'), read(control / 'source-witness.json')
    for name, identity in previous['sources'].items():
        if name not in ('config.json', 'fine-tuning.json', 'fit_bank.py'):
            assert witness['sources'][name] == identity, name
    assert witness['sources']['original_fit_bank.py'] == previous['sources']['fit_bank.py']
    assert witness['normalization_sha256'] == previous['normalization_sha256']
    assert read(project / 'assembly.json')['runtime_manifest_sha256'] == read(control / 'assembly.json')['runtime_manifest_sha256']
    base = output.with_suffix('.base.json')
    qualify(project, base)
    import numpy as np
    import torch
    import model
    import data
    import fine_tuning
    import fit_bank
    import gradient_clipping
    inputs = source_inputs(project)
    _, parents = data.load(tuple(inputs[k].read_bytes() for k in ('dataset', 'dataset_extra')), read(inputs['splits']))
    assert all(p['entry']['split'] == 'train' for p in parents)
    cache = {(i, j): data.sample(p, j) for i, p in enumerate(parents) for j in range(len(p['frames']))}
    normalization = model.fit_normalization(parents, data.fit_normalization)
    initial = ROOT / next(row['path'] for row in read(project / 'project.json')['data'] if row['id'] == 'initial_weights')
    network = model.PressureCorrectionNet()
    fine_tuning.initialize(network, initial, data.helpers())
    fit_bank.bind(cache, config['updates'])
    stats = fine_tuning.STATS['fundamental_bank']
    count = len(fit_bank.SAMPLES)
    prior = read(ROOT / config['control_result'])['training'][0]
    assert count == 8 and config['updates'] == 200 and config['examples'] == 1600
    assert stats['sampling_order_sha256'] == prior['fundamental_bank']['sampling_order_sha256']
    assert stats['amplitude_rms_pa'] == prior['fundamental_bank']['amplitude_rms_pa']
    assert stats['cases'] == prior['fundamental_bank']['cases']
    assert stats['expected_exposures'] == [200] * count
    assert stats['resident_tensor_bytes'] <= 8 * 1024**2
    captured = fit_bank.bank
    # Rebuild all tensors from the unchanged captured bank owner, not an edited label projection.
    expected, cases, _ = captured.build(next(iter(cache.values())), a['fitting_bank'], stats['amplitude_rms_pa'])
    assert all(torch.equal(sample[key], other[key]) for sample, other in zip(fit_bank.SAMPLES, expected, strict=True) for key in sample)
    assert all(sorted(c.tolist()) == list(range(count)) for c in fit_bank.ORDER.reshape(-1, count))
    assert config['schedule'] == [dict(update=e + 1, learning_rate=r) for e, r in
        [(100, 1e-5), (150, 1e-6), (175, 1e-7), (190, 1e-8)]]
    extras = {name + suffix: None if suffix == '_sample' else 0. for name in
              ('analytical', 'supplemental', 'direction', 'current') for suffix in ('_sample', '_weight')}
    # Literal scalar oracle: mean([6,-2,0,0,0,0,0,0])=.5. Summing gives 4;
    # clipping each case gives mean([1,-1,0,...])=0. Both wrong paths change SGD's answer.
    literal = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        literal.weight.fill_(2.)
    literal_stats = dict(optimizer_examples=0, optimizer_exposures=[0]*8, optimizer_updates=0)
    with patch.object(captured, 'SAMPLES', [6., -2., 0., 0., 0., 0., 0., 0.]), \
            patch.object(captured, 'ORDER', np.arange(8)), \
            patch.object(captured, 'loss', side_effect=lambda n, value, _: n.weight.sum()*value), \
            patch.dict(fine_tuning.STATS, fundamental_bank=literal_stats), \
            patch.object(gradient_clipping, 'clip_l2_', wraps=gradient_clipping.clip_l2_) as clip:
        loss, gradient = fit_bank.one_update(literal, torch.optim.SGD(literal.parameters(), lr=.1), [], [0], {}, **extras)
        assert loss == 1. and gradient == .5 and torch.equal(literal.weight, torch.tensor([[1.95]]))
        assert clip.call_count == 1 and literal_stats['optimizer_updates'] == 1
    before = data.helpers().state_inventory(network)
    params = list(network.parameters())
    reference = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    values, defects = [], []
    network.train()
    for index in fit_bank.ORDER[:count]:
        sample = fit_bank.SAMPLES[int(index)]
        defect = float(torch.linalg.vector_norm(model.full_operator(sample['reference'], sample['cell_size_m'], sample['rho'])-sample['rhs']) / torch.linalg.vector_norm(sample['rhs']))
        assert defect < 1e-10
        defects.append(defect)
        value = captured.loss(network, sample, normalization)
        gradients = torch.autograd.grad(value, params)
        reference.add_(torch.cat([g.detach().reshape(-1).double() for g in gradients]), alpha=1/count)
        values.append(float(value.detach()))
        del value, gradients
        assert data.helpers().state_inventory(network) == before
    assert bool(torch.isfinite(reference).all())
    oracle = copy.deepcopy(network)
    cursor = 0
    for parameter in oracle.parameters():
        parameter.grad = reference[cursor:cursor+parameter.numel()].reshape(parameter.shape).float().clone()
        cursor += parameter.numel()
    torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1., error_if_nonfinite=True)
    torch.optim.Adam(oracle.parameters(), lr=config['learning_rate']).step()
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    observed = {}
    original_loss, original_clip = captured.loss, gradient_clipping.clip_l2_

    def unchanged_loss(net, sample, norm):
        assert data.helpers().state_inventory(net) == before, 'No intermediate update allowed'
        return original_loss(net, sample, norm)

    def clip_once(parameters, limit):
        parameters = list(parameters)
        actual = torch.cat([p.grad.detach().reshape(-1).double() for p in parameters])
        relative = float(torch.linalg.vector_norm(actual-reference) / torch.linalg.vector_norm(reference))
        assert relative <= config['aggregate_gradient_relative_l2_tolerance'], relative
        observed['mean_gradient_relative_l2'] = relative
        return original_clip(parameters, limit)

    with patch.object(captured, 'loss', side_effect=unchanged_loss) as calls, \
            patch.object(gradient_clipping, 'clip_l2_', side_effect=clip_once) as clips, \
            patch.object(optimizer, 'step', wraps=optimizer.step) as steps, \
            patch.object(optimizer, 'zero_grad', wraps=optimizer.zero_grad) as zeros:
        value, gradient = fit_bank.one_update(network, optimizer, [], [0], normalization, **extras)
        assert calls.call_count == count and clips.call_count == steps.call_count == zeros.call_count == 1
    assert math.isclose(value, sum(values)/count, rel_tol=1e-12)
    difference = max(float((p-q).abs().max()) for p, q in zip(network.parameters(), oracle.parameters(), strict=True))
    assert difference <= 2e-7, difference
    assert stats['optimizer_examples'] == 8 and stats['optimizer_exposures'] == [1]*8 and stats['optimizer_updates'] == 1
    assert fine_tuning.STATS['teacher_forwards'] == 0
    assert not torch.cuda.is_initialized() and torch.get_num_threads() == 1
    workspace = 5 * reference.numel() * 8  # Conservative live vector bound, separate from model/Adam states.
    assert workspace <= 16 * 1024**2
    result = read(base)
    result.update(balanced_cycle=dict(status='passed', case_count=count, examples=8, optimizer_updates=1,
        losses=calls.call_count, zero_grad_calls=1, global_clip_calls=1, adam_step_calls=1,
        **observed, gradient_relative_l2_tolerance=config['aggregate_gradient_relative_l2_tolerance'],
        adam_oracle_max_parameter_absolute_error=difference, adam_parameter_absolute_tolerance=2e-7,
        no_intermediate_weight_update=True, literal_mean_and_clipping_oracle=True,
        unchanged_case_tensors=True, independent_operator_defects=defects,
        sampling_order_sha256=stats['sampling_order_sha256'], cases=cases,
        resident_tensor_bytes=stats['resident_tensor_bytes'], gradient_workspace_bound_bytes=workspace,
        expected_total_examples=config['examples'], expected_total_optimizer_updates=config['updates'],
        source_hashes={name:sha(project/name) for name in ('fit_bank.py', 'original_fit_bank.py')}),
        qualifier_source_sha256=sha(__file__), aggregate_qualification_optimizer_updates=1,
        elapsed_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_initialized=False, qualification_only_not_retained_training=True)
    write(output, result)
    print('Qualified balanced cycle', observed, 'Adam max error', difference, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'project', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    check(args.config.resolve(), args.project.resolve(), args.output.resolve())
