"""Qualify the actual paired objective, data, sampling and largest-grid autograd."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/graphics-3d/rheon-surrogate-lab')
HERE = ROOT/'experiments/pressure-methods-v1'
PREVIOUS = ROOT/'experiments/pressure-four-pass-distillation-v1'
sys.path.insert(0, str(PREVIOUS))
from common import read, write, sha, source_inputs


def qualify(project, output):
    started = time.perf_counter()
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    base_source = HERE/'qualify_hybrid_base.py' if (project/'hybrid-method.json').exists() else PREVIOUS/'qualify.py'
    spec = importlib.util.spec_from_file_location('retained_objective_qualifier', base_source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base_path = output.with_name(output.stem+'-base.json')
    if output.exists() or base_path.exists():
        raise ValueError('Fresh qualification outputs required')
    base.qualify(project, base_path)
    import torch
    import model
    import data
    import fine_tuning
    import mixed_order
    paths = source_inputs(project)
    _, parents = data.load(tuple(paths[k].read_bytes() for k in ['dataset','dataset_extra']), read(paths['splits']))
    config = read(project/'config.json')
    normalization = model.fit_normalization(parents, data.fit_normalization)
    network = model.PressureCorrectionNet().eval()
    initial = ROOT/next(r['path'] for r in read(project/'project.json')['data'] if r['id'] == 'initial_weights')
    fine_tuning.initialize(network, initial, data.helpers())
    inventory = data.helpers().state_inventory(network)
    order = mixed_order.order(parents, config)
    if order.shape != (config['training']['updates'], 1, 2):
        raise ValueError('Declared sampler shape differs')
    exposures = mixed_order.STATS['tier_exposures']
    if sum(exposures.values()) != config['training']['updates']:
        raise ValueError('Exposure count differs from declared updates')
    largest = max(parents, key=lambda p:p['arrays']['base'][0].size)
    sample = data.sample(largest, 0)
    network.train()
    loss = model.training_loss(network, sample, normalization).mean()
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise ValueError('Largest-grid objective is not finite/differentiable')
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
        raise ValueError('Largest-grid gradient failed')
    if data.helpers().state_inventory(network) != inventory or torch.cuda.is_initialized():
        raise ValueError('Qualification changed weights or initialized CUDA')
    import hashlib
    result = read(base_path)
    result.update(matched_control=dict(cohort=read(project/'domain-assembly.json')['cohort'], largest_counts_xyz=largest['counts_xyz'],
        largest_grid_loss=float(loss.detach()), finite_largest_grid_backward=True,
        sample_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), sampling=mixed_order.STATS,
        normalization='Fixed original C53 train-only normalization',
        extra_banks='Disabled; recorded physical cases only'),
        source_sha256=sha(__file__), base_qualification_sha256=sha(base_path),
        elapsed_s=time.perf_counter()-started)
    write(output, result)
    print('Qualified actual paired training objective', project.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.project.resolve(), args.output.resolve())
