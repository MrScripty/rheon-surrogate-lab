"""Qualify a captured 64-cube model contract without any optimizer update.

Only the shape ceiling changes. Full original small-grid parity and real
large-grid differentiability are checked before preparing managed training.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def prepare(output):
    source = ROOT/'runs/pressure-current-policy-v1/current-late-project-02'
    witness = read(source/'source-witness.json')
    if output.exists():
        raise ValueError('Fresh numerical model probe required')
    for name, expected in witness['sources'].items():
        if sha(source/name) != expected:
            raise ValueError('Captured original source changed')
    output.mkdir(parents=True)
    for name in witness['sources']:
        shutil.copyfile(source/name, output/name)
    edits = []
    for name, before, after in [('policy.py', 'MAX_CELLS=65536', 'MAX_CELLS=262144'),
                               ('portable_policy.py', 'MAX_CELLS = 65536', 'MAX_CELLS = 262144')]:
        text = (output/name).read_text()
        if text.count(before) != 1:
            raise ValueError('Unique captured shape-ceiling seam required')
        (output/name).write_text(text.replace(before, after))
        edits.append(dict(path=name, original=before, replacement=after))
    write(output/'probe.json', dict(schema='rheon-pressure-methods-domain-model-probe-v1',
        kind='Numerical shape/differentiability probe only; not an assembled training Project.',
        original_source=str(source.relative_to(ROOT)), original_witness_sha256=sha(source/'source-witness.json'),
        edits=edits, source_sha256={name:sha(output/name) for name in witness['sources']},
        preparer_sha256=sha(__file__), optimizer_updates=0))
    print('Captured model probe with explicit 262144-cell ceiling', flush=True)


def qualify(project, output):
    tick = time.perf_counter()
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    spec = read(project/'probe.json')
    for name, expected in spec['source_sha256'].items():
        if sha(project/name) != expected:
            raise ValueError('Probe source changed')
    sys.path.insert(0, str(project))
    import torch
    import model
    import data
    model.configure('cpu')
    torch.set_num_threads(1)
    config = read(project/'config.json')
    model.select(config)
    artifacts = ROOT/'runs/pressure-current-policy-v1/current-late-study-03/restore'
    normalization = read(artifacts/'normalization.json')
    inventory = read(artifacts/'history.json')['checkpoints'][f"checkpoint-{config['training']['updates']:03d}.npz"]['inventory']
    network = model.PressureCorrectionNet().eval()
    data.helpers().load_weights(network, artifacts/'weights.npz', inventory)
    def weights_identity():
        result = hashlib.sha256()
        for name, value in network.state_dict().items():
            result.update(name.encode())
            result.update(value.detach().numpy().tobytes())
        return result.hexdigest()
    before = weights_identity()
    reader_spec = importlib.util.spec_from_file_location('pressure_methods_training_view', HERE/'view.py')
    view = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(view)
    _, parents = view.load(ROOT/'runs/pressure-methods-v1/domain-training-view-01')
    checks = []
    with torch.no_grad():
        for parent in parents:
            for i in range(len(parent['frames'])):
                sample = view.sample(parent, i, torch)
                rhs = model.full_operator(sample['reference'], sample['cell_size_m'], sample['rho'])
                base = model.full_operator(sample['base'], sample['cell_size_m'], sample['rho'])
                residual = sample['rhs']-rhs
                reference_norm = float(torch.linalg.vector_norm(residual))
                rhs_norm = float(torch.linalg.vector_norm(sample['rhs']))
                closure_norm = float(torch.linalg.vector_norm(sample['rhs']-base-sample['native_input_residual']))
                if reference_norm > max(1e-12, 1e-8*rhs_norm) or closure_norm > max(1e-12, 1e-10*rhs_norm):
                    raise ValueError('Decoded physical target/input closure failed')
                checks.append(dict(trajectory_id=parent['trajectory_id'], frame=parent['frames'][i]['index'],
                    counts_xyz=parent['counts_xyz'], reference_relative_residual=reference_norm/max(rhs_norm, 1e-300),
                    input_closure_relative=closure_norm/max(rhs_norm, 1e-300)))
    # All consumed small-grid fixed-pass controls must be bit-for-bit unchanged.
    material = read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    inputs = ROOT/material['config']['validation_inputs']
    _, validation = data.load(tuple((inputs/name).read_bytes() for name in
        ['validation_dataset.npz', 'validation_dataset_extra.npz']), read(inputs/'validation_splits.json'))
    old = read(ROOT/'runs/pressure-detached-correction-v1/cpu-01/C53/snapshots.json')
    expected = {(r['trajectory_id'], r['native_frame'], r['passes']):r['relative_residual'] for r in old['rows']}
    import physical_residual
    parity = 0
    with torch.no_grad():
        for parent in validation:
            for i, frame in enumerate(parent['frames']):
                sample = data.sample(parent, i)
                pressure = sample['base']
                for passes in range(1, 9):
                    residual = sample['native_input_residual']-model.full_operator(pressure-sample['base'], sample['cell_size_m'], sample['rho'])
                    pressure = physical_residual.predict_pressure(network, {**sample, 'base':pressure, 'native_input_residual':residual},
                        normalization, model.owner, {**model.CORRECTION, 'steps':1})
                    if passes not in [1, 2, 4, 8]:
                        continue
                    residual = sample['rhs']-model.full_operator(pressure, sample['cell_size_m'], sample['rho'])
                    value = float(torch.linalg.vector_norm(residual))/max(float(torch.linalg.vector_norm(sample['rhs'])), 1e-300)
                    if value != expected[parent['entry']['trajectory_id'], frame['index'], passes]:
                        raise ValueError('Shape-budget extension changed small-grid numerical prediction')
                    parity += 1
    if parity != 512:
        raise ValueError('All 128x4 original fixed-pass comparisons required')
    parent = next(p for p in parents if p['counts_xyz'] == [64]*3)
    sample = view.sample(parent, 0, torch)
    calls = []
    hook = network.register_forward_hook(lambda *_:calls.append(1))
    network.train()
    try:
        prediction = physical_residual.predict_pressure(network, sample, normalization, model.owner,
            {**model.CORRECTION, 'steps':2})
        loss = model.pressure_loss(prediction, sample, normalization).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Large-grid nonfinite training loss')
        loss.backward()
        if len(calls) != 2 or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in network.parameters()):
            raise ValueError('Large-grid two-pass finite gradient qualification failed')
    finally:
        hook.remove()
    if weights_identity() != before or torch.cuda.is_initialized():
        raise ValueError('Qualification changed weights or initialized CUDA')
    write(output, dict(status='passed', optimizer_updates=0, cuda_initialized=False, threads=torch.get_num_threads(),
        source_sha256=sha(__file__), probe_sha256=sha(project/'probe.json'),
        view_sha256=sha(ROOT/'runs/pressure-methods-v1/domain-training-view-01/manifest.json'),
        weights_sha256=sha(artifacts/'weights.npz'), normalization_sha256=sha(artifacts/'normalization.json'),
        small_grid_exact_parity_comparisons=parity, physical_target_checks=checks,
        large_grid=dict(counts_xyz=[64]*3, forward_calls=2, finite_backward=True, loss=float(loss.detach())),
        parameter_bytes_unchanged=True, elapsed_s=time.perf_counter()-tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        qualification_scope='Numerical shape and differentiability, not model accuracy or managed training admission.'))
    print('Qualified expanded model shape, 512 exact parity checks, and 64-cube backward', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'qualify'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.project.resolve())
    else:
        if args.output is None:
            parser.error('--output required for qualification')
        qualify(args.project.resolve(), args.output.resolve())
