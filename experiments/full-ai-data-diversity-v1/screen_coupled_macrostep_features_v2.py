"""Train-only full-grid necessary equation/tracer capacity screen, no fitting.

Equation columns include the actual restored learned pressure forward followed
by the native equation operator in this loss-only script. No numerical inverse
is constructed. Full pressure/velocity qualification remains required later.
"""
import argparse
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from coupled_macrostep_v2 import CoupledMacroBackbone
from coupled_macrostep_features_v2 import columns
from capacity_qr_v2 import compress
from cuda_native_transfer_v1 import FIELDS
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory
from numerics import fine_action


def controls(width):
    return torch.stack((torch.zeros(width, device='cuda', dtype=torch.float64),
                        torch.linspace(-.25, .25, width, device='cuda', dtype=torch.float64),
                        torch.full((width,), .1, device='cuda', dtype=torch.float64)))


def norms_check(factor, vectors, direct, reg):
    augmented = torch.cat((vectors, -torch.ones((3, 1), dtype=torch.float64, device='cuda')), dim=1)
    predicted = (factor@augmented.T).square().sum(dim=0)
    error = (predicted-direct).abs()
    if not bool((error <= reg['full_map_control_relative_tolerance']*direct.abs()+reg['full_map_control_absolute_tolerance']).all()):
        raise ValueError('Actual coupled full-map losses disagree with affine feature compression')
    return float(error.max())


@torch.no_grad()
def equation_design(model, context, baseline, truth, conditioned, reg):
    width = 490; total = 3*width
    shape = context['shape']; sizes = dimensions(context['counts'])[:3]
    matrix = torch.empty((total, math.prod(shape)), dtype=torch.float64, device='cuda')
    vectors = controls(total)
    directions = [[torch.zeros(s, dtype=torch.float64, device='cuda') for s in sizes] for _ in range(3)]
    h = torch.tensor([context['h']], device='cuda', dtype=torch.float64)
    rho = torch.tensor([context['rho']], device='cuda', dtype=torch.float64)
    for component, name in enumerate(('vx', 'vy', 'vz')):
        features = columns(model, context, name, conditioned)
        if features.shape[0] != width: raise ValueError('Exact coupled velocity feature width required')
        for control in range(3):
            directions[control][component] = torch.einsum('f,fzyx->zyx', vectors[control, component*width:(component+1)*width], features)
        for start in range(0, width, reg['feature_columns_per_chunk']):
            end = min(width, start+reg['feature_columns_per_chunk']); batch = end-start
            inputs = [torch.zeros((batch, *s), dtype=torch.float64, device='cuda') for s in sizes]
            inputs[component] = features[start:end]
            responses = model.rhs(inputs, context)
            for i in range(batch):
                pressure = model.pressure((responses[i]*(context['rho']/context['h'])).reshape(1, 1, *shape), context['counts'])
                # This native action constructs a supervised loss column only.
                matrix[component*width+start+i] = fine_action(pressure, h, rho).flatten()
        del features, inputs, responses
    target = truth['rhs']-fine_action(baseline['pressure'].reshape(1, 1, *shape), h, rho).flatten()
    denominator = truth['rhs'].norm().clamp_min(1e-30)
    factor, stats = compress(matrix.T, target, denominator, reg)
    del matrix
    direct = []
    for direction in directions:
        output = model.emit(context, direction, float_carriers=False)
        residual = (fine_action(output['pressure'].reshape(1, 1, *shape), h, rho).flatten()-truth['rhs'])/denominator
        direct.append(residual.square().sum())
    stats['full_coupled_map_control_vectors'] = 3
    stats['maximum_full_map_absolute_norm_error'] = norms_check(factor, vectors, torch.stack(direct), reg)
    return factor, stats


@torch.no_grad()
def tracer_design(model, context, truth, norm, conditioned, reg):
    features = columns(model, context, 'tracer', conditioned)
    width = 637
    if features.shape[0] != width: raise ValueError('Exact coupled tracer feature width required')
    target = truth['tracer']-context['tracer'].double().flatten()
    denominator = truth['tracer'].norm().clamp_min(norm['relative_denominator_rms_floor']['tracer']*math.sqrt(target.numel()))
    factor, stats = compress(features.flatten(1).T, target, denominator, reg)
    vectors = controls(width); direct = []
    for vector in vectors:
        direction = torch.einsum('f,fzyx->zyx', vector, features)
        output = model.emit(context, tracer_delta=direction, float_carriers=False)
        direct.append(((output['tracer']-truth['tracer'])/denominator).square().sum())
    stats['full_coupled_map_control_vectors'] = 3
    stats['maximum_full_map_absolute_norm_error'] = norms_check(factor, vectors, torch.stack(direct), reg)
    return factor, stats


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh CUDA-only capacity screen required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen coupled capacity dependency changed: '+p)
    gate = read(ROOT/reg['operator_qualification'])
    if gate['status'] != 'passed_coupled_zero_identity_and_full_training_gradient_controls' or gate['zero_complete_outputs_exact'] != 98 or gate['training_gradient_windows'] != 58:
        raise ValueError('Actual exact zero fields and every training gradient window required')
    bank = read(ROOT/reg['bank'])
    if bank['split'] != 'train' or len(bank['trajectories']) != 148: raise ValueError('Original full148training bank required')
    cases = [(i, c) for i, c in enumerate(bank['trajectories']) if c['reference_substeps'] == 8]
    if len(cases) != 29 or any(c['split'] != 'train' for _, c in cases): raise ValueError('Every29active training scene required')
    os.sched_setaffinity(0, {reg['host_cpu']}); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); started = time.monotonic()
    model = CoupledMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model); norm = read(ROOT/reg['normalization'])
    result = dict(status='running', active=None, rows=[], parts=[], scenes_completed=0, source_evidence={},
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  optimizer_updates=0, coefficient_fitting=False, cpu_model_forward=False, latency_claim=False,
                  development_or_reserved_used=False, expanded_reserved52_references_opened=0,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'progress.json', result)
    def bounded():
        if time.monotonic()-started > reg['maximum_wall_seconds']: raise TimeoutError('Finite coupled capacity screen deadline')
        if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']
                or sum(p['bytes'] for p in result['parts']) > reg['maximum_output_bytes']): raise ValueError('Coupled screen resource ceiling exceeded')
    try:
        with torch.no_grad():
            for index, case in cases:
                if digest(ROOT/case['receipt']) != case['receipt_sha256']: raise ValueError('Training receipt changed')
                scene = read(ROOT/case['receipt'])['scene']; specs = {s['step']: s for s in case['states']}
                def load(step):
                    spec = specs[step]; p = ROOT/spec['path']
                    if digest(p) != spec['sha256']: raise ValueError('Native training field bytes changed')
                    result['source_evidence'][str(p.relative_to(ROOT))] = spec['sha256']
                    with np.load(p, allow_pickle=False) as a: return {n: a[n].copy() for n in (*FIELDS, 'rhs')}
                for start in reg['starts']:
                    initial = load(start); fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    context = model.context(fields, scene, start); baseline = model.emit(context)
                    truth = {n: torch.from_numpy(a).to(device='cuda', dtype=torch.float64).flatten() for n, a in load(start+1).items()}
                    for conditioned in ('absolute_velocity',):
                        basis = 'cross_carrier_absolute_velocity'
                        factors = {}; metrics = {}
                        result['active'] = dict(identifier=case['identifier'], start=start, basis=basis, metric='equation'); save(); bounded()
                        q, stats = equation_design(model, context, baseline, truth, conditioned, reg)
                        factors['equation'] = q.cpu().numpy().copy(); metrics['equation'] = stats; del q
                        result['active']['metric'] = 'tracer'; save(); bounded()
                        q, stats = tracer_design(model, context, truth, norm, conditioned, reg)
                        factors['tracer'] = q.cpu().numpy().copy(); metrics['tracer'] = stats; del q
                        path = args.output/f'scene-{index:03d}-start-{start:03d}-{basis}.npz'
                        np.savez_compressed(path, **factors)
                        result['parts'].append(dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size,
                                                   arrays={n: dict(dtype=a.dtype.str, shape=list(a.shape)) for n, a in factors.items()}))
                        result['rows'].append(dict(identifier=case['identifier'], scene_index=index, group=case['group'], start=start,
                                                  counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], basis=basis, metrics=metrics, part=path.name))
                        del factors; bounded(); save()
                    print('Coupled full-grid training capacity', case['identifier'], start, 'both feature bases complete', flush=True)
                result['scenes_completed'] += 1; save()
        if len(result['rows']) != 58 or inventory(model) != before: raise ValueError('All29training/two-window/single-nonlinear-basis scope and unchanged assets required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT/p) != h: raise ValueError('Coupled screen dependency changed during execution')
        summaries = []
        for basis in reg['feature_bases']:
            rows = [r for r in result['rows'] if r['basis'] == basis]
            summaries.append(dict(basis=basis, training_windows=len(rows), metrics={m: dict(
                maximum_computed_orthogonal_residual=max(r['metrics'][m]['computed_orthogonal_residual'] for r in rows),
                training_windows_exceeding_1e5=sum(r['metrics'][m]['computed_orthogonal_residual'] > 1e-5 for r in rows)) for m in ('equation', 'tracer')}))
        result.update(status='completed_coupled_equation_tracer_necessary_capacity_screen', active=None, summaries=summaries,
                      frozen_assets_unchanged=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      elapsed_diagnostic_seconds=time.monotonic()-started, full_generator_target_pass=False,
                      limitations='Necessary optimistic per-example equation/tracer F64affine capacity screen only. Pressure/velocity fit losses and deployedF32carrier/native240/K/seed/independent/timing verification remain required. No coefficients solved or local/managed optimizer. All119N1training regimes remain fixed controls; every148scene required in any later fit.')
        write(args.output/'results.json', result)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        model.close(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
