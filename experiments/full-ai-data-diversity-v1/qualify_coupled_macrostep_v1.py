"""Exact zero-correction fields and train-only full-loss/adjoint CUDA controls."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from coupled_macrostep_v1 import CoupledMacroBackbone, wall
from cuda_native_transfer_v1 import FIELDS
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory
from numerics import fine_action


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def pattern(shape, phase):
    return torch.sin(torch.arange(math.prod(shape), device='cuda', dtype=torch.float64)*.031 + phase).reshape(shape)


def gradient_controls(model, context, truth, norm, reg):
    sizes = dimensions(context['counts'])
    baseline = model.emit(context)
    zeros = [torch.zeros(s, device='cuda', dtype=torch.float64, requires_grad=True) for s in sizes]
    p, velocities = model.response(zeros[:3], context)
    refs = {n: torch.from_numpy(a).to(device='cuda', dtype=torch.float64).flatten() for n, a in truth.items()}
    pressure = baseline['pressure'].detach().double() + p.flatten()
    rhs = fine_action(pressure.reshape(1, 1, *context['shape']),
                      pressure.new_tensor([context['h']]), pressure.new_tensor([context['rho']])).flatten()
    equation = ((rhs-refs['rhs'])/refs['rhs'].norm().clamp_min(1e-30)).square().sum()
    centered, target = pressure-pressure.mean(), refs['pressure']-refs['pressure'].mean()
    pdenom = target.norm().clamp_min(norm['relative_denominator_rms_floor']['pressure']*math.sqrt(target.numel()))
    losses = [equation, ((centered-target)/pdenom).square().sum()]
    for group, names, deltas in [('velocity', ('vx', 'vy', 'vz'), velocities), ('tracer', ('tracer',), [zeros[3]])]:
        predicted = torch.cat([baseline[n].detach().double()+d.flatten() for n, d in zip(names, deltas)])
        target = torch.cat([refs[n] for n in names])
        denominator = target.norm().clamp_min(norm['relative_denominator_rms_floor'][group]*math.sqrt(target.numel()))
        losses.append(((predicted-target)/denominator).square().sum())
    objective = torch.stack(losses).mean()
    gradients = torch.autograd.grad(objective, zeros)
    if not bool(objective.isfinite()) or not all(bool(g.isfinite().all()) for g in gradients):
        raise ValueError('Every original supervised loss and diagnostic gradient must be finite')
    norms = [float(g.norm()) for g in gradients]
    if max(norms) <= 0.: raise ValueError('The zero correction must have a nonzero learning signal')
    # Check the complete learned pressure/velocity response's adjoint against
    # a fixed unfitted direction. No coefficient solution or optimizer update.
    direction = [wall(pattern(s, .3+c), c) for c, s in enumerate(sizes[:3])]
    q = [torch.zeros_like(v, requires_grad=True) for v in direction]
    rp, rv = model.response(q, context)
    duals = [pattern(context['shape'], 1.7), *[pattern(s, 2.1+c) for c, s in enumerate(sizes[:3])]]
    scalar = sum((a.flatten()*b.flatten()).sum() for a, b in zip((rp, *rv), duals))
    adjoints = torch.autograd.grad(scalar, q)
    with torch.no_grad():
        dp, dv = model.response(direction, context)
        lhs = sum((a.flatten()*b.flatten()).sum() for a, b in zip((dp, *dv), duals))
        rhs = sum((a*b).sum() for a, b in zip(direction, adjoints))
        bound = sum(a.norm()*b.norm() for a, b in zip((dp, *dv), duals))
        error = float((lhs-rhs).abs()/bound.clamp_min(1e-30))
    if not math.isfinite(error) or error > reg['maximum_adjoint_norm_relative_error']:
        raise ValueError('Restored learned coupled map failed fixed-vector adjoint consistency')
    return dict(four_metric_squared_losses=[float(v.detach()) for v in losses], mean_objective=float(objective.detach()),
                field_gradient_norms=norms, all_gradients_finite=True, adjoint_norm_relative_error=error,
                scope='F64 affine correction response; actual carrier reconstruction is separatelyF32. This is an operator/gradient diagnostic, not trained-model accuracy.')


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda': raise ValueError('Fresh CUDA-only qualification required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen coupled operator dependency changed: '+p)
    cases = read(ROOT/reg['case_selection'])['cases']
    gold = read(ROOT/reg['grouped_results'])
    expected = {(r['identifier'], r['start'], r['K'], r['lead']): r['output_sha256'] for r in gold['rows'] if r['mode'] == 'grouped4'}
    if len(cases) != 30 or len(expected) != 98: raise ValueError('All original29training/oneconsumeddevelopment grouped controls required')
    os.sched_setaffinity(0, {reg['host_cpu']}); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); started = time.monotonic()
    model = CoupledMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model); norm = read(ROOT/reg['normalization'])
    result = dict(status='running', active=None, rows=[], gradient_rows=[], source_evidence={},
                  inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                  optimizer_updates=0, coefficient_fitting=False, cpu_model_forward=False, latency_claim=False,
                  expanded_reserved52_references_opened=0, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'results.json', result)
    try:
        for case in cases:
            if digest(ROOT/case['receipt']) != case['receipt_sha256']: raise ValueError('Consumed receipt changed')
            scene = read(ROOT/case['receipt'])['scene']; specs = {s['step']: s for s in case['states']}
            def load(step):
                spec = specs[step]; p = ROOT/spec['path']
                if digest(p) != spec['sha256']: raise ValueError('Native current/loss field bytes changed')
                result['source_evidence'][str(p.relative_to(ROOT))] = spec['sha256']
                with np.load(p, allow_pickle=False) as a: return {n: a[n].copy() for n in (*FIELDS, 'rhs')}
            for start in case['diagnostic_starts']:
                initial = load(start); fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                original = hashes(fields); prefix = []
                for k in case['diagnostic_horizons']:
                    result['active'] = dict(identifier=case['identifier'], start=start, K=k); save()
                    outputs = model.predict_zero(fields, scene, start, k)
                    identities = [hashes(o) for o in outputs]
                    if hashes(fields) != original or identities[:len(prefix)] != prefix: raise ValueError('Caller ownership/causal prefixes changed')
                    prefix = identities
                    for lead, identity in enumerate(identities, 1):
                        if identity != expected[(case['identifier'], start, k, lead)]: raise ValueError('Zero coupled correction changed grouped4field bytes')
                        result['rows'].append(dict(identifier=case['identifier'], split=case['split'], counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                                                  start=start, K=k, lead=lead, output_sha256=identity, all_field_bits_exact=True))
                if case['split'] == 'train':
                    context = model.context(fields, scene, start)
                    control = gradient_controls(model, context, load(start+1), norm, reg)
                    result['gradient_rows'].append(dict(identifier=case['identifier'], start=start, counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], **control))
                if time.monotonic()-started > reg['maximum_wall_seconds']: raise TimeoutError('Finite coupled operator qualification deadline')
            if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']): raise ValueError('Coupled qualification memory ceiling exceeded')
            save(); print('Coupled learned operators', case['identifier'], 'zero bits and training adjoints qualified', flush=True)
        if len(result['rows']) != 98 or len(result['gradient_rows']) != 58 or inventory(model) != before:
            raise ValueError('Full zero/gradient populations and unchanged restored assets required')
        for p, h in reg['source_sha256'].items():
            if digest(ROOT/p) != h: raise ValueError('Coupled dependency changed during qualification')
        result.update(status='passed_coupled_zero_identity_and_full_training_gradient_controls', active=None,
                      zero_complete_outputs_exact=98, training_gradient_windows=58, frozen_assets_unchanged=True,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      elapsed_diagnostic_seconds=time.monotonic()-started, full_generator_target_pass=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        model.close(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
