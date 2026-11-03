"""Training-only full-grid CUDA QR capacity screen; never fit coefficients.

The numerical equation action is supervised loss construction only. This
script cannot be imported by a deployed model to obtain native predictions.
"""
import argparse
import collections
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from macrostep_backbone_v1 import FrozenMacroBackbone
from macrostep_spatial_features_v1 import columns
from numerics import fine_action
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


@torch.no_grad()
def compress(matrix, target, denominator, reg):
    """Stream bounded rows into QR; validate three fixed, unfitted vectors."""
    n, width = matrix.shape
    controls = torch.stack((torch.zeros(width, device='cuda', dtype=torch.float64),
                            torch.linspace(-.25, .25, width, device='cuda', dtype=torch.float64),
                            torch.full((width,), .1, device='cuda', dtype=torch.float64)))
    direct = torch.zeros(3, device='cuda', dtype=torch.float64)
    factor = torch.empty((0, width + 1), device='cuda', dtype=torch.float64)
    for start in range(0, n, reg['maximum_design_rows_per_chunk']):
        end = min(n, start + reg['maximum_design_rows_per_chunk'])
        block = torch.cat((matrix[start:end], target[start:end, None]), dim=1) / denominator
        residuals = block[:, :-1] @ controls.T - block[:, -1:]
        direct += residuals.square().sum(dim=0)
        _, factor = torch.linalg.qr(torch.cat((factor, block), dim=0), mode='r')
    vectors = torch.cat((controls, -torch.ones((3, 1), dtype=torch.float64, device='cuda')), dim=1)
    compressed = (factor @ vectors.T).square().sum(dim=0)
    errors = (direct - compressed).abs()
    allowed = reg['compression_relative_tolerance'] * direct.abs() + reg['compression_absolute_tolerance']
    if not bool(factor.isfinite().all()) or not bool((errors <= allowed).all()):
        raise ValueError('Full-grid residual norms changed during bounded QR')
    if factor.shape != (width + 1, width + 1):
        raise ValueError('Full finite QR factor shape required')
    # This is an optimistic computed projection diagnostic, not a solved head.
    # Rank deficiency can only make the last-diagonal diagnostic too optimistic.
    singular = torch.linalg.svdvals(factor[:, :-1])
    rank = int((singular > singular.max() * max(n, width) * torch.finfo(torch.float64).eps).sum())
    return factor, dict(computed_orthogonal_residual=float(factor[-1, -1].abs()),
                        zero_head_relative_error=float(direct[0].sqrt()),
                        computed_feature_rank=rank, feature_columns=width, full_grid_rows=n,
                        control_vectors=3, maximum_absolute_compression_error=float(errors.max()))


def run(args):
    reg = read(args.protocol)
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only screen required')
    for p, h in reg['source_sha256'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen screen dependency changed: ' + p)
    bank = read(ROOT / reg['bank'])
    if bank['split'] != 'train' or len(bank['trajectories']) != 148:
        raise ValueError('Original full148training bank required')
    cases = [(i, c) for i, c in enumerate(bank['trajectories']) if c['reference_substeps'] == 8]
    if len(cases) != 29 or any(c['split'] != 'train' for _, c in cases):
        raise ValueError('All29N8training scenes and no development/reserved scene required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    model = FrozenMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model)
    norm = read(ROOT / reg['normalization'])
    progress = dict(status='running', active=None, rows=[], parts=[], source_evidence={}, scenes_completed=0,
                    inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
                    cpu_model_forward=False, optimizer_updates=0, local_coefficient_fitting=False,
                    development_or_reserved_used=False, expanded_reserved52_references_opened=0,
                    latency_claim=False, protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    def save(): write(args.output / 'progress.json', progress)
    def bounded():
        if time.monotonic() - started > reg['maximum_screen_wall_seconds']:
            raise TimeoutError('Finite spatial capacity screen deadline')
        if torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
            raise ValueError('GPU screen allocation ceiling exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > reg['maximum_peak_rss_bytes']:
            raise ValueError('Host screen RSS ceiling exceeded')
    try:
        with torch.no_grad():
            for index, case in cases:
                if digest(ROOT / case['receipt']) != case['receipt_sha256']:
                    raise ValueError('Training receipt changed')
                scene = read(ROOT / case['receipt'])['scene']
                specs = {s['step']: s for s in case['states']}
                def load(step):
                    spec = specs[step]; p = ROOT / spec['path']
                    if digest(p) != spec['sha256']: raise ValueError('Native training fields changed')
                    progress['source_evidence'][str(p.relative_to(ROOT))] = spec['sha256']
                    with np.load(p, allow_pickle=False) as a:
                        return {n: a[n].copy() for n in (*FIELDS, 'rhs')}
                for start in reg['starts']:
                    initial = load(start)
                    fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    features = model.feature_states(fields, scene, start, 3)[0]
                    truth = load(start + 1)
                    refs = {n: torch.from_numpy(a).to(device='cuda', dtype=torch.float64).flatten() for n, a in truth.items()}
                    velocity_denominator = torch.cat([refs[n] for n in ('vx', 'vy', 'vz')]).norm().clamp_min(
                        norm['relative_denominator_rms_floor']['velocity'] * math.sqrt(sum(refs[n].numel() for n in ('vx', 'vy', 'vz'))))
                    shape = (1, 1, *reversed(scene['counts']))
                    h = torch.tensor([1. / scene['cells_per_metre']], device='cuda', dtype=torch.float64)
                    rho = torch.tensor([1000.], device='cuda', dtype=torch.float64)
                    for conditioned in (False, True):
                        basis = 'velocity_conditioned_spatial' if conditioned else 'temporal_spatial'
                        factors = {}; stats = {}
                        for name in FIELDS:
                            progress['active'] = dict(identifier=case['identifier'], start=start, basis=basis, field=name)
                            save(); bounded()
                            values = columns(features, name, scene, model.model, conditioned)
                            anchor = features[-1][name].double().flatten()
                            if name == 'pressure':
                                # Reference operator used only for supervised equation loss.
                                action = torch.empty_like(values)
                                for col in range(0, values.shape[0], reg['equation_columns_per_chunk']):
                                    end = min(values.shape[0], col + reg['equation_columns_per_chunk'])
                                    action[col:end] = fine_action(values[col:end, None], h, rho)[:, 0]
                                base_action = fine_action(anchor.reshape(shape), h, rho).flatten()
                                q, st = compress(action.flatten(1).T, refs['rhs'] - base_action,
                                                 refs['rhs'].norm().clamp_min(1e-30), reg)
                                factors['equation'] = q.cpu().numpy().copy(); stats['equation'] = st
                                del action, q
                                values -= values.flatten(1).mean(dim=1)[:, None, None, None]
                                target = refs[name] - refs[name].mean()
                                baseline = anchor - anchor.mean()
                                denominator = target.norm().clamp_min(norm['relative_denominator_rms_floor'][name] * math.sqrt(target.numel()))
                            else:
                                target = refs[name]; baseline = anchor
                                denominator = velocity_denominator if name in ('vx', 'vy', 'vz') else target.norm().clamp_min(
                                    norm['relative_denominator_rms_floor']['tracer'] * math.sqrt(target.numel()))
                            q, st = compress(values.flatten(1).T, target - baseline, denominator, reg)
                            factors[name] = q.cpu().numpy().copy(); stats[name] = st
                            del q, values
                            bounded()
                        stats['velocity'] = dict(computed_orthogonal_residual=math.sqrt(sum(stats[n]['computed_orthogonal_residual'] ** 2 for n in ('vx', 'vy', 'vz'))),
                                                 interpretation='Independent per-component heads; combined original velocity denominator.')
                        path = args.output / f'scene-{index:03d}-start-{start:03d}-{basis}.npz'
                        np.savez_compressed(path, **factors)
                        progress['parts'].append(dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size,
                                                      arrays={n: dict(dtype=a.dtype.str, shape=list(a.shape)) for n, a in factors.items()}))
                        progress['rows'].append(dict(identifier=case['identifier'], scene_index=index, group=case['group'],
                                                     counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'], start=start,
                                                     teacher_substeps=8, student_stages=3, basis=basis, metrics=stats, part=path.name))
                        del factors
                        save()
                    del features, fields, initial, truth, refs
                    print('Spatial full-grid training capacity', case['identifier'], start, 'both bases complete', flush=True)
                progress['scenes_completed'] += 1; save()
        if len(progress['rows']) != 116 or inventory(model) != before:
            raise ValueError('Every29scene/two-window/two-basis design and unchanged assets required')
        for p, hsh in reg['source_sha256'].items():
            if digest(ROOT / p) != hsh: raise ValueError('Screen dependency changed during execution')
        summary = []
        for basis in reg['feature_bases']:
            rows = [r for r in progress['rows'] if r['basis'] == basis]
            summary.append(dict(basis=basis, training_windows=len(rows), metrics={m: dict(
                maximum_computed_orthogonal_residual=max(r['metrics'][m]['computed_orthogonal_residual'] for r in rows),
                training_windows_exceeding_1e5=sum(r['metrics'][m]['computed_orthogonal_residual'] > 1e-5 for r in rows))
                for m in ('equation', 'pressure', 'velocity', 'tracer')}))
        progress.update(status='completed_all29N8_two_window_two_basis_spatial_capacity_screen', active=None,
                        frozen_assets_unchanged=True, summaries=summary, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        elapsed_diagnostic_seconds=time.monotonic() - started, full_generator_target_pass=False,
                        limitations='Computed F64 per-example/per-metric optimistic feature-span diagnostic. No coefficients solved, no managed fit, no deployed-head accuracy or timing qualification. Rank-deficient QR may be optimistic. All119N1training regimes remain fixed controls; this screen only studies29activeN8scenes.')
        write(args.output / 'results.json', progress)
    except BaseException as e:
        progress.update(status='failed', error=repr(e)); raise
    finally:
        model.close(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
