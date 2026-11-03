"""Causal feature and numerical-rank controls; no fitting or model admission."""
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
from coupled_macrostep_features_v1 import columns as previous_columns
from nonlinear_coupled_features_v1 import columns, WIDTHS
from capacity_rank_v3 import compress
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def error(actual, reference):
    return float((actual-reference).norm()/reference.norm().clamp_min(1.))


def scaled(context, factor=1., pressure_bias=0., constant_tracer=False):
    result = dict(context)
    result['features'] = [{n: v.clone()*factor+(pressure_bias if n == 'pressure' else 0.)
                           for n, v in state.items()} for state in context['features']]
    result['provisional'] = [v.clone()*factor for v in context['provisional']]
    result['tracer'] = context['tracer'].clone()*factor
    result['previous'] = context['previous'].clone()*factor+pressure_bias
    if constant_tracer:
        for state in result['features']:
            state['tracer'].fill_(.5)
        result['tracer'].fill_(.5)
    return result


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only feature/rank controls required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen nonlinear feature source changed: '+p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); started = time.monotonic()
    result = dict(status='running', rows=[], graph_rows=[], rank_rows=[], active=None,
        inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
        optimizer_updates=0, coefficient_fitting=False, cpu_model_forward=False,
        future_native_labels_read=False, development_or_reserved_used=False,
        expanded_reserved52_references_opened=0, latency_claim=False,
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    model = None
    def save(): write(args.output/'results.json', result)
    def check(value):
        if not math.isfinite(value) or value > reg['maximum_feature_control_error']:
            raise ValueError('Nonlinear feature metamorphic control failed: '+str(value))
    def bounded():
        if time.monotonic()-started > reg['maximum_wall_seconds']:
            raise TimeoutError('Finite nonlinear feature qualification deadline')
        if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']):
            raise ValueError('Nonlinear feature qualification resource ceiling exceeded')
    try:
        with torch.no_grad():
            # Analytically known rank cases distinguish fake QR span from data span.
            for name in ('orthogonal', 'zero', 'rank_deficient', 'tiny_reachable', 'underdetermined'):
                if name == 'underdetermined':
                    matrix = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], device='cuda', dtype=torch.float64)
                    target = torch.tensor([0., 1.], device='cuda', dtype=torch.float64)
                    expected_rank, expected_bound = 2, 0.
                else:
                    matrix = torch.zeros((10, 3), device='cuda', dtype=torch.float64)
                    target = torch.zeros(10, device='cuda', dtype=torch.float64)
                    if name == 'orthogonal':
                        matrix[:3] = torch.eye(3, device='cuda', dtype=torch.float64); target[3] = 1.
                        expected_rank, expected_bound = 3, 1.
                    elif name == 'zero':
                        target[0] = 1.; expected_rank, expected_bound = 0, 1.
                    elif name == 'rank_deficient':
                        matrix[0, :2] = 1.; target[1] = 1.; expected_rank, expected_bound = 1, 1.
                    else:
                        matrix[0, 0] = 1.; matrix[1, 1] = 1e-18; target[1] = 1.
                        expected_rank, expected_bound = 1, 1.-1e-18*reg['head_l2_norm_limit']
                factor, stats = compress(matrix, target, matrix.new_tensor(1.), reg)
                if stats['effective_rank'] != expected_rank or abs(stats['computed_bounded_head_residual_lower_bound']-expected_bound) > 1e-10:
                    raise ValueError('Known rank/bounded-norm projection case failed: '+name)
                result['rank_rows'].append(dict(case=name, **stats)); del matrix, target, factor
            bank = read(ROOT/reg['bank'])
            if bank['split'] != 'train' or len(bank['trajectories']) != 148:
                raise ValueError('Original training bank required')
            cases = {r['identifier']: r for r in bank['trajectories']}
            model = CoupledMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
            before = inventory(model)
            for index, identifier in enumerate(reg['identifiers']):
                case = cases[identifier]
                if case['split'] != 'train' or case['reference_substeps'] != 8:
                    raise ValueError('Consumed active training input only')
                if digest(ROOT/case['receipt']) != case['receipt_sha256']:
                    raise ValueError('Training receipt changed')
                scene = read(ROOT/case['receipt'])['scene']
                spec = next(v for v in case['states'] if v['step'] == reg['start'])
                if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Current training state changed')
                with np.load(ROOT/spec['path'], allow_pickle=False) as archive:
                    fields = {n: torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
                context = model.context(fields, scene, reg['start'])
                for name in ('vx', 'vy', 'vz', 'tracer'):
                    result['active'] = dict(identifier=identifier, field=name); save()
                    actual = columns(model, context, name)
                    base_width = 91 if name == 'tracer' else 70
                    prefix = previous_columns(model, context, name, True)
                    if (actual.shape[0] != WIDTHS[name] or actual.dtype != torch.float64
                            or actual.device.type != 'cuda' or not bool(actual.isfinite().all())
                            or not torch.equal(actual[:4*base_width], prefix)):
                        raise ValueError('Exact nested conditioned feature prefix/shape required')
                    del prefix
                    doubled = columns(model, scaled(context, factor=2.), name)
                    degrees = ([1]*base_width+[2]*(3*base_width)
                               +([2]*168+[3]*78 if name == 'tracer' else [1]*108+[3]*60))
                    scales = torch.tensor([2.**d for d in degrees], device='cuda', dtype=torch.float64)
                    homogeneous = error(doubled, actual*scales[:, None, None, None]); check(homogeneous); del doubled, scales
                    shifted = columns(model, scaled(context, pressure_bias=1.25), name)
                    gauge_error = error(shifted, actual); check(gauge_error); del shifted
                    if name == 'tracer':
                        constant = columns(model, scaled(context, constant_tracer=True), name)
                        if not bool((constant[364:532] == 0.).all()):
                            raise ValueError('Added pressure/tracer feedback must vanish for constant tracer')
                        del constant
                    else:
                        axis = 3-{'vx':0, 'vy':1, 'vz':2}[name]
                        if bool(actual.select(axis, 0).abs().max() != 0.) or bool(actual.select(axis, actual.shape[axis]-1).abs().max() != 0.):
                            raise ValueError('All velocity columns must obey normal walls')
                    result['rows'].append(dict(identifier=identifier, field=name, columns=WIDTHS[name],
                        nested_prefix_exact=True, polynomial_scaling_error=homogeneous, pressure_gauge_error=gauge_error,
                        counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre']))
                    del actual
                    if index == 0:
                        static = scaled(context)
                        for _ in range(2): columns(model, static, name)
                        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream): captured = columns(model, static, name)
                        torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
                        initial = captured.clone()
                        for field, amplitude in [('pressure', .01), ('vx', .03)]:
                            v = static['features'][-1][field]
                            v.add_(torch.linspace(-amplitude, amplitude, v.numel(), device='cuda', dtype=v.dtype).reshape_as(v))
                        graph.replay(); torch.cuda.synchronize()
                        expected = columns(model, static, name)
                        if torch.equal(initial, expected):
                            raise ValueError('Changed-input graph control did not change features')
                        metric = error(captured, expected); check(metric)
                        result['graph_rows'].append(dict(field=name, changed_input_after_capture=True, relative_error=metric))
                        del graph, captured, expected, initial, static
                    bounded(); save()
                print('Nonlinear coupled features', identifier, 'prefix/polynomial/gauge/wall controls passed', flush=True)
            if inventory(model) != before or len(result['rows']) != 12 or len(result['graph_rows']) != 4:
                raise ValueError('All frozen feature controls and unchanged learned assets required')
            for p, h in reg['source_sha256'].items():
                if digest(ROOT/p) != h: raise ValueError('Feature control source changed during execution')
            result.update(status='passed_nonlinear_coupled_features_rank_and_changed_input_controls', active=None,
                frozen_assets_unchanged=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                elapsed_diagnostic_seconds=time.monotonic()-started, full_generator_target_pass=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        if model is not None: model.close()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
