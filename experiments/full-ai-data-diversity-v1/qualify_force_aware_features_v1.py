"""Qualify force-conditioned inputs against frozen causal/current-state controls."""
import argparse
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from force_aware_coupled_context_v1 import ForceAwareCoupledBackbone
from force_aware_coupled_features_v1 import columns, WIDTHS, PREFIX
from nonlinear_coupled_features_v1 import columns as previous_columns
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def error(a, b):
    return float((a-b).norm()/b.norm().clamp_min(1.))


def altered(context, factor=1., constant=False, zero=False, gauge=0., tracer=False):
    result = dict(context)
    forces = context['fine_forcing']
    result['fine_forcing'] = [[(torch.zeros_like(v) if zero else v.clone()*factor)
                              for v in forces[0 if constant else i]] for i in range(8)]
    if gauge or tracer:
        result['features'] = [{n: v.clone() for n, v in state.items()} for state in context['features']]
        for state in result['features']:
            state['pressure'].add_(gauge)
            if tracer: state['tracer'].fill_(.5)
    return result


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only feature qualification required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen force-aware dependency changed: '+p)
    context_gate = read(ROOT/reg['context_qualification'])
    if context_gate['status'] != 'passed_force_aware_context_inputs_and_zero_head_compatibility' or len(context_gate['rows']) != 58:
        raise ValueError('Complete force-aware context qualification required')
    bank = read(ROOT/reg['bank']); cases = {c['identifier']: c for c in bank['trajectories']}
    if bank['split'] != 'train' or len(cases) != 148: raise ValueError('Full original training bank required')
    os.sched_setaffinity(0, {reg['host_cpu']}); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True); began = time.monotonic(); model = None
    result = dict(status='running', active=None, rows=[], graph_rows=[], collision_rows=[],
        inference_device='cuda', display_label='GPU (CUDA)', gpu_model=torch.cuda.get_device_name(),
        optimizer_updates=0, coefficient_fitting=False, cpu_model_forward=False,
        future_native_labels_read=False, development_or_reserved_used=False,
        expanded_reserved52_references_opened=0, fine_reference_trajectories_reexecuted=False,
        latency_claim=False, native_accuracy_claim=False, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'results.json', result)
    def check(e):
        if not math.isfinite(e) or e > reg['maximum_feature_control_error']:
            raise ValueError('Force feature control failed: '+str(e))
    def bounded():
        if time.monotonic()-began > reg['maximum_wall_seconds']: raise TimeoutError('Force feature deadline')
        if torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']:
            raise ValueError('Force feature qualification resource ceiling')
    def load(identifier):
        case = cases[identifier]
        if case['split'] != 'train' or case['reference_substeps'] != 8 or digest(ROOT/case['receipt']) != case['receipt_sha256']:
            raise ValueError('Consumed active training input required')
        spec = next(v for v in case['states'] if v['step'] == reg['start'])
        if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Current state changed')
        with np.load(ROOT/spec['path'], allow_pickle=False) as archive:
            fields = {n: torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
        return read(ROOT/case['receipt'])['scene'], fields
    try:
        with torch.no_grad():
            model = ForceAwareCoupledBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
            before = inventory(model)
            for index, identifier in enumerate(reg['identifiers']):
                scene, fields = load(identifier); context = model.context(fields, scene, reg['start'])
                for name in FIELDS[1:]:
                    result['active'] = dict(identifier=identifier, field=name); save()
                    actual = columns(model, context, name); prefix = previous_columns(model, context, name)
                    if actual.shape[0] != WIDTHS[name] or not torch.equal(actual[:PREFIX[name]], prefix) or not bool(actual.isfinite().all()):
                        raise ValueError('Exact prefix/width/finite CUDA feature required')
                    del prefix
                    for tag, changed in [('zero_force', altered(context, zero=True)), ('constant_force', altered(context, constant=True))]:
                        value = columns(model, changed, name)
                        if not torch.equal(value[:PREFIX[name]], actual[:PREFIX[name]]) or bool(value[PREFIX[name]:].abs().max() != 0.):
                            raise ValueError(tag+' must preserve prefix and zero new columns')
                        del value, changed
                    reverse = columns(model, altered(context, factor=-1.), name)
                    sign_error = error(reverse[PREFIX[name]:], -actual[PREFIX[name]:]); check(sign_error); del reverse
                    biased = columns(model, altered(context, gauge=1.25), name)
                    gauge_error = error(biased, actual); check(gauge_error); del biased
                    if name == 'tracer':
                        uniform = columns(model, altered(context, tracer=True), name)
                        if bool(uniform[PREFIX[name]:].abs().max() != 0.): raise ValueError('Uniform tracer must zero force displacement columns')
                        del uniform
                    else:
                        axis = 3-{'vx':0, 'vy':1, 'vz':2}[name]
                        if bool(actual.select(axis,0).abs().max() != 0.) or bool(actual.select(axis, actual.shape[axis]-1).abs().max() != 0.):
                            raise ValueError('All velocity columns must satisfy normal walls')
                    result['rows'].append(dict(identifier=identifier, field=name, columns=WIDTHS[name],
                        prefix_exact=True, zero_force_new_columns_zero=True, constant_force_new_columns_zero=True,
                        force_sign_error=sign_error, pressure_gauge_error=gauge_error,
                        counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre']))
                    del actual
                    if index == 0:
                        static = altered(context)
                        for _ in range(2): columns(model, static, name)
                        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream()); graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream): captured = columns(model, static, name)
                        torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize(); initial = captured[PREFIX[name]:].clone()
                        for v in static['fine_forcing'][4]:
                            v.add_(torch.linspace(-.01, .02, v.numel(), device='cuda', dtype=v.dtype).reshape_as(v))
                        graph.replay(); torch.cuda.synchronize(); expected = columns(model, static, name)
                        if torch.equal(initial, expected[PREFIX[name]:]): raise ValueError('Known-force graph input failed to change added channels')
                        metric = error(captured, expected); check(metric)
                        result['graph_rows'].append(dict(field=name, changed_force_inputs_after_capture=True, relative_error=metric))
                        del graph, captured, expected, initial, static
                    bounded(); save()
                print('Force-aware features', identifier, 'prefix/force/gauge/wall controls passed', flush=True)
            collision = read(ROOT/reg['force_collision']); scene, fields = load(reg['collision_identifier'])
            original = model.controls.generate_from_packet; source = collision['pattern_source']
            raw = original(tuple(scene['counts']), model.packet(scene, source['known_control_start'], 1, 8).to('cuda'))[source['internal_index']]
            pattern = [v/source['original_maximum'] for v in raw]
            try:
                for row in collision['rows']:
                    contexts = []
                    for sign in (1., -1.):
                        forces = [[torch.zeros_like(v) for v in pattern] for _ in range(8)]
                        a,b = row['pulse_indices']; forces[a] = [v*(sign*row['amplitude']) for v in pattern]; forces[b] = [-v for v in forces[a]]
                        model.controls.generate_from_packet = lambda counts, packet, values=forces: values
                        contexts.append(model.context(fields, scene, reg['start']))
                    field_rows = []
                    for name in FIELDS[1:]:
                        a,b = [columns(model, c, name) for c in contexts]
                        if not torch.equal(a[:PREFIX[name]], b[:PREFIX[name]]): raise ValueError('Prior input collision prefix changed')
                        field_rows.append(dict(field=name, prefix_exact=True, added_columns_different=not torch.equal(a[PREFIX[name]:],b[PREFIX[name]:])))
                        del a,b
                    if not any(r['added_columns_different'] for r in field_rows): raise ValueError('Force-aware feature set still aliases controlled pulse ordering')
                    result['collision_rows'].append(dict(amplitude=row['amplitude'], fields=field_rows, added_features_distinguish_order=True))
                    bounded(); save()
            finally: model.controls.generate_from_packet = original
            if len(result['rows']) != 12 or len(result['graph_rows']) != 4 or len(result['collision_rows']) != 3 or inventory(model) != before:
                raise ValueError('Complete controls and unchanged learned assets required')
            for p,h in reg['source_sha256'].items():
                if digest(ROOT/p) != h: raise ValueError('Frozen force-aware source changed during qualification')
            result.update(status='passed_force_aware_features_and_changed_force_graph_controls',active=None,
                frozen_assets_unchanged=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                elapsed_diagnostic_seconds=time.monotonic()-began)
    except BaseException as e:
        result.update(status='failed',error=repr(e)); raise
    finally:
        if model is not None: model.close()
        save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
