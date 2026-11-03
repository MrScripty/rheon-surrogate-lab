"""Verify retained fine-control inputs without changing reduced dynamics.

No fine reference trajectory is rerun. The prior forcing collision remains the
target evidence; this qualifies a changed context/input encoding, not a model.
"""
import argparse
import ast
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from coupled_macrostep_v2 import CoupledMacroBackbone
from force_aware_coupled_context_v1 import ForceAwareCoupledBackbone, forcing_contrasts
from diagnose_force_order_identifiability_v1 import hashes, context_fields
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def get_method(path, class_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'context')


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only context qualification required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen context qualification source changed: '+p)
    reference = get_method(ROOT/'experiments/full-ai-data-diversity-v1/coupled_macrostep_v1.py', 'CoupledMacroBackbone')
    changed = get_method(ROOT/'experiments/full-ai-data-diversity-v1/force_aware_coupled_context_v1.py', 'ForceAwareCoupledBackbone')
    for node in ast.walk(changed):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
            node.keywords = [k for k in node.keywords if k.arg not in ('fine_forcing', 'forcing_group_widths')]
    if ast.dump(reference, include_attributes=False) != ast.dump(changed, include_attributes=False):
        raise ValueError('Dynamics changed beyond two retained context fields')
    collision = read(ROOT/reg['force_collision'])
    if collision['status'] != 'completed_controlled_force_order_identifiability_diagnostic' or len(collision['rows']) != 3:
        raise ValueError('Completed frozen force-order collision evidence required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    models = [CoupledMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build']),
              ForceAwareCoupledBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])]
    before = [inventory(m) for m in models]
    if before[0] != before[1]: raise ValueError('Identical restored learned coefficients required')
    bank = read(ROOT/reg['bank']); cases = [c for c in bank['trajectories'] if c['reference_substeps'] == 8]
    if bank['split'] != 'train' or len(bank['trajectories']) != 148 or len(cases) != 29:
        raise ValueError('Full original active training population required')
    args.output.mkdir(parents=True); began = time.monotonic()
    result = dict(status='running', active=None, rows=[], collision_rows=[],
        dynamics_AST_identical_except_retained_fields=True, inference_device='cuda', display_label='GPU (CUDA)',
        gpu_model=torch.cuda.get_device_name(), cpu_model_forward=False, optimizer_updates=0,
        coefficient_fitting=False, future_native_labels_read=False, development_or_reserved_used=False,
        expanded_reserved52_references_opened=0, fine_reference_trajectories_reexecuted=False,
        latency_claim=False, native_accuracy_claim=False, full_generator_target_pass=False,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'results.json', result)
    def load(case, start):
        if case['split'] != 'train' or digest(ROOT/case['receipt']) != case['receipt_sha256']:
            raise ValueError('Training receipt changed')
        spec = next(s for s in case['states'] if s['step'] == start)
        if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Training current state changed')
        with np.load(ROOT/spec['path'], allow_pickle=False) as archive:
            fields = {n: torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
        return read(ROOT/case['receipt'])['scene'], fields
    def bounded():
        if time.monotonic()-began > reg['maximum_wall_seconds']: raise TimeoutError('Finite force-aware context deadline')
        if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']):
            raise ValueError('Context qualification resource ceiling exceeded')
    try:
        with torch.no_grad():
            for case in cases:
                for start in reg['starts']:
                    scene, fields = load(case, start); original_fields = hashes(fields)
                    result['active'] = dict(identifier=case['identifier'], start=start); save()
                    contexts = [m.context(fields, scene, start) for m in models]
                    if hashes(context_fields(contexts[0])) != hashes(context_fields(contexts[1])):
                        raise ValueError('Retaining force inputs changed reduced dynamics context')
                    outputs = [m.emit(c) for m, c in zip(models, contexts)]
                    if hashes(outputs[0]) != hashes(outputs[1]): raise ValueError('Zero-head complete state changed')
                    expected = models[0].controls.generate_from_packet(tuple(scene['counts']), models[0].packet(scene, start, 1, 8).to('cuda'))
                    if any(not torch.equal(a, b) for fa, fb in zip(expected, contexts[1]['fine_forcing']) for a, b in zip(fa, fb)):
                        raise ValueError('Retained ordered forcing differs from original known controls')
                    contrasts = forcing_contrasts(contexts[1])
                    if len(contrasts) != 4 or any(not bool(v.isfinite().all()) for force in contrasts for v in force):
                        raise ValueError('Four complete finite known-control contrasts required')
                    if hashes(fields) != original_fields: raise ValueError('Caller input modified')
                    result['rows'].append(dict(identifier=case['identifier'], start=start, counts_xyz=scene['counts'],
                        cells_per_metre=scene['cells_per_metre'], old_context_bits_exact=True, zero_head_complete_output_bits_exact=True,
                        ordered_fine_force_fields_exact=24, temporal_contrast_fields=12))
                    del contexts, outputs, expected, contrasts, fields
                    bounded(); save()
                print('Force-aware context', case['identifier'], 'both windows retain original dynamics/outputs and all fine forces', flush=True)
            case = next(c for c in cases if c['identifier'] == reg['collision_identifier'])
            scene, fields = load(case, reg['collision_start']); model = models[1]
            original = model.controls.generate_from_packet
            source = collision['pattern_source']
            force = original(tuple(scene['counts']), model.packet(scene, source['known_control_start'], 1, 8).to('cuda'))[source['internal_index']]
            pattern = [v/source['original_maximum'] for v in force]
            try:
                for row in collision['rows']:
                    pulse = [v*row['amplitude'] for v in pattern]; contexts = []
                    for sign in (1., -1.):
                        external = [[torch.zeros_like(v) for v in pulse] for _ in range(8)]
                        a, b = row['pulse_indices']; external[a] = [v*sign for v in pulse]; external[b] = [-v*sign for v in pulse]
                        model.controls.generate_from_packet = lambda counts, packet, values=external: values
                        contexts.append(model.context(fields, scene, reg['collision_start']))
                    if hashes(context_fields(contexts[0])) != hashes(context_fields(contexts[1])):
                        raise ValueError('Prior controlled context collision no longer matches')
                    encoded = [forcing_contrasts(c) for c in contexts]
                    different = any(not torch.equal(a, b) for ca, cb in zip(*encoded) for a, b in zip(ca, cb))
                    if not different: raise ValueError('Force-aware input still aliases opposite pulse ordering')
                    result['collision_rows'].append(dict(amplitude=row['amplitude'], original_context_bits_exact=True,
                        ordered_control_inputs_distinguishable=True, fine_reference_reused_without_reexecution=True))
                # Generic known-control checks: constant histories have no within-group contrasts.
                synthetic = dict(contexts[0]); synthetic['fine_forcing'] = [[v.clone() for v in pulse] for _ in range(8)]
                if any(bool((v != 0.).any()) for step in forcing_contrasts(synthetic) for v in step):
                    raise ValueError('Constant known-force history has nonzero temporal contrasts')
                stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream): captured = forcing_contrasts(synthetic)
                torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
                synthetic['fine_forcing'][4][0].add_(.001)
                graph.replay(); torch.cuda.synchronize(); expected = forcing_contrasts(synthetic)
                if any(not torch.equal(a, b) for ca, cb in zip(captured, expected) for a, b in zip(ca, cb)):
                    raise ValueError('Changed-input control contrast graph stale')
                if not bool((captured[2][0] != 0.).any()): raise ValueError('Changed-input graph did not change contrast')
                result['constant_force_contrasts_zero'] = True
                result['changed_input_contrast_graph_exact'] = True
                del graph, captured, expected
            finally: model.controls.generate_from_packet = original
            if len(result['rows']) != 58 or [inventory(m) for m in models] != before:
                raise ValueError('All active training windows and unchanged learned coefficients required')
            for p, h in reg['source_sha256'].items():
                if digest(ROOT/p) != h: raise ValueError('Context source changed during qualification')
            result.update(status='passed_force_aware_context_inputs_and_zero_head_compatibility', active=None,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_diagnostic_seconds=time.monotonic()-began,
                note='The input distinction is restored; no new head is fitted and old feature columns still ignore the extra controls. This is not a native accuracy or latency result.')
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        for model in models: model.close()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
