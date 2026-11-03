"""Controlled forcing-order collision in the reduced learned feature context.

Synthetic known-control sequences share grouped sums. Fine references here are
the restored eight-stage learned generator, not fresh native solver labels.
This is an input-identifiability diagnostic, never an accuracy/timing admission.
"""
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
from coupled_macrostep_v2 import CoupledMacroBackbone
from nonlinear_coupled_features_v1 import columns
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def context_fields(context):
    result = {f'stage{stage}_{name}': value for stage, state in enumerate(context['features']) for name, value in state.items()}
    result.update({f'provisional{c}': value for c, value in enumerate(context['provisional'])})
    result.update(previous=context['previous'], tracer=context['tracer'])
    return result


def difference(a, b, norm):
    result = {}
    for name, fields in [('pressure', ('pressure',)), ('velocity', ('vx', 'vy', 'vz')), ('tracer', ('tracer',))]:
        x = torch.cat([a[n].double().flatten() for n in fields])
        y = torch.cat([b[n].double().flatten() for n in fields])
        if name == 'pressure': x, y = x-x.mean(), y-y.mean()
        floor = norm['relative_denominator_rms_floor'][name]*math.sqrt(x.numel())
        da, db = x.norm().clamp_min(floor), y.norm().clamp_min(floor)
        result[name] = dict(l2_distance=float((x-y).norm()), maximum_absolute_difference=float((x-y).abs().max()),
            common_prediction_relative_error_lower_bound=float((x-y).norm()/(da+db)))
    return result


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only force-order diagnostic required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen force diagnostic source changed: '+p)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    bank = read(ROOT/reg['bank']); case = next(c for c in bank['trajectories'] if c['identifier'] == reg['identifier'])
    if bank['split'] != 'train' or case['split'] != 'train' or case['reference_substeps'] != 8:
        raise ValueError('Consumed active training current state only')
    if digest(ROOT/case['receipt']) != case['receipt_sha256']: raise ValueError('Training receipt changed')
    scene = read(ROOT/case['receipt'])['scene']; counts = tuple(scene['counts'])
    spec = next(v for v in case['states'] if v['step'] == reg['start'])
    if digest(ROOT/spec['path']) != spec['sha256']: raise ValueError('Current input changed')
    with np.load(ROOT/spec['path'], allow_pickle=False) as archive:
        fields = {n: torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
    before_fields = hashes(fields); norm = read(ROOT/reg['normalization'])
    model = CoupledMacroBackbone(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
    before = inventory(model); original = model.controls.generate_from_packet
    args.output.mkdir(parents=True); began = time.monotonic()
    result = dict(status='running', rows=[], active=None, inference_device='cuda', display_label='GPU (CUDA)',
        gpu_model=torch.cuda.get_device_name(), cpu_model_forward=False, optimizer_updates=0,
        coefficient_fitting=False, latency_claim=False, fresh_native_reference=False,
        target_source='Restored learned eight-stage generator under controlled raw force inputs; not newly native-qualified forcing scenes',
        future_native_labels_read=False, development_or_reserved_used=False, expanded_reserved52_references_opened=0,
        protocol_sha256=digest(args.protocol), source_sha256=reg['source_sha256'])
    def save(): write(args.output/'results.json', result)
    try:
        with torch.no_grad():
            pattern = None
            for start in reg['pattern_known_control_starts']:
                generated = original(counts, model.packet(scene, start, 1, 8).to('cuda'))
                for index, force in enumerate(generated):
                    maximum = max(float(v.abs().max()) for v in force)
                    if maximum > 0.:
                        pattern = [v/maximum for v in force]
                        result['pattern_source'] = dict(known_control_start=start, internal_index=index,
                            original_maximum=maximum, source='Existing known-force generator spatial pattern; amplitude/order controlled synthetically')
                        break
                if pattern is not None: break
            if pattern is None: raise ValueError('No nonzero known-force pattern in frozen diagnostic inputs')
            for amplitude in reg['integrated_force_amplitudes']:
                result['active'] = dict(amplitude=amplitude); save()
                pulse = [v*amplitude for v in pattern]
                contexts, fine, grouped = [], [], []
                for sign in (1., -1.):
                    external = [[torch.zeros_like(v) for v in pulse] for _ in range(8)]
                    external[reg['pulse_indices'][0]] = [v*sign for v in pulse]
                    external[reg['pulse_indices'][1]] = [-v*sign for v in pulse]
                    sums = []
                    at = 0
                    for width in (2, 2, 3, 1):
                        value = external[at]
                        for j in range(1, width): value = [a+b for a, b in zip(value, external[at+j])]
                        sums.extend(value); at += width
                    grouped.append(hashes({str(i): v for i, v in enumerate(sums)}))
                    model.controls.generate_from_packet = lambda requested_counts, packet, value=external: value
                    contexts.append(model.context(fields, scene, reg['start']))
                    fine.append(model.feature_states_grouped(fields, scene, reg['start'], (1,)*8)[0][-1])
                equal_context = hashes(context_fields(contexts[0])) == hashes(context_fields(contexts[1]))
                if grouped[0] != grouped[1] or not equal_context:
                    raise ValueError('Registered force-order pair does not share exact grouped context')
                features_exact = {}
                for name in ('vx', 'vy', 'vz', 'tracer'):
                    a, b = columns(model, contexts[0], name), columns(model, contexts[1], name)
                    features_exact[name] = bool(torch.equal(a, b)); del a, b
                if not all(features_exact.values()): raise ValueError('Equal contexts unexpectedly produced different feature columns')
                metrics = difference(fine[0], fine[1], norm)
                result['rows'].append(dict(amplitude=amplitude, pulse_indices=reg['pulse_indices'],
                    grouped_force_sums_exact=True, all_grouped_context_bits_exact=True, all_feature_columns_exact=features_exact,
                    fine_output_sha256=[hashes(v) for v in fine], fine_output_differences=metrics,
                    any_fine_output_bits_different=hashes(fine[0]) != hashes(fine[1])))
                if hashes(fields) != before_fields or inventory(model) != before:
                    raise ValueError('Diagnostic changed caller input or learned parameters')
                if time.monotonic()-began > reg['maximum_wall_seconds']: raise TimeoutError('Finite force-order diagnostic deadline')
                if (torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > reg['maximum_peak_rss_bytes']):
                    raise ValueError('Force-order diagnostic resource ceiling exceeded')
                print('Force-order collision', amplitude, 'grouped context/feature bits identical; fine differences', metrics, flush=True)
                save()
            for p, h in reg['source_sha256'].items():
                if digest(ROOT/p) != h: raise ValueError('Force diagnostic source changed during execution')
            result.update(status='completed_controlled_force_order_identifiability_diagnostic', active=None,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_diagnostic_seconds=time.monotonic()-began,
                full_generator_target_pass=False, native_accuracy_claim=False)
    except BaseException as e:
        result.update(status='failed', error=repr(e)); raise
    finally:
        model.controls.generate_from_packet = original
        model.close(); save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
