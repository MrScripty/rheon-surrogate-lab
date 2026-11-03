"""Native long-data correction expressibility and coupled gradient controls.

Teacher-injected corrections are offline labels, not deployable predictions.
The candidate phase/response sees no future fields or target equation RHS.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import LongTemporalDataset, FIELDS
from temporal_group_projector_v1 import TemporalGroupProjector
from temporal_control_packets_v1 import control_packet
from recurrent_sampler_native import dimensions
from coupled_macrostep_v1 import wall
from internal_residual_loss_v1 import metrics as physical_losses
from diagnose_causal_composition import inventory, measure


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only target qualification required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen target qualification source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    model = None
    started = time.monotonic()
    result = dict(status='running', rows=[], cpu_model_forward=False, new_parameter_fits=0,
        inference_device='cuda', display_label='GPU (CUDA)', teacher_derived_oracle_injection=True,
        deployable_candidate=False, latency_claim=False, reserved52_references_opened=0,
        full_generator_target_pass=False, source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol))
    try:
        dataset = LongTemporalDataset(read(ROOT / reg['index']))
        model = TemporalGroupProjector(reg['transport_pair'], reg['shared_pair'], reg['build'], reg['control_build'])
        before = inventory(model)
        norm = read(ROOT / reg['normalization'])
        for identifier, (_, receipt, _) in dataset.receipts.items():
            scene = receipt['scene']
            counts, shape = tuple(scene['counts']), tuple(reversed(scene['counts']))
            sizes = dimensions(counts)
            shapes = dict(pressure=shape, **dict(zip(FIELDS[1:], sizes)))
            for start in reg['dense_starts']:
                for width in reg['widths']:
                    if time.monotonic() - started > reg['maximum_wall_seconds']:
                        raise TimeoutError('Finite native target qualification wall limit')
                    initial = dataset._load(identifier, start)
                    endpoint = dataset._load(identifier, start + width)
                    last = dataset._load(identifier, start + width - 1)
                    fields = {n: torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    original = hashes(fields)
                    phase = model.phase(fields, scene, start, width)
                    with torch.no_grad():
                        zero = model.emit(phase)
                        packet = torch.from_numpy(control_packet(scene, start, width)).to('cuda')
                        coarse = model.predict_packet(fields, scene, 1, (width,), packet)[0]
                        if hashes(zero) != hashes(coarse):
                            raise ValueError('Differentiable zero path changed qualified coarse bytes')
                        last_fields = {n: torch.from_numpy(last[n]).to('cuda') for n in FIELDS}
                        fine_phase = model.phase(last_fields, scene, start + width - 1, 1)
                        fine_p = torch.from_numpy(endpoint['pressure']).to('cuda').reshape(shape)
                        momentum = [wall(model.linear([fine_p], s, 'momentum', c)
                            * (1 / (phase['rho'] * phase['h'])), c) for c, s in enumerate(sizes[:3])]
                        required = [v.double() + (phase['reference_dt_s'] - phase['dt']) * m
                                    for v, m in zip(fine_phase['provisional'], momentum)]
                        delta_v = [a - b.double() for a, b in zip(required, phase['provisional'])]
                        target_t = torch.from_numpy(endpoint['tracer']).to('cuda').reshape(shape)
                        delta_t = target_t.double() - phase['base_tracer'].double()
                        injected = model.emit(phase, delta_v, delta_t)
                        measured = measure({n: v.cpu() for n, v in injected.items()}, endpoint, scene, norm)
                        baseline = measure({n: v.cpu() for n, v in zero.items()}, endpoint, scene, norm)
                    sample = dict(h=phase['h'], rho=phase['rho'], target_rhs=torch.from_numpy(endpoint['rhs']).to('cuda').reshape(shape))
                    for n in FIELDS:
                        sample['target_' + n] = torch.from_numpy(endpoint[n]).to('cuda').reshape(shapes[n])

                    def loss(alpha):
                        output = model.emit(phase, [alpha * v for v in delta_v], alpha * delta_t, float_carriers=False)
                        values = physical_losses({n: v.reshape(shapes[n]) for n, v in output.items()}, sample, norm)
                        return sum(values.values()), values

                    alpha = torch.zeros((), device='cuda', dtype=torch.float64, requires_grad=True)
                    objective, components = loss(alpha)
                    objective.backward()
                    gradient = float(alpha.grad)
                    epsilon = reg['gradient_epsilon']
                    with torch.no_grad():
                        negative, _ = loss(-epsilon)
                        positive, _ = loss(epsilon)
                    numerical = float((positive - negative) / (2 * epsilon))
                    gap = abs(gradient - numerical)
                    allowed = reg['gradient_absolute_tolerance'] + reg['gradient_relative_tolerance'] * max(abs(gradient), abs(numerical))
                    gradient_ok = math.isfinite(gradient) and math.isfinite(numerical) and gap <= allowed
                    if hashes(fields) != original:
                        raise ValueError('Offline label builder changed candidate input state')
                    result['rows'].append(dict(identifier=identifier, split=receipt['split'],
                        counts_xyz=scene['counts'], cells_per_metre=scene['cells_per_metre'],
                        profile=scene['forcing_program']['profile'], motion=scene['forcing_program']['motion'],
                        start_fine_step=start, width=width, local_dt_s=phase['dt'],
                        zero_path_bits_exact=True, baseline_metrics=baseline, teacher_injected_metrics=measured,
                        teacher_injected_pass=all(v <= reg['maximum_error'] for v in measured['relative'].values()),
                        directional_gradient=gradient, finite_difference_gradient=numerical,
                        gradient_absolute_gap=gap, gradient_allowed_gap=allowed, gradient_pass=gradient_ok,
                        unrounded_zero_alpha_squared_losses={n: float(v.detach()) for n, v in components.items()},
                        complete_force_rows=len(phase['forcing']), future_fields_only_used_as_offline_targets=True))
                    if torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']:
                        raise ValueError('Native target qualification GPU memory limit')
                    del phase, fine_phase, zero, coarse, injected, objective, components, sample, fields, alpha
                write(args.output / 'results.json', result)
            print('Native temporal targets and gradients:', identifier, flush=True)
        if len(result['rows']) != reg['expected_groups'] or inventory(model) != before:
            raise ValueError('Full registered target population and fixed learned parameters required')
        result.update(status='completed_native_temporal_group_target_and_gradient_controls',
            all_teacher_injections_pass=all(r['teacher_injected_pass'] for r in result['rows']),
            all_gradient_controls_pass=all(r['gradient_pass'] for r in result['rows']),
            native_scene_count=len(dataset.receipts), all_widths_explicit=True,
            new_parameter_fits=0, learned_parameters_unchanged=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if model is not None:
            model.close()
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
