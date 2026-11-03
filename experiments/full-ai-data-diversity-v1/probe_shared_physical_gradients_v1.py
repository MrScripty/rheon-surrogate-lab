"""Check actual full-state physical loss gradients before any managed fit.

This is one real cold pulse window, including quiet outputs before the first
sampled force. It performs backward once, with no optimizer or weight changes.
"""
import argparse
import math
import os
from pathlib import Path
import time

import torch

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index, LongTemporalDataset, FIELDS
from shared_temporal_generator_v2 import SharedTemporalGenerator
from internal_residual_loss_v1 import metrics
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh CUDA-only physical gradient probe required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen physical-gradient source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), inference_device='cuda',
        display_label='GPU (CUDA)', cpu_model_forward=False, new_parameter_fits=0,
        optimizer_updates=0, backward_calls=0, reserved52_references_opened=0,
        full_generator_target_pass=False, physical_head_losses=[])
    model = None
    try:
        dataset = LongTemporalDataset(build_index([ROOT / p for p in reg['audits']]))
        example = next(e for e in dataset.index['examples'] if all(e[n] == reg[n]
            for n in ('identifier', 'start_fine_step', 'output_steps_per_s', 'K')))
        scene = dataset.receipts[example['identifier']][1]['scene']
        inputs = dataset.inputs(example)
        fields = {n: torch.from_numpy(inputs['current'][n]).to('cuda') for n in FIELDS}
        shape = tuple(reversed(scene['counts']))
        shapes = dict(pressure=shape, **dict(zip(FIELDS[1:], dimensions(tuple(scene['counts'])))))
        model = SharedTemporalGenerator(reg['transport_pair'], reg['shared_pair'], reg['build'],
            reg['control_build'], hidden=reg['hidden_width'], force_sum_build=reg['force_sum_build'])
        model.network.requires_grad_(True)
        before = inventory(model)
        rate, k, start = reg['output_steps_per_s'], reg['K'], reg['start_fine_step']
        packet, events, mask = model.packets(scene, start, rate, k)
        outputs = model.predict_packet(fields, scene, rate, k, packet, events, mask)
        objective = fields['pressure'].new_zeros(())
        normalization = read(ROOT / reg['normalization'])
        for lead, (output, step) in enumerate(zip(outputs, example['target_fine_steps']), 1):
            truth = dataset._load(example['identifier'], step)
            sample = dict(h=1 / scene['cells_per_metre'], rho=1000.,
                target_rhs=torch.from_numpy(truth['rhs']).to('cuda').reshape(shape))
            for n in FIELDS:
                sample['target_' + n] = torch.from_numpy(truth[n]).to('cuda').reshape(shapes[n])
            values = metrics({n: v.reshape(shapes[n]) for n, v in output.items()}, sample, normalization)
            objective = objective + sum(values.values()) / k
            result['physical_head_losses'].append(dict(lead=lead, elapsed_s=lead / rate,
                target_rhs_l2=float(sample['target_rhs'].norm()),
                loss_components={n: float(v.detach()) if math.isfinite(float(v.detach())) else None for n, v in values.items()},
                predicted_pressure_l2=float(output['pressure'].detach().norm()),
                target_pressure_l2=float(sample['target_pressure'].norm())))
        result['objective_finite'] = bool(torch.isfinite(objective))
        result['objective'] = float(objective.detach()) if result['objective_finite'] else None
        objective.backward()
        result['backward_calls'] = 1
        bad = [n for n, p in model.network.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        result['nonfinite_or_missing_gradient_parameters'] = bad
        result['model_parameters_unchanged'] = inventory(model) == before
        if not result['model_parameters_unchanged']:
            raise ValueError('Physical probe changed model parameters')
        if (time.monotonic() - started > reg['maximum_wall_seconds']
                or torch.cuda.max_memory_allocated() > reg['maximum_gpu_allocated_bytes']):
            raise ValueError('Registered physical-gradient probe resource budget exceeded')
        if bad or not result['objective_finite']:
            result['status'] = 'failed_nonfinite_shared_physical_gradients'
            raise AssertionError('Nonfinite physical training gradients on real cold pulse window')
        result['status'] = 'passed_real_shared_physical_gradient_probe'
    except BaseException as error:
        if result['status'] == 'running':
            result['status'] = 'failed'
        result['error'] = repr(error)
        raise
    finally:
        if model is not None:
            model.close()
        result.update(elapsed_seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
