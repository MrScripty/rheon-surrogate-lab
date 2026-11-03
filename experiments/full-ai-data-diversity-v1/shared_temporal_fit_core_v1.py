"""Shared complete-state physical objective for captured managed CUDA fits.

This module exposes no local optimizer runner. Qualification uses backward
without updates; the managed fitting entrypoint must validate typed custody.
"""
import math
import time

import torch

from captured_long_training_bank_v1 import CapturedLongTrainingBank
from shared_temporal_generator_v3 import SharedTemporalGenerator
from internal_residual_loss_v1 import metrics
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def make_model(configuration, width):
    model = SharedTemporalGenerator(configuration['transport_pair'], configuration['shared_pair'],
        configuration['build'], configuration['control_build'], hidden=width,
        force_sum_build=configuration['force_sum_build'])
    model.network.requires_grad_(True)
    return model


def physical_block(model, current, scene, example, target_arrays, normalization):
    rate, k, start = (example[n] for n in ('output_steps_per_s', 'K', 'start_fine_step'))
    shape = tuple(reversed(scene['counts']))
    shapes = dict(pressure=shape, **dict(zip(('vx', 'vy', 'vz', 'tracer'), dimensions(tuple(scene['counts'])))))
    # Only current state and deployment-available prescribed controls enter the model.
    packet, events, mask = model.packets(scene, start, rate, k)
    outputs = model.predict_packet(current, scene, rate, k, packet, events, mask)
    total = current['pressure'].new_zeros(())
    values = {n: 0. for n in ('equation', 'pressure', 'velocity', 'tracer')}
    for output, step in zip(outputs, example['target_fine_steps']):
        truth = target_arrays[step]
        sample = dict(h=1 / scene['cells_per_metre'], rho=1000., target_rhs=truth['rhs'].reshape(shape))
        sample.update({'target_' + n: truth[n].reshape(size) for n, size in shapes.items()})
        components = metrics({n: output[n].reshape(size) for n, size in shapes.items()}, sample, normalization)
        total = total + sum(components.values()) / k
        for name, value in components.items():
            values[name] += float(value.detach()) / k
    if len(outputs) != k:
        raise ValueError('Every requested physical output is required')
    return total, values


def full_objective(model, bank, normalization, backward, bounded, progress=None):
    total = 0.
    components = {n: 0. for n in ('equation', 'pressure', 'velocity', 'tracer')}
    checked_blocks = checked_outputs = 0
    rows = []
    for index, window in enumerate(bank.windows):
        item = bank.cuda_window(window)
        weight = window['scene_weight'] * window['window_weight'] / 12
        row = dict(identifier=window['identifier'], start_fine_step=window['start_fine_step'], objective=0., blocks=0)
        for example in window['configurations']:
            model.network.zero_grad(set_to_none=True)
            with torch.enable_grad() if backward else torch.no_grad():
                loss, measured = physical_block(model, item['current'], item['scene'], example, item['targets'], normalization)
                scalar = float(loss.detach())
                if not math.isfinite(scalar) or any(not math.isfinite(v) for v in measured.values()):
                    raise ValueError('Nonfinite physical objective: ' + repr((window['identifier'], example)))
                if backward:
                    (loss * weight).backward()
                    bad = [n for n, p in model.network.named_parameters()
                           if (p.grad is None and not (example['K'] == 1 and n.startswith('interior_')))
                           or (p.grad is not None and not torch.isfinite(p.grad).all())]
                    if bad:
                        raise ValueError('Nonfinite/missing executed gradient: ' + repr((window['identifier'], example, bad)))
            total += scalar * weight
            row['objective'] += scalar / 12
            row['blocks'] += 1
            for name in components:
                components[name] += measured[name] * weight
            checked_blocks += 1
            checked_outputs += example['K']
            del loss, measured
            bounded()
        rows.append(row)
        del item
        if progress:
            progress(index + 1, len(bank.windows), total)
    if checked_blocks != 12 * len(bank.windows):
        raise ValueError('Incomplete all12-window objective')
    return dict(objective=total, components=components, windows=len(bank.windows),
        blocks=checked_blocks, physical_outputs=checked_outputs, rows=rows,
        independent_training_masters=24, every_selected_window_used=True,
        equal_scene_and_temporal_configuration_weights=True, within_block_head_weight='1/K')
