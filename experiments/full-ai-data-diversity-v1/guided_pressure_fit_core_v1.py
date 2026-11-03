"""Captured full-population CUDA loss and zero-update gradient qualification."""
import math

import torch

from guided_pressure_residual_v1 import GuidedPressureResidual
from internal_residual_loss_v1 import metrics
from recurrent_sampler_native import dimensions


def make_model(configuration, width):
    model = GuidedPressureResidual(configuration['transport_pair'], configuration['shared_pair'],
        configuration['build'], configuration['control_build'], hidden=width,
        force_sum_build=configuration['force_sum_build'])
    model.network.requires_grad_(True)
    return model


def physical_block(model, current, scene, example, targets, normalization, certify_zero_heads=False, diagnostics=None):
    rate, k, start = (example[n] for n in ('output_steps_per_s', 'K', 'start_fine_step'))
    shape = tuple(reversed(scene['counts']))
    shapes = dict(pressure=shape, **dict(zip(('vx', 'vy', 'vz', 'tracer'), dimensions(tuple(scene['counts'])))))
    packet, events, mask = model.packets(scene, start, rate, k)
    outputs = model.predict_packet(current, scene, rate, k, packet, events, mask)
    if len(outputs) != k:
        raise ValueError('Every requested physical output is required')
    if certify_zero_heads:
        with torch.no_grad():
            if any(not torch.equal(output[n].detach().view(torch.uint8), guide[n].view(torch.uint8))
                   for output, guide in zip(outputs, model.last_guide_outputs) for n in shapes):
                raise ValueError('New zero residual heads changed frozen guide output bytes')
    total = current['pressure'].new_zeros(())
    values = {n: 0. for n in ('equation', 'pressure', 'velocity', 'tracer')}
    for lead, (output, step) in enumerate(zip(outputs, example['target_fine_steps']), 1):
        truth = targets[step]
        sample = dict(h=1 / scene['cells_per_metre'], rho=1000., target_rhs=truth['rhs'].reshape(shape))
        sample.update({'target_' + n: truth[n].reshape(size) for n, size in shapes.items()})
        components = metrics({n: output[n].reshape(size) for n, size in shapes.items()}, sample, normalization)
        total = total + sum(components.values()) / k
        for name, value in components.items():
            value_scalar = float(value.detach())
            values[name] += value_scalar / k
            if diagnostics is not None:
                relative = math.sqrt(value_scalar)
                if relative >= diagnostics.setdefault('worst_relative', {}).get(name, -1.):
                    diagnostics['worst_relative'][name] = relative
                    diagnostics.setdefault('worst_lead', {})[name] = lead
    return total, values


def full_objective(model, bank, normalization, backward, bounded, progress=None, certify_zero_heads=False, gradient_receiver=None):
    total = 0.
    components = {n: 0. for n in ('equation', 'pressure', 'velocity', 'tracer')}
    sums = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in model.network.named_parameters()} if backward else {}
    checked_blocks = checked_outputs = 0
    worst_relative = {n: 0. for n in components}
    worst_locations = {}
    rows = []
    for index, window in enumerate(bank.windows):
        item = bank.cuda_window(window)
        weight = window['scene_weight'] * window['window_weight'] / 12
        row = dict(identifier=window['identifier'], start_fine_step=window['start_fine_step'], objective=0., blocks=0)
        for example in window['configurations']:
            model.network.zero_grad(set_to_none=True)
            with torch.enable_grad() if backward else torch.no_grad():
                diagnostics = {}
                loss, measured = physical_block(model, item['current'], item['scene'], example,
                    item['targets'], normalization, certify_zero_heads=certify_zero_heads, diagnostics=diagnostics)
                scalar = float(loss.detach())
                if not math.isfinite(scalar) or any(not math.isfinite(v) for v in measured.values()):
                    raise ValueError('Nonfinite guided physical objective')
                if backward:
                    (loss * weight).backward()
                    for name, parameter in model.network.named_parameters():
                        if parameter.grad is None:
                            if example['K'] == 1 and name.startswith('interior_'):
                                continue
                            raise ValueError('Missing executed guided gradient: ' + name)
                        if not torch.isfinite(parameter.grad).all():
                            raise ValueError('Nonfinite guided gradient: ' + name)
                        sums[name].add_(parameter.grad.double())
            total += scalar * weight
            row['objective'] += scalar / 12
            row['blocks'] += 1
            for name in components:
                components[name] += measured[name] * weight
            for name, relative in diagnostics['worst_relative'].items():
                if relative >= worst_relative[name]:
                    worst_relative[name] = relative
                    worst_locations[name] = dict(identifier=window['identifier'],
                        start_fine_step=window['start_fine_step'], output_steps_per_s=example['output_steps_per_s'],
                        K=example['K'], lead=diagnostics['worst_lead'][name])
            checked_blocks += 1
            checked_outputs += example['K']
            del loss, measured
            bounded()
        rows.append(row)
        del item
        if progress:
            progress(index + 1, len(bank.windows), total)
    if checked_blocks != 12 * len(bank.windows):
        raise ValueError('Incomplete guided all12-window objective')
    if gradient_receiver is not None:
        gradient_receiver.update(sums)
    norms = {n: dict(l2=float(g.square().sum().sqrt()), maximum_absolute=float(g.abs().max()))
             for n, g in sums.items()}
    return dict(objective=total, components=components, worst_relative=worst_relative,
        maximum_relative_error=max(worst_relative.values()), worst_locations=worst_locations, windows=len(bank.windows),
        blocks=checked_blocks, physical_outputs=checked_outputs, rows=rows,
        independent_training_masters=24, every_selected_window_used=True,
        equal_scene_and_temporal_configuration_weights=True, within_block_head_weight='1/K',
        zero_head_byte_parity_blocks=checked_blocks if certify_zero_heads else 0,
        zero_head_byte_parity_outputs=checked_outputs if certify_zero_heads else 0,
        complete_population_gradient_norms=norms,
        complete_population_gradient_l2=math.sqrt(math.fsum(v['l2'] ** 2 for v in norms.values())),
        gradient_vectors_exported=False, optimizer_updates=0)
