"""Zero-update diagnostics for the pressure heads with a fixed shared trunk.

Measure the full-population pressure/equation Gauss-Newton matrix on CUDA.
No solve, parameter proposal, optimizer or parameter write occurs here. Native
equation actions and future fields belong solely to the supervised loss.
"""
import math

import torch

from coupled_macrostep_v1 import _RestoredPressure
from internal_residual_loss_v1 import equation_action


def full_pressure_geometry(model, bank, normalization, bounded, progress=None, diagnostic_receiver=None):
    width = model.network.hidden
    names = ('endpoint_pressure.weight', 'interior_pressure.weight')
    parameters = dict(model.network.named_parameters())
    selected = [parameters[n] for n in names]
    if any(torch.count_nonzero(p) != 0 for p in selected):
        raise ValueError('This initial-point diagnostic requires certified zero pressure heads')
    matrix = torch.zeros((2 * width, 2 * width), device='cuda', dtype=torch.float64)
    gradient = torch.zeros(2 * width, device='cuda', dtype=torch.float64)
    measured_gradient = torch.zeros_like(gradient)
    blocks = outputs_checked = 0
    pressure_equation_objective = 0.
    maximum_gradient_relative_disagreement = 0.
    rows = []
    for index, window in enumerate(bank.windows):
        item = bank.cuda_window(window)
        scene, current = item['scene'], item['current']
        weight = window['scene_weight'] * window['window_weight'] / 12
        row_loss = 0.
        for example in window['configurations']:
            rate, k, start = (example[n] for n in ('output_steps_per_s', 'K', 'start_fine_step'))
            features = []
            handles = [getattr(model.network, kind + '_pressure').register_forward_pre_hook(
                lambda layer, inputs, kind=kind: features.append((kind, inputs[0].detach())))
                for kind in ('endpoint', 'interior')]
            try:
                packet, events, mask = model.packets(scene, start, rate, k)
                with torch.enable_grad():
                    outputs = model.predict_packet(current, scene, rate, k, packet, events, mask)
            finally:
                for handle in handles:
                    handle.remove()
            if [n for n, value in features] != ['interior'] * (k - 1) + ['endpoint']:
                raise ValueError('Pressure feature execution order changed')
            phase = model.guide._guide_phase
            shape = phase['shape']
            loss = torch.zeros((), device='cuda', dtype=torch.float64)
            analytic = torch.zeros_like(gradient)
            for lead, (output, guide, (kind, hidden), step) in enumerate(zip(
                    outputs, model.last_guide_outputs, features, example['target_fine_steps']), 1):
                if any(not torch.equal(output[n].detach().view(torch.uint8), guide[n].view(torch.uint8))
                       for n in output):
                    raise ValueError('Initial model differs from guide bytes')
                truth = item['targets'][step]
                pressure = output['pressure'].reshape(shape)
                target = truth['pressure'].double().reshape(shape)
                target = target - target.mean()
                denominator_p = target.norm().clamp_min(
                    normalization['relative_denominator_rms_floor']['pressure'] * math.sqrt(target.numel()))
                rhs = truth['rhs'].reshape(shape)
                denominator_e = rhs.norm().clamp_min(max(normalization.get('equation_rhs_floor', 1e-30),
                    normalization.get('optimization_equation_rms_floor', 0.) * math.sqrt(rhs.numel())))
                residual_p = ((pressure - pressure.mean() - target) / denominator_p).flatten()
                residual_e = ((equation_action(pressure, phase['h'], phase['rho']) - rhs) / denominator_e).flatten()
                loss = loss + (residual_p.square().sum() + residual_e.square().sum()) / k
                with torch.no_grad():
                    scale = model.pressure_scale(guide, phase)
                    columns_p, columns_e = [], []
                    for channel in range(width):
                        column = _RestoredPressure.apply((hidden[0, channel].double() * scale).reshape(
                            1, 1, *shape), model.model, phase['counts'])[0, 0]
                        columns_p.append(((column - column.mean()) / denominator_p).flatten())
                        columns_e.append((equation_action(column, phase['h'], phase['rho']) / denominator_e).flatten())
                    jp, je = torch.stack(columns_p, 1), torch.stack(columns_e, 1)
                    offset = 0 if kind == 'endpoint' else width
                    region = slice(offset, offset + width)
                    matrix[region, region].add_((jp.T @ jp + je.T @ je) * (2 * weight / k))
                    analytic[region].add_((jp.T @ residual_p.detach() + je.T @ residual_e.detach()) * (2 / k))
            observed = torch.autograd.grad(loss, selected, allow_unused=True)
            observed = torch.cat([torch.zeros_like(p).flatten().double() if g is None else g.flatten().double()
                                  for p, g in zip(selected, observed)])
            if not torch.isfinite(analytic).all() or not torch.isfinite(observed).all():
                raise ValueError('Nonfinite pressure-head geometry/gradient')
            relative = float((analytic - observed).norm() / observed.norm().clamp_min(1e-14))
            if diagnostic_receiver is not None:
                diagnostic_receiver.append(dict(identifier=window['identifier'], counts=scene['counts'],
                    start=start, rate=rate, K=k, analytic_l2=float(analytic.norm()), observed_l2=float(observed.norm()),
                    relative_disagreement=relative, maximum_absolute_disagreement=float((analytic-observed).abs().max())))
            if relative > 1e-4 and float((analytic - observed).abs().max()) > 1e-14:
                raise ValueError('Learned pressure columns disagree with actual autograd')
            maximum_gradient_relative_disagreement = max(maximum_gradient_relative_disagreement, relative)
            gradient.add_(analytic * weight)
            measured_gradient.add_(observed * weight)
            value = float(loss.detach())
            pressure_equation_objective += weight * value
            row_loss += value / 12
            blocks += 1
            outputs_checked += k
            del loss, outputs, features, observed, jp, je, columns_p, columns_e, residual_p, residual_e
            bounded()
        rows.append(dict(identifier=window['identifier'], start_fine_step=window['start_fine_step'],
            pressure_equation_objective=row_loss))
        del item
        if progress:
            progress(index + 1, len(bank.windows), pressure_equation_objective)
    if blocks != 12 * len(bank.windows) or not torch.isfinite(matrix).all():
        raise ValueError('Incomplete/nonfinite full-population pressure geometry')
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) * .5)
    scale = matrix.diag().sqrt()
    normalized = matrix / (scale[:, None] * scale[None, :]).clamp_min(1e-300)
    normalized_eigenvalues = torch.linalg.eigvalsh((normalized + normalized.T) * .5)
    return dict(windows=len(bank.windows), blocks=blocks, physical_outputs=outputs_checked,
        zero_head_byte_parity_outputs=outputs_checked, independent_training_masters=24,
        pressure_parameters=2 * width, pressure_equation_objective=pressure_equation_objective,
        Hessian_eigenvalues=eigenvalues.tolist(), correlation_eigenvalues=normalized_eigenvalues.tolist(),
        diagonal_curvatures=matrix.diag().tolist(),
        complete_population_pressure_gradient_l2=float(gradient.norm()),
        actual_autograd_pressure_gradient_l2=float(measured_gradient.norm()),
        maximum_per_block_gradient_relative_disagreement=maximum_gradient_relative_disagreement,
        full_gradient_absolute_disagreement=float((gradient - measured_gradient).abs().max()),
        rows=rows, optimizer_updates=0, parameter_fits=0,
        gradient_vectors_exported=False, parameter_solve_executed=False,
        interpretation='Fixed-body initial pressure-head curvature only; does not measure achievable trained accuracy.')
