"""Matched attached/detached auxiliary correction over the unchanged fitting loss."""
import copy

import torch
import original_fine_tuning as original
import original_fit_bank

SPEC = None
ACTIVE_DEPTH = 0


def __getattr__(name):
    return getattr(original, name)


def select(spec):
    original_fit_bank.validate(spec['fitting_bank'])
    feedback = spec['rollout_feedback']
    if (set(feedback) != {'mode', 'weight', 'depth_seed'}
            or feedback['mode'] not in ('U', 'A', 'D')
            or feedback['weight'] != .05 or type(feedback['depth_seed']) is not int
            or spec['training_passes'] != 4 or spec['teacher_weight'] != 0):
        raise ValueError('Closed matched four-pass auxiliary correction required')
    original.select({k: v for k, v in spec.items() if k not in ('fitting_bank', 'rollout_feedback')})
    global SPEC
    SPEC = copy.deepcopy(spec)


def auxiliary(model, network, sample, normalization, pressures, criterion, depth, mode):
    if depth not in (0, 1, 2, 3) or len(pressures) != 4:
        raise ValueError('Four current-weight states and a bounded depth required')
    carry = sample['base'] if depth == 0 else pressures[depth - 1]
    if mode == 'D':
        carry = carry.detach()
    elif mode != 'A':
        raise ValueError('Explicit auxiliary gradient boundary required')
    remaining = sample['native_input_residual'] - model.full_operator(
        carry - sample['base'], sample['cell_size_m'], sample['rho'])
    import physical_residual
    prediction = physical_residual.predict_pressure(network,
        {**sample, 'base': carry, 'native_input_residual': remaining},
        normalization, model.owner, {'activation': 'linear', 'steps': 1})
    canonical = model.pressure_loss(prediction, sample, normalization) if criterion is None else criterion(prediction)
    residual = sample['native_input_residual'] - model.full_operator(
        prediction - sample['base'], sample['cell_size_m'], sample['rho'])
    equation = torch.log1p(original.relative_square(residual, sample['native_input_residual'])
                          / SPEC['relative_equation_target']**2)
    return canonical + equation, prediction, carry


def loss(model, network, sample, normalization, criterion=None):
    pressures = []
    value = original.loss(model, network, sample, normalization, criterion, intermediates=pressures)
    mode = SPEC['rollout_feedback']['mode']
    if mode == 'U' or not pressures:
        return value
    extra, _, _ = auxiliary(model, network, sample, normalization, pressures, criterion, ACTIVE_DEPTH, mode)
    return value + SPEC['rollout_feedback']['weight'] * extra
