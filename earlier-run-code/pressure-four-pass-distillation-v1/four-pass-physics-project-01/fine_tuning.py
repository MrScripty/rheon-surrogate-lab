"""Differentiable pressure-equation feedback through bounded shared corrections.

The frozen teacher is auxiliary. Native/physical labels remain authoritative.
No teacher is needed for deployment, and no optimizer can update its parameters.
"""
from collections import OrderedDict
import copy
import math

import torch

SPEC = None
TEACHER = None
CACHE = OrderedDict()
CACHE_BYTES = 0
STATS = {'teacher_forwards': 0, 'teacher_cache_hits': 0, 'teacher_cache_peak_bytes': 0}


def select(spec):
    expected = {'schema', 'arm', 'initial_weights_sha256', 'initial_inventory', 'training_passes',
                'teacher_passes', 'teacher_weight', 'early_weight', 'relative_equation_target', 'teacher_cache_bytes'}
    if set(spec) != expected or spec['schema'] != 'rheon-fine-tuning-objective-v1':
        raise ValueError('Explicit closed fine-tuning objective required')
    if spec['training_passes'] not in (2, 4) or spec['teacher_passes'] != 8:
        raise ValueError('Matched two/four-pass objective and eight-pass teacher required')
    for name in ['teacher_weight', 'early_weight']:
        if not isinstance(spec[name], (int, float)) or not math.isfinite(spec[name]) or not 0 <= spec[name] <= 1:
            raise ValueError('Finite bounded feedback weight required')
    if spec['relative_equation_target'] != 1e-6 or not 0 < spec['teacher_cache_bytes'] <= 256*1024**2:
        raise ValueError('Unchanged equation target and bounded teacher cache required')
    global SPEC
    SPEC = copy.deepcopy(spec)


def initialize(network, path, helpers):
    from policy import read, sha
    if sha(read(path, 2*1024**2)) != SPEC['initial_weights_sha256']:
        raise ValueError('Fine-tuning initialization changed')
    helpers.load_weights(network, path, SPEC['initial_inventory'])
    global TEACHER, CACHE_BYTES
    TEACHER = copy.deepcopy(network).eval().requires_grad_(False)
    CACHE.clear()
    CACHE_BYTES = 0
    STATS.update(teacher_forwards=0, teacher_cache_hits=0, teacher_cache_peak_bytes=0)


def teacher_pressure(model, sample, normalization):
    global CACHE_BYTES
    if TEACHER is None:
        raise ValueError('Frozen initialization must be bound before training')
    # Cache values retain their sample to prevent reuse of tensor identities.
    key = tuple(id(sample[name]) for name in ['base', 'native_input_residual', 'cell_size_m', 'rho', 'dt'])
    if key in CACHE:
        CACHE.move_to_end(key)
        STATS['teacher_cache_hits'] += 1
        return CACHE[key][0]
    import physical_residual
    with torch.no_grad():
        result = physical_residual.predict_pressure(TEACHER, sample, normalization, model.owner,
                                                     {'activation': 'linear', 'steps': SPEC['teacher_passes']}).detach()
    STATS['teacher_forwards'] += SPEC['teacher_passes']
    size = result.numel()*result.element_size()
    while CACHE and CACHE_BYTES + size > SPEC['teacher_cache_bytes']:
        _, (_, _, old_size) = CACHE.popitem(last=False)
        CACHE_BYTES -= old_size
    if size <= SPEC['teacher_cache_bytes']:
        CACHE[key] = (result, sample, size)
        CACHE_BYTES += size
        STATS['teacher_cache_peak_bytes'] = max(STATS['teacher_cache_peak_bytes'], CACHE_BYTES)
    return result


def relative_square(value, reference):
    return value.flatten(1).square().sum(1) / reference.flatten(1).square().sum(1).clamp_min(1e-300)


def loss(model, network, sample, normalization, criterion=None):
    import physical_residual
    pressures = []
    prediction = physical_residual.predict_pressure(network, sample, normalization, model.owner,
                        {'activation': 'linear', 'steps': SPEC['training_passes']}, pressures)
    canonical = model.pressure_loss(prediction, sample, normalization) if criterion is None else criterion(prediction)
    if not pressures:
        return canonical
    initial = sample['native_input_residual']
    base = sample['base']
    goal = SPEC['relative_equation_target']
    # The logarithm keeps late, small residuals trainable without changing the
    # accepted physical threshold. Earlier stages have one fixed total weight.
    equations = [torch.log1p(relative_square(initial-model.full_operator(p-base, sample['cell_size_m'], sample['rho']), initial)/goal**2)
                 for p in pressures]
    result = canonical + equations[-1]
    if len(equations) > 1:
        result = result + SPEC['early_weight']*torch.stack(equations[:-1]).mean(0)
    if SPEC['teacher_weight']:
        teacher = teacher_pressure(model, sample, normalization)
        error = prediction-teacher
        target = teacher-base
        error = error-error.mean(dim=(2, 3, 4), keepdim=True)
        target = target-target.mean(dim=(2, 3, 4), keepdim=True)
        result = result + SPEC['teacher_weight']*torch.log1p(relative_square(error, target)/goal**2)
    return result
