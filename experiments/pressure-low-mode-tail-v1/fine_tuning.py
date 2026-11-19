"""Final-pass low spatial-mode objective over the existing physical feedback."""
import copy
import math

import torch
import original_fine_tuning as original

SPEC = None


def __getattr__(name):
    return getattr(original, name)


def select(spec):
    feedback = spec['low_mode_feedback']
    if set(feedback) != {'weight', 'max_mode', 'signal_floor_fraction'}:
        raise ValueError('Closed final low-mode feedback required')
    if (type(feedback['weight']) not in (int, float) or not math.isfinite(feedback['weight'])
            or not 0 <= feedback['weight'] <= 1 or feedback['max_mode'] != 3
            or feedback['signal_floor_fraction'] != .1):
        raise ValueError('Bounded fixed low-mode normalization required')
    original.select({k:v for k,v in spec.items() if k != 'low_mode_feedback'})
    global SPEC
    SPEC = copy.deepcopy(spec)


def final_feedback(prediction, sample):
    import band_feedback
    specification = SPEC['low_mode_feedback']
    relative_pressure_square = band_feedback.loss(prediction, sample, specification)
    return specification['weight']*torch.log1p(relative_pressure_square/SPEC['relative_equation_target']**2)


def loss(model, network, sample, normalization, criterion=None):
    return original.loss(model, network, sample, normalization, criterion,
                         final_feedback=final_feedback if SPEC['low_mode_feedback']['weight'] else None)
