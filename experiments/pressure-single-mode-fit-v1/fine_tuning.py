"""Explicit single manufactured fitting case over the existing loss owner."""
import copy
import math
import original_fine_tuning as original

SPEC = None


def __getattr__(name):
    return getattr(original, name)


def select(spec):
    case = spec['fitting_case']
    if set(case) != {'counts_xyz', 'cells_per_metre', 'mode_zyx', 'amplitude_reference'}:
        raise ValueError('Closed manufactured fitting case required')
    counts = case['counts_xyz']
    if (len(counts) != 3 or any(type(n) is not int or not 5 <= n <= 128 for n in counts)
            or math.prod(counts) > 65536 or type(case['cells_per_metre']) is not int
            or not 1 <= case['cells_per_metre'] <= 128):
        raise ValueError('Supported counts with cubic voxel spacing required')
    mode = case['mode_zyx']
    if len(mode) != 3 or not any(mode) or any(type(k) is not int or not 0 <= k < n for k,n in zip(mode,reversed(counts))):
        raise ValueError('Nonconstant bounded Neumann mode required')
    if case['amplitude_reference'] != 'original-train-correction-rms' or spec['teacher_weight'] != 0:
        raise ValueError('Train-only calibration and no teacher for capacity test')
    original.select({k:v for k,v in spec.items() if k != 'fitting_case'})
    global SPEC
    SPEC = copy.deepcopy(spec)


def loss(model, network, sample, normalization, criterion=None):
    return original.loss(model, network, sample, normalization, criterion)
