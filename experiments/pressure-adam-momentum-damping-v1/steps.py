"""Damp only saved Adam first moments before an ordinary disposable update."""
from pathlib import Path
import sys

import torch

PREVIOUS = Path(__file__).resolve().parents[1] / 'pressure-adam-step-screen-v1'
# The historical endpoint owner is immutable; import under its distinct filename.
import importlib.util
spec = importlib.util.spec_from_file_location('captured_adam_steps', PREVIOUS / 'steps.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

FACTORS = (0., .25, .5, .75, 1.)


def damped_adam(before, gradient, group, step_before, first, second, factor):
    if factor not in FACTORS:
        raise ValueError('Predeclared first-moment retention factor required')
    if not isinstance(step_before, int) or step_before < 0:
        raise ValueError('Captured nonnegative integer Adam step required')
    values = (before, gradient, first, second)
    if any(v.dtype != torch.float32 or v.shape != before.shape or
           v.device.type != 'cpu' or not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Finite matching CPU float32 vectors required')
    if bool((second < 0).any()):
        raise ValueError('Nonnegative Adam second moment required')
    return owner.adam(before, gradient, group, step_before, (first * factor, second))
