"""Explicit training-only metric weights; physical acceptance metrics stay unchanged."""
import math

from internal_residual_dataset_v1 import METRICS


def validate(weights):
    if set(weights) != set(METRICS) or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in weights.values()
    ):
        raise ValueError('Four finite positive training metric weights required')
    return weights


def weighted_sum(values, weights):
    return sum(values[name] * weights[name] for name in METRICS)


def fixed_objective(view, normalization, weights):
    fixed = view.fixed_objective(normalization)
    return dict(fixed, objective=weighted_sum(fixed['per_metric_scene_sum'], weights) / 148)
