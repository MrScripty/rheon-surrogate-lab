"""CUDA parameter-trial primitives for a finite managed full-population fit.

No training runner is provided here. The managed entrypoint must recompute
the complete balanced objective for each trial and keep an unchanged initial
checkpoint. A quadratic estimate proposes a step; measured loss accepts it.
"""
import math

import torch


@torch.no_grad()
def snapshot(network):
    values = network.state_dict()
    if any(p.device.type != 'cuda' or p.dtype != torch.float32 for p in values.values()):
        raise ValueError('Exact CUDA F32 network state required')
    return {n: p.clone() for n, p in values.items()}


@torch.no_grad()
def normalized_direction(network, gradient):
    parameters = dict(network.named_parameters())
    if set(parameters) != set(gradient):
        raise ValueError('Complete balanced population gradient required')
    for name, value in gradient.items():
        if (value.device.type != 'cuda' or value.dtype != torch.float64
                or value.shape != parameters[name].shape or not torch.isfinite(value).all()):
            raise ValueError('Finite typed full-population gradient differs')
    norm = float(torch.stack([g.square().sum() for g in gradient.values()]).sum().sqrt())
    if not math.isfinite(norm):
        raise ValueError('Nonfinite global gradient norm')
    return ({n: -g / norm for n, g in gradient.items()} if norm else None), norm


@torch.no_grad()
def parameter_trial(network, base, direction, step):
    if not math.isfinite(step) or step < 0 or set(base) != set(direction):
        raise ValueError('Finite nonnegative registered trial step required')
    current = network.state_dict()
    if set(current) != set(base):
        raise ValueError('Trial network inventory differs')
    for name, parameter in current.items():
        value, delta = base[name], direction[name]
        if (value.device.type != 'cuda' or value.dtype != torch.float32
                or delta.device != value.device or delta.dtype != torch.float64
                or value.shape != parameter.shape or delta.shape != value.shape):
            raise ValueError('Exact CUDA parameter/direction layout required')
        proposed = torch.where(delta == 0, value, (value.double() + step * delta).float())
        if not torch.isfinite(proposed).all():
            raise ValueError('Nonfinite parameter trial')
        parameter.copy_(proposed)


def quadratic_trial_scale(initial_loss, probe_loss, probe_step, gradient_norm):
    if (not all(math.isfinite(v) for v in (initial_loss, probe_loss, probe_step, gradient_norm))
            or initial_loss < 0 or probe_loss < 0 or probe_step <= 0 or gradient_norm <= 0):
        raise ValueError('Finite measured full-population losses and descent direction required')
    coefficient = (probe_loss - initial_loss + probe_step * gradient_norm) / probe_step ** 2
    candidate = min(probe_step, gradient_norm / (2 * coefficient)) if coefficient > 0 else probe_step
    return dict(probe_step=probe_step, fitted_quadratic_coefficient=coefficient,
        proposed_step=candidate, predicted_decrease=gradient_norm * candidate - coefficient * candidate ** 2,
        requires_actual_complete_population_evaluation=True)
