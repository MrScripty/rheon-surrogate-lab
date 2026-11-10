"""What the benchmark measures, and the maths for each number.

Every metric compares a subject's own rollout with the reference trajectory of
the same case. `DEFINITIONS` is what a result document and Tuldok show; the
functions below are the only place the numbers are computed.

The error definitions are those of the lab's earlier matched benchmarks
(relative L2 in float64, pressure mean-free, the three velocity components
together), so a one-pass window here equals the block maximum reported there.
"""
import torch

from ..fields import divergence, remove_mean

DEFINITIONS = [
    {'id': 'velocity_error_worst', 'label': 'Worst velocity error', 'units': 'ratio',
     'description': 'Largest relative L2 velocity error over the rollout. All three staggered components '
                    'together: norm of (prediction - reference) divided by the norm of the reference.'},
    {'id': 'velocity_error_final', 'label': 'Final velocity error', 'units': 'ratio',
     'description': 'Relative L2 velocity error at the last output of the rollout.'},
    {'id': 'pressure_error_worst', 'label': 'Worst pressure error', 'units': 'ratio',
     'description': 'Largest relative L2 pressure error over the rollout, after removing the mean from both '
                    'fields (pressure in a sealed box is only defined up to a constant).'},
    {'id': 'pressure_error_final', 'label': 'Final pressure error', 'units': 'ratio',
     'description': 'Relative L2 mean-free pressure error at the last output of the rollout.'},
    {'id': 'tracer_error_worst', 'label': 'Worst tracer error', 'units': 'ratio',
     'description': 'Largest relative L2 tracer error over the rollout.'},
    {'id': 'tracer_error_final', 'label': 'Final tracer error', 'units': 'ratio',
     'description': 'Relative L2 tracer error at the last output of the rollout.'},
    {'id': 'equation_error_worst', 'label': 'Worst pressure equation residual', 'units': 'ratio',
     'description': "Largest relative residual of the reference's pressure equation when the predicted "
                    'pressure is put into it: norm of (right-hand side - operator applied to the pressure) '
                    'divided by the norm of the right-hand side. Only for cases that record the right-hand side.'},
    {'id': 'divergence_worst', 'label': 'Worst divergence', 'units': '1/s',
     'description': 'Largest absolute velocity divergence of any cell at any output.'},
    {'id': 'energy_ratio_final', 'label': 'Final kinetic energy ratio', 'units': 'ratio',
     'description': 'Sum of squared velocity at the last output divided by the same sum for the reference. '
                    'Below one means the rollout lost energy, above one that it gained energy.'},
    {'id': 'valid_outputs', 'label': 'Outputs within the error limit', 'units': 'count',
     'description': "Number of outputs from the start of the rollout before the velocity error first "
                    "exceeds the suite's limit or a field stops being finite."},
    {'id': 'time_per_output_ms', 'label': 'Time per output', 'units': 'ms',
     'description': "Median wall time of the case's first pass, repeated after warm-up, divided by the outputs "
                    'a pass produces. A pass starts with the state and forcing in host memory and ends with '
                    'every output back in host memory. A subject that runs in its own process reports its own '
                    'computation time instead, without the cost of moving the state in and out.'},
    {'id': 'simulated_seconds_per_second', 'label': 'Simulated seconds per second', 'units': 'ratio',
     'description': 'Simulated time advanced per second of wall time: the output interval divided by '
                    'the time per output.'},
]
IDS = tuple(definition['id'] for definition in DEFINITIONS)
TIMING_IDS = ('time_per_output_ms', 'simulated_seconds_per_second')
ERROR_IDS = tuple(i for i in IDS if i not in TIMING_IDS and i != 'valid_outputs')


def _square_sum(*tensors):
    return sum((tensor.double() ** 2).sum() for tensor in tensors).item()


def _ratio(error, size):
    """sqrt(error / size); with a zero reference it is 0 for an exact match and undefined otherwise."""
    if size:
        return (error / size) ** 0.5
    return 0.0 if error == 0 else None


def relative_error(predicted, reference):
    """Relative L2 error of one or several fields taken together, or None if it is undefined."""
    return _ratio(_square_sum(*(p.double() - r.double() for p, r in zip(predicted, reference))),
                  _square_sum(*reference))


def pressure_operator(pressure, spacing, density):
    """The sealed-box pressure operator: net face flux of the pressure gradient, zero through walls."""
    result = torch.zeros_like(pressure)
    for axis in (-1, -2, -3):
        flux = torch.diff(pressure, dim=axis) * (spacing / density)
        count = pressure.shape[axis]
        result.narrow(axis, 0, count - 1).sub_(flux)
        result.narrow(axis, 1, count - 1).add_(flux)
    return result


def compare(predicted, reference, spacing, density, right_hand_side=None):
    """Scores of one predicted state against its reference state (single case, no batch dimension)."""
    velocity = (predicted.u, predicted.v, predicted.w)
    reference_velocity = (reference.u, reference.v, reference.w)
    scores = {
        'velocity_error': relative_error(velocity, reference_velocity),
        'pressure_error': relative_error((remove_mean(predicted.pressure.double()),),
                                         (remove_mean(reference.pressure.double()),)),
        'tracer_error': relative_error((predicted.tracer,), (reference.tracer,)),
        'equation_error': None,
        'divergence': (divergence(*(c.double() for c in velocity)).abs().max() / spacing).item(),
        'energy_ratio': None,
    }
    reference_energy = _square_sum(*reference_velocity)
    if reference_energy:
        scores['energy_ratio'] = _square_sum(*velocity) / reference_energy
    if right_hand_side is not None:
        right_hand_side = right_hand_side.double()
        residual = right_hand_side - pressure_operator(predicted.pressure.double(), spacing, density)
        scores['equation_error'] = _ratio(_square_sum(residual), _square_sum(right_hand_side))
    return scores


def is_finite(state):
    return all(torch.isfinite(tensor).all().item()
               for tensor in (state.u, state.v, state.w, state.tracer, state.pressure))


def summarise(per_output, velocity_error_limit):
    """Reduce the per-output scores of one rollout to the error metrics above.

    `per_output` holds the `compare` result of every output that stayed finite, in
    order. A metric that was undefined at every output is None.
    """
    valid = 0
    for scores in per_output:
        if scores['velocity_error'] is not None and scores['velocity_error'] > velocity_error_limit:
            break
        valid += 1

    def worst(name):
        defined = [scores[name] for scores in per_output if scores[name] is not None]
        return max(defined) if defined else None

    def final(name):
        return per_output[-1][name] if per_output else None

    summary = {'valid_outputs': valid, 'divergence_worst': worst('divergence'),
               'equation_error_worst': worst('equation_error'), 'energy_ratio_final': final('energy_ratio')}
    for name in ('velocity', 'pressure', 'tracer'):
        summary[f'{name}_error_worst'] = worst(f'{name}_error')
        summary[f'{name}_error_final'] = final(f'{name}_error')
    return summary
