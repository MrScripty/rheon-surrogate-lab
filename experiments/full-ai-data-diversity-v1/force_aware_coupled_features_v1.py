"""Causal force-order augmentation; no labels, equation stencil or inverse.

All point projections use restored learned weights. Known forcing is already
integrated over each fine interval; never multiply it by another timestep.
"""
import torch
from nonlinear_coupled_features_v1 import columns as previous_columns, shifted
from coupled_macrostep_features_v1 import OFFSETS
from coupled_macrostep_v1 import wall
from force_aware_coupled_context_v1 import forcing_contrasts
from recurrent_sampler_native import dimensions


WIDTHS = {'vx': 500, 'vy': 500, 'vz': 500, 'tracer': 778}
PREFIX = {'vx': 448, 'vy': 448, 'vz': 448, 'tracer': 610}


def difference(value, axis):
    offset = [0, 0, 0]; offset[2-axis] = 1
    plus = shifted(value, tuple(offset)); offset[2-axis] = -1
    return plus-shifted(value, tuple(offset))


@torch.no_grad()
def columns(model, context, name, mode='force_order'):
    if mode != 'force_order' or name not in WIDTHS:
        raise ValueError('One registered force-aware feature family required')
    kind = {'vx': 0, 'vy': 1, 'vz': 2, 'tracer': 3}[name]
    sizes = dimensions(context['counts']); shape = sizes[kind]
    prefix = previous_columns(model, context, name)
    result = torch.empty((WIDTHS[name], *shape), device='cuda', dtype=torch.float64)
    cursor = PREFIX[name]; result[:cursor] = prefix; del prefix
    late = context['features'][-1]
    velocity = [late[n].reshape(s).double() for n, s in zip(('vx', 'vy', 'vz'), sizes)]
    scale = 1./(60.*context['h'])
    contrasts = forcing_contrasts(context)
    if kind < 3:
        # 28 own-force columns, then12 U*d(force), then12 force*d(velocity).
        for force in contrasts:
            for offset in OFFSETS:
                result[cursor] = shifted(force[kind], offset); cursor += 1
        current = [model.linear(velocity, shape, 'point', kind*3+a)*scale for a in range(3)]
        for force in contrasts:
            for axis in range(3):
                result[cursor] = current[axis]*difference(force[kind], axis); cursor += 1
        for force in contrasts:
            for axis in range(3):
                at_target = model.linear(force, shape, 'point', kind*3+axis)*scale
                result[cursor] = at_target*difference(velocity[kind], axis); cursor += 1
        result[PREFIX[name]:] = wall(result[PREFIX[name]:], kind)
    else:
        tracer = late['tracer'].reshape(shape).double()
        for force in contrasts:
            for axis in range(3):
                displacement = model.linear(force, shape, 'point', 9+axis)*scale
                for direction in (-1, 1):
                    offset = [0, 0, 0]; offset[2-axis] = direction
                    value = displacement*(shifted(tracer, tuple(offset))-tracer)
                    for offset in OFFSETS:
                        result[cursor] = shifted(value, offset); cursor += 1
    if cursor != WIDTHS[name]:
        raise ValueError('Frozen force feature order/width changed')
    return result
