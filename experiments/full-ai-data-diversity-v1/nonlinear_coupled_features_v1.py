"""Causal polynomial/pressure-history augmentation of the coupled feature basis.

No teacher, numerical equation action, coefficient solve or pressure inverse
belongs here. All pressure feedback uses restored learned momentum/point maps.
"""
import itertools
import torch
from coupled_macrostep_features_v1 import columns as conditioned_columns, OFFSETS
from coupled_macrostep_v1 import wall
from recurrent_sampler_native import dimensions


FULL_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))
QUADRATIC_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
WIDTHS = {'vx': 448, 'vy': 448, 'vz': 448, 'tracer': 610}


def shifted(value, offset):
    axes = [(torch.arange(n, device='cuda') + d).clamp(0, n-1)
            for n, d in zip(value.shape, offset)]
    return value[axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]]


@torch.no_grad()
def columns(model, context, name, mode='pressure_history_quadratic'):
    if mode != 'pressure_history_quadratic' or name not in WIDTHS or len(context['features']) != 4:
        raise ValueError('One registered causal coupled feature family required')
    kind = {'vx': 0, 'vy': 1, 'vz': 2, 'tracer': 3}[name]
    sizes = dimensions(context['counts']); shape = sizes[kind]
    prefix = conditioned_columns(model, context, name, True)
    base_width = 91 if kind == 3 else 70
    center = prefix[:base_width].reshape(base_width//7, 7, *shape)[:, 3]
    result = torch.empty((WIDTHS[name], *shape), device='cuda', dtype=torch.float64)
    result[:4*base_width] = prefix
    cursor = 4*base_width
    pressure = [s['pressure'].reshape(context['shape']).double() for s in context['features']]
    pressure = [p-p.mean() for p in pressure]
    histories = [p-pressure[-1] for p in pressure[:-1]] + [pressure[-1]]
    scale = context['dt']*(1./(context['rho']*context['h']))
    if kind < 3:
        for p in histories:
            feedback = wall(model.linear([p], shape, 'momentum', kind)*scale, kind)
            for offset in FULL_OFFSETS:
                result[cursor] = shifted(feedback, offset); cursor += 1
    else:
        tracer = context['features'][-1]['tracer'].reshape(shape).double()
        contrasts = []
        for axis in range(3):
            for direction in (-1, 1):
                offset = [0, 0, 0]; offset[2-axis] = direction
                contrasts.append(shifted(tracer, tuple(offset))-tracer)
        for p in histories:
            feedback = [wall(model.linear([p], sizes[c], 'momentum', c)*scale, c) for c in range(3)]
            for axis in range(3):
                at_cell = model.linear(feedback, shape, 'point', 9+axis)*(1./(60.*context['h']))
                for direction in range(2):
                    value = at_cell*contrasts[2*axis+direction]
                    for offset in OFFSETS:
                        result[cursor] = shifted(value, offset); cursor += 1
    late = context['features'][-1]
    velocity = [late[n].reshape(s).double() for n, s in zip(('vx', 'vy', 'vz'), sizes)]
    current = [model.linear(velocity, shape, 'point', kind*3+a)*(1./(60.*context['h'])) for a in range(3)]
    width = center.shape[0]
    for a, b in QUADRATIC_PAIRS:
        result[cursor:cursor+width] = center*(current[a]*current[b]); cursor += width
    if cursor != WIDTHS[name]:
        raise ValueError('Frozen feature order/width changed')
    return result if kind == 3 else wall(result, kind)
