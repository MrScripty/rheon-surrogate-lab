"""Causal cross-carrier feature columns for the final coupled correction."""
import torch
from recurrent_sampler_native import dimensions
from coupled_macrostep_v1 import wall


OFFSETS = ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))


@torch.no_grad()
def columns(model, context, name, conditioned):
    if name not in ('vx', 'vy', 'vz', 'tracer') or len(context['features']) != 4:
        raise ValueError('Registered causal grouped4context and carrier required')
    kind = {'vx': 0, 'vy': 1, 'vz': 2, 'tracer': 3}[name]
    sizes = dimensions(context['counts']); shape = sizes[kind]
    late = context['features'][-1]
    current = [late[n].reshape(s).double() for n, s in zip(('vx', 'vy', 'vz'), sizes)]
    values = []
    for state in context['features'][:-1]:
        difference = [state[n].reshape(s).double()-v for n, s, v in zip(('vx', 'vy', 'vz'), sizes, current)]
        for axis in range(3): values.append(model.linear(difference, shape, 'point', kind*3+axis))
    if name == 'tracer':
        values.extend(state[name].double().reshape(shape)-late[name].double().reshape(shape) for state in context['features'][:-1])
    base = context['tracer'] if name == 'tracer' else context['provisional'][kind]
    values.append(base.double()-late[name].double().reshape(shape))
    if len(values) != (13 if kind == 3 else 10): raise ValueError('Exact feature inventory required')
    result = torch.empty((len(values)*7, *shape), dtype=torch.float64, device='cuda')
    coordinates = [torch.arange(n, device='cuda') for n in shape]
    col = 0
    for value in values:
        for dz, dy, dx in OFFSETS:
            z, y, x = [(a+d).clamp(0, n-1) for a, d, n in zip(coordinates, (dz, dy, dx), shape)]
            result[col] = value[z[:, None, None], y[None, :, None], x[None, None, :]]
            col += 1
    if conditioned:
        base_columns = result
        result = torch.empty((4*base_columns.shape[0], *shape), dtype=torch.float64, device='cuda')
        width = base_columns.shape[0]; result[:width] = base_columns
        for axis in range(3):
            modulation = model.linear(current, shape, 'point', kind*3+axis) * (1./(60.*context['h']))
            result[(axis+1)*width:(axis+2)*width] = base_columns * modulation
    return result if kind == 3 else wall(result, kind)
