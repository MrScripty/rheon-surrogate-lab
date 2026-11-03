"""One bounded nonlinear feature extension: signed and absolute point velocity."""
import torch
from coupled_macrostep_features_v1 import columns as base_columns
from coupled_macrostep_v1 import wall
from recurrent_sampler_native import dimensions


@torch.no_grad()
def columns(model, context, name, mode):
    if mode != 'absolute_velocity': raise ValueError('Only the registered nonlinear basis is enabled')
    kind = {'vx': 0, 'vy': 1, 'vz': 2, 'tracer': 3}[name]
    sizes = dimensions(context['counts']); shape = sizes[kind]
    base = base_columns(model, context, name, False)
    width = base.shape[0]
    result = torch.empty((7*width, *shape), dtype=torch.float64, device='cuda')
    result[:width] = base
    state = context['features'][-1]
    current = [state[n].reshape(s).double() for n, s in zip(('vx', 'vy', 'vz'), sizes)]
    for axis in range(3):
        value = model.linear(current, shape, 'point', kind*3+axis) * (1./(60.*context['h']))
        result[(axis+1)*width:(axis+2)*width] = base*value
        result[(axis+4)*width:(axis+5)*width] = base*value.abs()
    return result if kind == 3 else wall(result, kind)
