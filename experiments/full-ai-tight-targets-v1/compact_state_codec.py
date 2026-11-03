"""Cell-shaped carrier features with separate physical-pressure precision.

Only the known zero upper normal face is omitted from each staggered velocity.
No interior face is averaged or interpolated. Closed sealed domains only.
"""
import torch
from torch.nn import functional as F

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def pack(fields, counts, scales):
    if set(fields) != set(FIELDS):
        raise ValueError('Complete current physical state required')
    nx, ny, nz = counts
    shapes = ((nz, ny, nx), (nz, ny, nx + 1), (nz, ny + 1, nx),
              (nz + 1, ny, nx), (nz, ny, nx))
    values = [torch.as_tensor(fields[name]).reshape(shape) for name, shape in zip(FIELDS, shapes)]
    if any(v.device.type != 'cpu' or not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Finite complete CPU state required')
    for value, axis in zip(values[1:4], (2, 1, 0)):
        if torch.count_nonzero(value.select(axis, 0)) or torch.count_nonzero(value.select(axis, value.shape[axis] - 1)):
            raise ValueError('Compact codec requires zero normal velocity on sealed walls')
    pressure = (values[0].double() / scales['pressure']).reshape(1, 1, nz, ny, nx)
    carriers = torch.stack([value[:nz, :ny, :nx].float() / scales['tracer' if index == 3 else 'velocity']
                            for index, value in enumerate(values[1:])]).unsqueeze(0)
    return pressure, carriers


def unpack(pressure, carriers, counts, scales):
    nx, ny, nz = counts
    if tuple(pressure.shape) != (1, 1, nz, ny, nx) or pressure.dtype != torch.float64:
        raise ValueError('Separate double pressure head is required')
    if tuple(carriers.shape) != (1, 4, nz, ny, nx) or carriers.dtype != torch.float32:
        raise ValueError('Four cell-shaped float32 carrier heads required')
    p = pressure[0, 0] * scales['pressure']
    result = {'pressure': (p - p[:1, :1, :1]).contiguous().reshape(-1)}
    for channel, (name, axis) in enumerate(zip(('vx', 'vy', 'vz'), (2, 1, 0))):
        value = carriers[0, channel] * scales['velocity']
        value.select(axis, 0).zero_()
        padding = [0] * 6
        padding[2 * (2 - axis) + 1] = 1
        result[name] = F.pad(value, padding, mode='constant', value=0).contiguous().reshape(-1)
    result['tracer'] = (carriers[0, 3] * scales['tracer']).contiguous().reshape(-1)
    return result
