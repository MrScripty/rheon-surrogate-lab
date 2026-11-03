"""Fused execution of the separately qualified compact state representation."""
import ctypes
from functools import lru_cache
import math
from pathlib import Path
import torch
from profile_frozen_components import ROOT, read, digest


@lru_cache(maxsize=1)
def library():
    directory = ROOT / 'runs/full-ai-tight-targets-v1/compact-codec-build-01'
    receipt = read(directory / 'receipt.json')
    binary = directory / 'libcompact_codec.so'
    if digest(binary) != receipt['binary_sha256'] or digest(Path(__file__).with_name('compact_codec.cpp')) != receipt['source_sha256']:
        raise ValueError('Registered compact codec changed')
    value = ctypes.CDLL(str(binary))
    arguments = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 3 + [ctypes.c_double, ctypes.c_float, ctypes.c_float]
    for name in ('rheon_compact_pack', 'rheon_compact_unpack'):
        function = getattr(value, name); function.argtypes = arguments; function.restype = ctypes.c_int
    return value


def checked_tensor(value, shape, dtype):
    value = torch.as_tensor(value)
    if value.device.type != 'cpu' or value.dtype != dtype or value.numel() != math.prod(shape) or not value.is_contiguous():
        raise ValueError('Exact contiguous physical field dtype and geometry required')
    if value.requires_grad:
        raise ValueError('Native codec is inference only')
    return value.reshape(shape)


def pack(fields, counts, scales):
    if set(fields) != {'pressure', 'vx', 'vy', 'vz', 'tracer'}:
        raise ValueError('Complete current physical state required')
    nx, ny, nz = counts
    shapes = ((nz, ny, nx), (nz, ny, nx+1), (nz, ny+1, nx), (nz+1, ny, nx), (nz, ny, nx))
    arrays = [checked_tensor(fields[name], shape, torch.float64 if name == 'pressure' else torch.float32)
              for name, shape in zip(('pressure', 'vx', 'vy', 'vz', 'tracer'), shapes)]
    pressure = torch.empty((1, 1, nz, ny, nx), dtype=torch.float64)
    carrier = torch.empty((1, 4, nz, ny, nx), dtype=torch.float32)
    status = library().rheon_compact_pack(*[v.data_ptr() for v in arrays], pressure.data_ptr(), carrier.data_ptr(),
        nx, ny, nz, scales['pressure'], scales['velocity'], scales['tracer'])
    if status: raise ValueError('Compact pack rejected geometry, finite inputs or sealed walls: ' + str(status))
    return pressure, carrier


def unpack(pressure, carrier, counts, scales):
    nx, ny, nz = counts
    pressure = checked_tensor(pressure, (1, 1, nz, ny, nx), torch.float64)
    carrier = checked_tensor(carrier, (1, 4, nz, ny, nx), torch.float32)
    shapes = ((nz, ny, nx), (nz, ny, nx+1), (nz, ny+1, nx), (nz+1, ny, nx), (nz, ny, nx))
    outputs = [torch.empty(shape, dtype=torch.float64 if i == 0 else torch.float32) for i, shape in enumerate(shapes)]
    status = library().rheon_compact_unpack(pressure.data_ptr(), carrier.data_ptr(), *[v.data_ptr() for v in outputs],
        nx, ny, nz, scales['pressure'], scales['velocity'], scales['tracer'])
    if status: raise ValueError('Compact unpack rejected geometry or nonfinite outputs: ' + str(status))
    return {name: value.reshape(-1) for name, value in zip(('pressure', 'vx', 'vy', 'vz', 'tracer'), outputs)}
