"""Compact CPU execution of prescribed forcing with axis Gaussian reuse."""
import ctypes
from functools import lru_cache
import math
from pathlib import Path
from profile_frozen_components import ROOT, read, digest
from fused_force_construction import fixed_geometry
from temporal_features_v2 import direction_phase
import torch


@lru_cache(maxsize=1)
def library():
    directory = ROOT / 'runs/full-ai-tight-targets-v1/factorized-force-build-01'
    receipt = read(directory / 'receipt.json')
    binary = directory / 'libcompact_force.so'
    if digest(binary) != receipt['binary_sha256'] or digest(Path(__file__).with_suffix('.cpp')) != receipt['source_sha256']:
        raise ValueError('Registered compact force kernel changed')
    lib = ctypes.CDLL(str(binary))
    lib.rheon_compact_prescribed_force.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 5 + [ctypes.c_double] * 2 + [ctypes.c_float]
    lib.rheon_compact_prescribed_force.restype = ctypes.c_int
    return lib


def force_features(scene, step, horizons, velocity_scale):
    if horizons not in (1, 2, 3, 4) or scene.get('sgs', False):
        raise ValueError('Qualified closed-domain forcing contract required')
    counts = tuple(scene['counts'])
    positions, spatial_phase, stride = fixed_geometry(counts, scene['cells_per_metre'])
    extents = [n / scene['cells_per_metre'] for n in counts]
    radius = scene['radius'] * min(extents)
    direction, phase = direction_phase(scene['seed'])
    omega = math.tau * scene['frequency']
    times = ((torch.arange(horizons, dtype=torch.float64) + step) / 60).reshape(horizons, 1)
    centers = torch.stack([extents[a] * (scene['center'][a] + .05 * torch.sin(omega * times + phase + a)) for a in range(3)])
    offsets = ((positions - centers.unsqueeze(0)) / radius).contiguous()
    frequency = torch.tensor([1, 2, 3], dtype=torch.float64).reshape(1, 3, 1, 1)
    component = torch.arange(3, dtype=torch.float64).reshape(3, 1, 1, 1)
    waves = torch.sin(spatial_phase + omega * times.reshape(1, 1, horizons, 1) * frequency + phase + component).contiguous()
    amplitude = (scene['acceleration'] * (.65 + .35 * torch.sin(omega * times + phase))).contiguous()
    direction_tensor = torch.tensor(direction, dtype=torch.float64)
    nx, ny, nz = counts
    output = torch.empty((horizons, 3, nz, ny, nx), dtype=torch.float32)
    status = library().rheon_compact_prescribed_force(offsets.data_ptr(), waves.data_ptr(), amplitude.data_ptr(), direction_tensor.data_ptr(),
        output.data_ptr(), nx, ny, nz, horizons, stride, scene['swirl'], scene['fluctuation'], velocity_scale)
    if status:
        raise ValueError('Compact forcing rejected input: ' + str(status))
    return output
