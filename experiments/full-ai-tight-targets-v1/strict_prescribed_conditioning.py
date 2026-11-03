"""Two fixed external-conditioning arithmetic controls; no learned dynamics change."""
import ctypes
import math
from functools import lru_cache
from profile_frozen_components import ROOT, read, digest
from fused_force_construction import fixed_geometry
from temporal_features_v2 import direction_phase
import torch


@lru_cache(maxsize=1)
def library():
    directory = ROOT / 'runs/full-ai-tight-targets-v1/strict-conditioning-build-10'
    receipt = read(directory / 'receipt.json')
    binary = directory / 'libstrict_conditioning.so'
    if digest(binary) != receipt['binary_sha256'] or digest(ROOT / 'experiments/full-ai-tight-targets-v1/strict_prescribed_conditioning.cpp') != receipt['source_sha256']:
        raise ValueError('Registered scalar Gaussian conditioning changed')
    lib = ctypes.CDLL(str(binary))
    lib.rheon_conditioning_force.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 4 + [ctypes.c_double] * 3 + [ctypes.c_void_p]
    lib.rheon_conditioning_force.restype = ctypes.c_int
    return lib


def forcing(scene, step, horizons, policy):
    if policy not in ('scalar_gaussian', 'scalar_gaussian_and_sine') or horizons not in (1, 2, 3, 4):
        raise ValueError('Registered fixed conditioning policy required')
    counts = tuple(scene['counts'])
    positions, spatial_phase, stride = fixed_geometry(counts, scene['cells_per_metre'])
    extents = [n / scene['cells_per_metre'] for n in counts]
    radius = scene['radius'] * min(extents)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Finite positive external force radius required')
    direction, phase = direction_phase(scene['seed'])
    omega = math.tau * scene['frequency']
    if policy == 'scalar_gaussian':
        times = ((torch.arange(horizons, dtype=torch.float64) + step) * (1 / 60)).reshape(horizons, 1)
        centers = torch.stack([extents[a] * (scene['center'][a] + .05 * torch.sin(omega * times + phase + a)) for a in range(3)])
        offsets = ((positions - centers.unsqueeze(0)) / radius).contiguous()
        frequency = torch.tensor([1, 2, 3], dtype=torch.float64).reshape(1, 3, 1, 1)
        component = torch.arange(3, dtype=torch.float64).reshape(3, 1, 1, 1)
        waves = torch.sin(spatial_phase + omega * times.reshape(1, 1, horizons, 1) * frequency + phase + component).contiguous()
        amplitude = (scene['acceleration'] * (.65 + .35 * torch.sin(omega * times + phase))).contiguous()
    else:
        offsets = torch.empty((3, 3, horizons, stride), dtype=torch.float64)
        waves = torch.empty_like(offsets)
        amplitude = torch.empty((horizons, 1), dtype=torch.float64)
        coordinates = positions[:, :, 0].tolist()
        for lead in range(horizons):
            time = float(step + lead) * (1 / 60)
            centers = [extents[a] * (scene['center'][a] + .05 * math.sin(omega * time + phase + a)) for a in range(3)]
            amplitude[lead, 0] = scene['acceleration'] * (.65 + .35 * math.sin(omega * time + phase))
            for d in range(3):
                for a in range(3):
                    offsets[d, a, lead] = torch.tensor([(p - centers[a]) / radius for p in coordinates[d][a]], dtype=torch.float64)
                    waves[d, a, lead] = torch.tensor([math.sin(math.tau * p / extents[a] + omega * time * (a + 1) + phase + d) for p in coordinates[d][a]], dtype=torch.float64)
    shapes = ((counts[2], counts[1], counts[0] + 1),
              (counts[2], counts[1] + 1, counts[0]),
              (counts[2] + 1, counts[1], counts[0]))
    lengths = [math.prod(shape) for shape in shapes]
    result = torch.empty((horizons, sum(lengths)), dtype=torch.float64)
    direction_tensor = torch.tensor(direction, dtype=torch.float64)
    status = library().rheon_conditioning_force(*counts, horizons, stride,
        offsets.data_ptr(), waves.data_ptr(), amplitude.data_ptr(), direction_tensor.data_ptr(),
        scene['swirl'], scene['fluctuation'], 1 / 60, result.data_ptr())
    if status:
        raise ValueError('Scalar conditioning contract rejected: ' + str(status))
    return [[part.reshape(shape) for part, shape in zip(row.split(lengths), shapes)] for row in result]
