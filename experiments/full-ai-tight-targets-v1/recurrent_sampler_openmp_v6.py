"""Strict row-parallel execution, explicitly limited to one, two or four cores."""
import ctypes
from functools import lru_cache
from pathlib import Path
from profile_frozen_components import ROOT, read, digest


@lru_cache(maxsize=1)
def library():
    directory = ROOT / 'runs/full-ai-tight-targets-v1/recurrent-sampler-openmp-build-06'
    receipt = read(directory / 'receipt.json')
    binary = directory / 'librecurrent_sampler.so'
    source = Path(__file__).with_suffix('.cpp')
    if digest(binary) != receipt['binary_sha256'] or digest(source) != receipt['source_sha256']:
        raise ValueError('Registered OpenMP source/binary changed')
    lib = ctypes.CDLL(str(binary))
    ptr, integer, double = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.rheon_learned_sampling.argtypes = [integer] * 3 + [ptr] * 14 + [double]
    lib.rheon_learned_sampling.restype = integer
    lib.rheon_learned_spatial_stage.argtypes = [integer] * 4 + [ptr] * 14 + [double]
    lib.rheon_learned_spatial_stage.restype = integer
    lib.rheon_conditioning_force.argtypes = [integer] * 5 + [ptr] * 4 + [double] * 3 + [ptr]
    lib.rheon_conditioning_force.restype = integer
    lib.rheon_set_execution_threads.argtypes = [integer]
    lib.rheon_set_execution_threads.restype = integer
    lib.rheon_probe_execution_team.argtypes = [ptr]
    lib.rheon_probe_execution_team.restype = integer
    return lib


def configure(cores):
    """Call before inference in a fresh, explicitly pinned worker process."""
    lib = library()
    if lib.rheon_set_execution_threads(len(cores)) != 0:
        raise ValueError('Only one, two or four execution workers allowed')
    placements = (ctypes.c_int * 4)(-1, -1, -1, -1)
    observed = lib.rheon_probe_execution_team(placements)
    cpus = list(placements)[:observed]
    if observed != len(cores) or set(cpus) != set(cores):
        raise ValueError('OpenMP workers did not occupy exactly the declared cores: ' + str(cpus))
    return dict(requested_threads=len(cores), observed_threads=observed,
                worker_cpus=cpus)
