"""Versioned physical-coordinate sampler for the unchanged learned graph."""
import argparse
import ctypes
from functools import lru_cache
from pathlib import Path
from profile_frozen_components import ROOT,read,digest
import recurrent_sampler_native as native


@lru_cache(maxsize=1)
def library():
    directory=ROOT/'runs/full-ai-tight-targets-v1/recurrent-sampler-physical-build-04'
    receipt=read(directory/'receipt.json');binary=directory/'librecurrent_sampler.so'
    source=Path(__file__).with_name('recurrent_sampler_physical_v4.cpp')
    if digest(binary)!=receipt['binary_sha256']or digest(source)!=receipt['source_sha256']:
        raise ValueError('Regular-address execution source/binary changed')
    lib=ctypes.CDLL(str(binary));ptr=ctypes.c_void_p;integer=ctypes.c_int;double=ctypes.c_double
    lib.rheon_learned_sampling.argtypes=[integer]*3+[ptr]*14+[double]
    lib.rheon_learned_sampling.restype=integer
    lib.rheon_learned_spatial_stage.argtypes=[integer]*4+[ptr]*14+[double]
    lib.rheon_learned_spatial_stage.restype=integer
    lib.rheon_conditioning_force.argtypes=[integer]*5+[ptr]*4+[double]*3+[ptr]
    lib.rheon_conditioning_force.restype=integer
    return lib


