"""Versioned integer-address implementation of the unchanged learned graph."""
import argparse
import ctypes
from functools import lru_cache
from pathlib import Path
from profile_frozen_components import ROOT,read,digest
import recurrent_sampler_native as native


@lru_cache(maxsize=1)
def library():
    directory=ROOT/'runs/full-ai-tight-targets-v1/recurrent-sampler-integer-build-02'
    receipt=read(directory/'receipt.json');binary=directory/'librecurrent_sampler.so'
    source=Path(__file__).with_name('recurrent_sampler_integer_v2.cpp')
    if digest(binary)!=receipt['binary_sha256']or digest(source)!=receipt['source_sha256']:
        raise ValueError('Integer-address execution source/binary changed')
    lib=ctypes.CDLL(str(binary));ptr=ctypes.c_void_p;integer=ctypes.c_int;double=ctypes.c_double
    lib.rheon_learned_sampling.argtypes=[integer]*3+[ptr]*14+[double]
    lib.rheon_learned_sampling.restype=integer
    lib.rheon_learned_spatial_stage.argtypes=[integer]*4+[ptr]*14+[double]
    lib.rheon_learned_spatial_stage.restype=integer
    lib.rheon_conditioning_force.argtypes=[integer]*5+[ptr]*4+[double]*3+[ptr]
    lib.rheon_conditioning_force.restype=integer
    return lib


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','benchmark'),required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,choices=(6,23),default=23)
    args=parser.parse_args()
    native.library=library
    if args.mode=='qualify':
        import qualify_recurrent_sampler as qualification
        from recurrent_sampler_constructor import construct
        qualification.RecurrentSamplerPrototype=construct
        qualification.run(args)
    else:
        import benchmark_recurrent_sampler as benchmark
        benchmark.run(args)
