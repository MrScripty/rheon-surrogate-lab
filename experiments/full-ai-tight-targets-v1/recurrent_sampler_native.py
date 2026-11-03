"""CPU execution of the declared learned sampler and learned sparse stages."""
import ctypes
from functools import lru_cache
from pathlib import Path
from profile_frozen_components import ROOT,read,digest
from fused_force_construction import fixed_geometry
from temporal_features_v2 import direction_phase
import math
import torch


@lru_cache(maxsize=1)
def library():
    directory=ROOT/'runs/full-ai-tight-targets-v1/recurrent-sampler-build-01';receipt=read(directory/'receipt.json')
    binary=directory/'librecurrent_sampler.so'
    if digest(binary)!=receipt['binary_sha256']or digest(Path(__file__).with_name('recurrent_sampler.cpp'))!=receipt['source_sha256']:
        raise ValueError('Registered learned sampler execution changed')
    lib=ctypes.CDLL(str(binary));ptr=ctypes.c_void_p;integer=ctypes.c_int;double=ctypes.c_double
    lib.rheon_learned_sampling.argtypes=[integer]*3+[ptr]*14+[double]
    lib.rheon_learned_sampling.restype=integer
    lib.rheon_learned_spatial_stage.argtypes=[integer]*4+[ptr]*14+[double]
    lib.rheon_learned_spatial_stage.restype=integer
    lib.rheon_conditioning_force.argtypes=[integer]*5+[ptr]*4+[double]*3+[ptr]
    lib.rheon_conditioning_force.restype=integer
    return lib


def dimensions(counts):
    nx,ny,nz=counts
    return ((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))


def pointers(values):return [v.data_ptr()if v is not None else None for v in values]


def transport(fields,counts,point,gate,mix,dt_over_h):
    outputs=[torch.empty(shape,dtype=torch.float32)for shape in dimensions(counts)]
    rows,channels,offsets,weights=point
    status=library().rheon_learned_sampling(*counts,*pointers(fields),*pointers(outputs),
        *pointers((rows,channels,offsets,weights,gate,mix)),dt_over_h)
    if status:raise ValueError('Learned sampler rejected input/parameter contract: '+str(status))
    return outputs


def spatial(kind,fields,counts,coefficient,scale,base=None):
    rows,channels,offsets,weights=coefficient
    if kind==1:
        output=torch.empty(tuple(reversed(counts)),dtype=torch.float64);targets=[None]*3;q=output
    else:targets=[torch.empty(shape,dtype=torch.float32)for shape in dimensions(counts)[:3]];q=None;output=targets
    inputs=list(fields)+[None]*(3-len(fields));bases=list(base)if base is not None else[None]*3
    status=library().rheon_learned_spatial_stage(kind,*counts,*pointers(inputs),*pointers(targets),
        q.data_ptr()if q is not None else None,*pointers(bases),*pointers((rows,channels,offsets,weights)),scale)
    if status:raise ValueError('Learned sparse stage rejected contract: '+str(status))
    return output


def forcing(scene,step,horizons):
    counts=tuple(scene['counts']);positions,spatial_phase,stride=fixed_geometry(counts,scene['cells_per_metre'])
    extents=[n/scene['cells_per_metre']for n in counts];radius=scene['radius']*min(extents)
    if radius<=0 or not math.isfinite(radius):raise ValueError('Finite positive force radius required')
    direction,phase=direction_phase(scene['seed']);omega=math.tau*scene['frequency']
    times=((torch.arange(horizons,dtype=torch.float64)+step)*(1./60.)).reshape(horizons,1)
    centers=torch.stack([extents[a]*(scene['center'][a]+.05*torch.sin(omega*times+phase+a))for a in range(3)])
    offsets=((positions-centers.unsqueeze(0))/radius).contiguous()
    frequency=torch.tensor([1,2,3],dtype=torch.float64).reshape(1,3,1,1)
    component=torch.arange(3,dtype=torch.float64).reshape(3,1,1,1)
    waves=torch.sin(spatial_phase+omega*times.reshape(1,1,horizons,1)*frequency+phase+component).contiguous()
    amplitude=(scene['acceleration']*(.65+.35*torch.sin(omega*times+phase))).contiguous()
    direction_tensor=torch.tensor(direction,dtype=torch.float64);shapes=dimensions(counts)[:3]
    lengths=[math.prod(shape)for shape in shapes];result=torch.empty((horizons,sum(lengths)),dtype=torch.float64)
    status=library().rheon_conditioning_force(*counts,horizons,stride,*pointers((offsets,waves,amplitude,direction_tensor)),
        scene['swirl'],scene['fluctuation'],1./60.,result.data_ptr())
    if status:raise ValueError('External conditioning construction failed: '+str(status))
    return [[part.reshape(shape)for part,shape in zip(row.split(lengths),shapes)]for row in result]
