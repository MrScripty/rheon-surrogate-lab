"""Deterministic pressure eigenmodes and unchanged-operator spectral labels."""
import math
import torch
def pattern(row,index):
    import numpy as np
    rng=np.random.Generator(np.random.PCG64(row['seed']+index));shape=row['counts_xyz'][::-1]
    low,high=row['mode_band'];modes=[]
    while len(modes)<row['mode_count']:
        mode=tuple(int(rng.integers(0,min(n-1,high)+1))for n in shape)
        if max(mode)<low or mode in modes:continue
        modes.append(mode)
    value=torch.zeros((1,1,*shape),dtype=torch.float64);details=[]
    for mode in modes:
        term=torch.ones(shape,dtype=torch.float64);coefficient=float(rng.normal())
        for axis,(n,k)in enumerate(zip(shape,mode)):
            view=[1]*3;view[axis]=n
            term*=torch.cos(math.pi/n*k*(torch.arange(n,dtype=torch.float64)+.5)).reshape(view)
        value+=coefficient*term
        details.append(dict(mode_zyx=list(mode),coefficient=coefficient,wavelength_metres_zyx=[2*n/row['cells_per_metre']/k if k else None for n,k in zip(shape,mode)]))
    value*=row['pressure_rms_pa']/float(value.square().mean().sqrt());value-=value[:,:,:1,:1,:1].clone()
    return value,details

