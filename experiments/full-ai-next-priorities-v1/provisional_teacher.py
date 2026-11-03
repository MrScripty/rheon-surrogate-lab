"""Offline provisional-velocity labels; never an inference input or solver."""
import json
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from temporal_features_v2 import geometry
from tied_momentum_readout import index_readout
from study_support import BASE,HERE,read,sha


def padded_velocity(fields,counts):
    nx,ny,nz=counts;values=[]
    for name,shape in zip(('vx','vy','vz'),((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))):
        v=torch.as_tensor(fields[name]).double().reshape(1,1,*shape)
        values.append(F.pad(v,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(values,1)*geometry(tuple(counts),'cpu')[1][1:4]


def vectors(value,counts):
    nx,ny,nz=counts
    return [value[0,0,:nz,:ny,:nx+1].contiguous().reshape(-1).numpy(),
        value[0,1,:nz,:ny+1,:nx].contiguous().reshape(-1).numpy(),
        value[0,2,:nz+1,:ny,:nx].contiguous().reshape(-1).numpy()]


def pressure_contribution(fields,scene,weight):
    p=torch.as_tensor(fields['pressure']).double().reshape(1,1,*scene['counts'][::-1])
    p=p-p.flatten()[0]
    return index_readout(p,weight)*(scene['cells_per_metre']/(60*1000))


def reconstructed_provisional(fields,scene,weight):
    # Native stores the corrected velocity as float32. Removing its pressure
    # contribution recovers the provisional velocity up to that rounding.
    return padded_velocity(fields,scene['counts'])-pressure_contribution(fields,scene,weight)


def packet(value,fields,scene):
    return dict(counts=scene['counts'],cell_size_m=1/scene['cells_per_metre'],dt_s=1/60,rho=1000.,
        pressure=np.asarray(fields['pressure'],dtype='f8').reshape(-1).tolist(),
        velocity=[v.astype('f4').astype('f8').tolist()for v in vectors(value,scene['counts'])])


def native(cases):
    build=BASE/'provisional-teacher-build-01';receipt=read(build/'build-receipt.json');binary=build/'rheon-provisional-teacher-probe'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'provisional_teacher_probe.rs')!=receipt['source_sha256']:
        raise ValueError('Independent native provisional teacher changed')
    if not 1<=len(cases)<=16:raise ValueError('Bounded offline cases required')
    raw=json.dumps(dict(cases=cases),allow_nan=False).encode()
    if len(raw)>64*1024**2:raise ValueError('Bounded teacher packet exceeded')
    run=subprocess.run([str(binary)],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
    outputs=json.loads(run.stdout)['cases']
    if len(outputs)!=len(cases)or any(v['pressure_solves']or v['native_steps_executed']for v in outputs):
        raise ValueError('Native primitive scope changed')
    return outputs
