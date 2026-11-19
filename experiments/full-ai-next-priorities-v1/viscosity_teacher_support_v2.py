"""Bounded native viscosity queries for offline controls and labels only."""
import json
import subprocess
import numpy as np
import torch
from study_support import BASE,HERE,read,sha


def native(cases):
    build=BASE/'viscosity-teacher-build-02';receipt=read(build/'build-receipt.json');binary=build/'rheon-viscosity-teacher-probe'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'viscosity_teacher_probe_v2.rs')!=receipt['source_sha256']:
        raise ValueError('Native viscosity probe changed')
    if sha(BASE/'temporal-build-02/rheon/src/viscosity.rs')!=receipt['native_viscosity_sha256']:
        raise ValueError('Captured native viscosity source changed')
    if not 1<=len(cases)<=2:raise ValueError('Bounded native viscosity cases required')
    raw=json.dumps(dict(cases=cases),allow_nan=False).encode()
    if len(raw)>64*1024**2:raise ValueError('Viscosity input cap exceeded')
    result=subprocess.run([str(binary)],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,check=True)
    outputs=json.loads(result.stdout)['cases']
    if len(outputs)!=len(cases)or any(o['pressure_solves']or o['native_steps_executed']for o in outputs):
        raise ValueError('Viscosity primitive scope changed')
    return outputs


def from_carrier(velocity,counts,cpm,density=1000.,mu=.001,dt=1/60):
    nx,ny,nz=counts;shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx));fields=[]
    for d,shape in enumerate(shapes):
        value=velocity[0,d,:shape[0],:shape[1],:shape[2]].detach().cpu().numpy().astype('f4',copy=True)
        selection=[slice(None)]*3;selection[2-d]=[0,-1];value[tuple(selection)]=0
        fields.append(value.reshape(-1).astype('f8').tolist())
    return dict(counts=list(counts),cell_size_m=1/cpm,dt_s=dt,density=density,dynamic_viscosity=mu,velocity=fields)


def packed_output(result,counts):
    nx,ny,nz=counts;out=torch.zeros(1,3,nz+1,ny+1,nx+1,dtype=torch.float32)
    for d,shape in enumerate(((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))):
        out[0,d,:shape[0],:shape[1],:shape[2]]=torch.tensor(result['updated_velocity'][d],dtype=torch.float32).reshape(shape)
    return out
