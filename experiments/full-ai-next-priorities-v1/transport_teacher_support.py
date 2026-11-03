"""Bounded independent native transport queries for offline qualification/data."""
import json
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from study_support import BASE,HERE,read,sha


def carrier(fields,counts):
    nx,ny,nz=counts;values=[]
    for name,shape in zip(('vx','vy','vz','tracer'),((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))):
        value=torch.from_numpy(np.asarray(fields[name],dtype='f4').copy()).reshape(1,1,*shape)
        values.append(F.pad(value,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(values,1)


def native(cases):
    build=BASE/'transport-teacher-build-01';receipt=read(build/'build-receipt.json');binary=build/'rheon-transport-teacher-probe'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'transport_teacher_probe.rs')!=receipt['source_sha256']:
        raise ValueError('Independent native transport teacher changed')
    if not 1<=len(cases)<=2:raise ValueError('Bounded transport cases required')
    raw=json.dumps(dict(cases=cases),allow_nan=False).encode()
    if len(raw)>64*1024**2:raise ValueError('Transport input cap exceeded')
    result=subprocess.run([str(binary)],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,check=True)
    outputs=json.loads(result.stdout)['cases']
    if len(outputs)!=len(cases)or any(r['pressure_solves']or r['native_steps_executed']for r in outputs):raise ValueError('Native primitive scope changed')
    return outputs


def packed_native_points(result,counts,key):
    nx,ny,nz=counts;values=[]
    for row in result['points']:
        shape=row['dimensions_xyz'][::-1]
        v=torch.tensor(row[key],dtype=torch.float64).reshape(1,3,*shape)
        values.append(F.pad(v,(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0])))
    return torch.stack(values,1)


def point_mask(counts,device='cpu'):
    nx,ny,nz=counts;out=torch.zeros(1,4,1,nz+1,ny+1,nx+1,device=device,dtype=torch.float64)
    for kind,shape in enumerate(((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))):
        out[:,kind,:,:shape[0],:shape[1],:shape[2]]=1
    return out
