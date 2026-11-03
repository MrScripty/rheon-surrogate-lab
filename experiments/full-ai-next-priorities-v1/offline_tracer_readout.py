"""Offline representation checks; native offsets are never predictor inputs."""
import json
import subprocess
import torch
from bundle_model_decoder import donors
from study_support import BASE,HERE,read,sha


def factorized_weights(shift):
    if shift.ndim!=5 or shift.shape[1]!=3 or not bool(torch.isfinite(shift).all())or float(shift.abs().max())>1:
        raise ValueError('Finite three-axis subcell displacement required')
    components=[]
    for d in range(3):
        s=shift[:,d:d+1].double()
        components.append(((-s).clamp_min(0),1-s.abs(),s.clamp_min(0)))
    x,y,z=components
    return torch.cat([z[a]*y[b]*x[c]for a in range(3)for b in range(3)for c in range(3)],1)


def readout(tracer,shift):
    return (donors(tracer).double()*factorized_weights(shift)).sum(1,keepdim=True).float()


def native(cases):
    build=BASE/'tracer-teacher-build-01';receipt=read(build/'build-receipt.json');binary=build/'rheon-tracer-teacher-probe'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'tracer_teacher_probe.rs')!=receipt['source_sha256']:
        raise ValueError('Registered native tracer teacher changed')
    if not 1<=len(cases)<=4:raise ValueError('Bounded teacher cases required')
    raw=json.dumps(dict(cases=cases),allow_nan=False).encode()
    if len(raw)>64*1024**2:raise ValueError('Bounded teacher input exceeded')
    output=subprocess.run([str(binary)],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,check=True)
    rows=json.loads(output.stdout)['cases']
    if len(rows)!=len(cases)or any(r['pressure_solves']or r['native_steps_executed']for r in rows):raise ValueError('Offline primitive scope changed')
    return rows
