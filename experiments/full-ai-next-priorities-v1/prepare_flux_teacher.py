"""Qualify synthetic flux labels against native Rheon RHS and freeze training data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from study_support import ROOT,HERE,BASE,read,write,sha
from flux_teacher import dimensionless_rhs


def fields(rng,counts,batch):
    nx,ny,nz=counts;result=[];physical=[]
    for d,shape in enumerate(((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))):
        value=rng.normal(size=(batch,1,*shape)).astype('<f4')
        lower=[slice(None)]*5;upper=lower.copy();lower[4-d]=0;upper[4-d]=-1
        value[tuple(lower)]=0;value[tuple(upper)]=0
        physical.append(value.reshape(batch,-1))
        result.append(F.pad(torch.from_numpy(value),(0,nx+1-shape[2],0,ny+1-shape[1],0,nz+1-shape[0]),mode='replicate'))
    return torch.cat(result,dim=1),physical


def run(args):
    if args.output.exists():raise ValueError('Fresh native-qualified flux teacher required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    build=BASE/'flux-teacher-native-build-01';binary=build/'rheon-flux-teacher-probe'
    if sha(binary)!=read(build/'build-receipt.json')['binary_sha256']:raise ValueError('Native teacher binary changed')
    rng=np.random.Generator(np.random.PCG64(2026101997));cases=[];expected=[]
    for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
        for cpm in (8,16):
            padded,physical=fields(rng,counts,1);h=1/cpm
            cases.append(dict(counts=counts,cell_size_m=h,dt_s=1/60,rho=1000.,velocity=[value[0].tolist()for value in physical]))
            expected.append(dimensionless_rhs(padded.double()).numpy().reshape(-1)*h*h/(1/60))
    result=subprocess.run([str(binary)],input=json.dumps(dict(cases=cases)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
    native=json.loads(result.stdout);qualified=[]
    for case,truth,reference in zip(cases,expected,native['cases']):
        rhs=np.asarray(reference['rhs'],dtype='f8');relative=float(np.linalg.norm(rhs-truth))/max(float(np.linalg.norm(rhs)),1e-30)
        if relative>1e-12 or reference['native_steps_executed']!=0 or reference['pressure_solves']!=0:
            raise ValueError('Offline flux teacher differs from native RHS assembly')
        qualified.append(dict(counts_xyz=case['counts'],cells_per_metre=round(1/case['cell_size_m']),relative_error=relative))
    rng=np.random.Generator(np.random.PCG64(2026101998));padded,_=fields(rng,[12,12,12],80)
    values=padded.numpy();target=dimensionless_rhs(padded.double()).numpy()
    archive=args.output/'teacher.npz';np.savez_compressed(archive,velocity=values,target=target)
    write(args.output/'manifest.json',dict(status='native_qualified',schema='rheon-linear-flux-teacher-v1',
        train_indices=list(range(64)),development_indices=list(range(64,80)),counts_xyz=[12,12,12],
        input_dtype='<f4',target_dtype='<f8',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        native_qualification=qualified,native_build_receipt_sha256=sha(build/'build-receipt.json'),
        source_sha256=sha(__file__),teacher_source_sha256=sha(HERE/'flux_teacher.py'),
        target='Dimensionless signed face differences; native h^2/dt conversion qualified separately',
        temporal_scenes_used=False,temporal_reserved_scenes_consumed=False,optimizer_updates=0))
    print('Native-qualified flux teacher, eight geometry/spacing checks;',archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
