"""Capture exact native strain-update labels, without assigning stencil weights."""
import argparse
import os
from pathlib import Path
import hashlib
import numpy as np
import torch
from prepare_flux_teacher import fields
from viscosity_teacher_support_v2 import native,from_carrier,packed_output
from temporal_features_v2 import geometry
from study_support import HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh viscosity calibration teacher required')
    registration_path=DOCS/'viscosity-calibration-registration-01.json';registration=read(registration_path)
    for name,digest in registration['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered viscosity-calibration source changed')
    qualification=BASE/'viscosity-teacher-qualification-02.json'
    if read(qualification)['status']!='passed'or sha(qualification)!=registration['native_qualification_sha256']:
        raise ValueError('Exact native primitive/label qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);args.output.mkdir(parents=True)
    rng=np.random.Generator(np.random.PCG64(registration['data_seed']));counts=[12,12,12]
    velocity,_=fields(rng,counts,80)
    velocity=(velocity*16).round().clamp(-16,16)/16
    velocity=torch.where(velocity==0,torch.zeros_like(velocity),velocity)
    mask=geometry(tuple(counts),'cpu')[1][1:4].unsqueeze(0).double()
    targets=[];lattice_errors=[]
    for start in range(0,80,2):
        cases=[from_carrier(velocity[i:i+1],counts,16,density=1.,mu=1.,dt=(1/16)**2/64)for i in (start,start+1)]
        for index,result in zip((start,start+1),native(cases)):
            target=(packed_output(result,counts).double()-velocity[index:index+1].double())/result['beta']
            target=target*mask
            error=float((target*16-(target*16).round()).abs().max())
            if error!=0:raise ValueError('Unrounded native label lattice differs')
            targets.append(target.numpy());lattice_errors.append(error)
    archive=args.output/'teacher.npz'
    np.savez_compressed(archive,velocity=velocity.numpy(),target=np.concatenate(targets),mask=mask.numpy())
    with np.load(archive,allow_pickle=False)as arrays:
        inventory={name:dict(shape=list(arrays[name].shape),dtype=arrays[name].dtype.str,
            sha256=hashlib.sha256(arrays[name].tobytes()).hexdigest())for name in arrays.files}
    write(args.output/'manifest.json',dict(status='native_qualified',schema='rheon-viscosity-calibration-teacher-v1',
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,inventory=inventory,
        train_indices=list(range(64)),development_indices=list(range(64,80)),counts_xyz=counts,
        target='Dimensionless negative symmetric-strain stiffness on free face degrees of freedom; exact native dyadic update labels',
        maximum_label_lattice_error=max(lattice_errors),native_qualification_sha256=sha(qualification),
        native_build_receipt_sha256=sha(BASE/'viscosity-teacher-build-02/build-receipt.json'),
        registration_sha256=sha(registration_path),source_sha256=registration['source_sha256'],
        analytic_coefficient_assignment=False,optimizer_updates=0,temporal_scenes_used=False,reserved_scenes_consumed=False))
    print('Captured native viscosity teacher:80 patterns, exact lattice; bytes',archive.stat().st_size,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
