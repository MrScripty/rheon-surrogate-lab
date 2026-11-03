"""Qualify valid eight-mode labels using independent actual stencil actions."""
import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from study_support import BASE,read,write,sha
import numerics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh modal teacher qualification required')
    torch.set_num_threads(1)
    bank=BASE/'tail-teacher-bank-01';manifest=read(bank/'manifest.json')
    if sha(bank/'teacher.npz')!=manifest['archive_sha256']:raise ValueError('Teacher archive changed')
    with np.load(bank/'teacher.npz',allow_pickle=False)as source:arrays={name:source[name].copy()for name in source.files}
    for name,value in arrays.items():
        spec=manifest['inventory'][name]
        if list(value.shape)!=spec['shape']or value.dtype.str!=spec['dtype']or hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Teacher array identity differs')
    counts=arrays['counts_zyx'];modes=arrays['mode_indices_zyx']
    if not np.array_equal(arrays['valid_mode'],(modes[None,:,:]<counts[:,None,:]).all(2)):
        raise ValueError('Validity mask differs from native mode range')
    if not np.array_equal(np.rint(np.exp(arrays['features'].astype('f8'))*16).astype('i4'),counts):
        raise ValueError('Training feature-derived mask would select different modes')
    fraction=arrays['teacher_gain']/(counts.max(1).astype('f8')**2/2)[:,None]
    relative=float(np.max(np.abs(np.exp(arrays['log_gain_fraction'].astype('f8'))-fraction)/fraction))
    if relative>1e-6:raise ValueError('Log label float32 conversion exceeds registered tolerance')
    checks=[]
    for shape in((16,16,16),(11,11,9),(8,6,15),(5,5,5)):
        triples=[(1,0,0),(0,1,0),(0,0,1),tuple(min(7,n-1)for n in shape),
                 (min(4,shape[0]-1),min(5,shape[1]-1),min(6,shape[2]-1))]
        for mode in triples:
            axis=[torch.cos(math.pi*k/n*(torch.arange(n,dtype=torch.float64)+.5))for n,k in zip(shape,mode)]
            field=axis[0][:,None,None]*axis[1][None,:,None]*axis[2][None,None,:]
            eigenvalue=sum(2*(1-math.cos(math.pi*k/n))for n,k in zip(shape,mode))
            pressure=(field/eigenvalue).reshape(1,1,*shape);pressure=pressure-pressure[:,:,:1,:1,:1].clone()
            for cells_per_metre in(8,16):
                h=torch.tensor([1/cells_per_metre],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
                actual=numerics.fine_action(pressure,h,rho)
                expected=field.reshape(1,1,*shape)/(1000*cells_per_metre)
                error=float((actual-expected).norm()/expected.norm())
                if error>1e-9:raise ValueError('Eight-mode offline inverse label fails actual native-equivalent stencil')
                checks.append(dict(shape_zyx=shape,mode_zyx=mode,cells_per_metre=cells_per_metre,operator_relative_error=error))
    write(args.output,dict(status='passed',teacher_manifest_sha256=sha(bank/'manifest.json'),teacher_archive_sha256=sha(bank/'teacher.npz'),
        label_max_relative_conversion=relative,modal_checks=checks,optimizer_updates=0,source_sha256=sha(__file__),
        numerical_inverse_only_offline=True,physical_reference_fields_used=False,reserved_scenes_consumed=False))
    print('Qualified eight-mode teacher against40actualstencil modal equations',flush=True)
