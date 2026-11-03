"""Verify offline modal labels against the actual pressure operator before any fit."""
import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from study_support import HERE,BASE,read,write,sha,load_checkpoint
import numerics

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh offline teacher qualification required')
    torch.set_num_threads(1);bank=BASE/'head-teacher-bank-01';manifest=read(bank/'manifest.json')
    if sha(bank/'teacher.npz')!=manifest['archive_sha256']:raise ValueError('Offline teacher archive changed')
    with np.load(bank/'teacher.npz',allow_pickle=False)as source:arrays={name:source[name].copy()for name in source.files}
    if set(arrays)!=set(manifest['inventory']):raise ValueError('Teacher field inventory differs')
    for name,value in arrays.items():
        spec=manifest['inventory'][name]
        if value.dtype.str!=spec['dtype']or list(value.shape)!=spec['shape']or hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Offline teacher array identity differs')
    counts=arrays['counts_zyx'];gains=arrays['teacher_gain'];caps=counts.max(1).astype('f8')**2/2
    if len(set(map(tuple,counts)))!=len(counts):raise ValueError('Teacher geometry split contains duplicates')
    if not np.array_equal(arrays['features'],np.log(counts.astype('f8')/16).astype('f4')):raise ValueError('Shape feature convention differs')
    relative=float(np.max(np.abs(np.exp(arrays['log_gain_fraction'].astype('f8'))-gains/caps[:,None])/(gains/caps[:,None])))
    if relative>1e-6:raise ValueError('Stored log labels exceed predeclared1e-6 relative conversion bound')
    rows=[]
    for shape in ((16,16,16),(11,11,9),(8,6,15),(9,13,48),(32,32,32)):
        for modes in ((1,0,0),(0,1,0),(0,0,1),(1,2,3),(3,3,3)):
            axis=[torch.cos(math.pi*k/n*(torch.arange(n,dtype=torch.float64)+.5))for n,k in zip(shape,modes)]
            field=axis[0][:,None,None]*axis[1][None,:,None]*axis[2][None,None,:]
            eigenvalue=sum(2*(1-math.cos(math.pi*k/n))for n,k in zip(shape,modes))
            pressure=(field/eigenvalue).reshape(1,1,*shape);pressure=pressure-pressure[:,:,:1,:1,:1].clone()
            action=numerics.fine_action(pressure,torch.tensor([1/16],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))
            expected=field.reshape(1,1,*shape)/16000
            error=float(torch.linalg.vector_norm(action-expected)/torch.linalg.vector_norm(expected))
            if error>1e-9:raise ValueError('Analytical offline label does not solve the actual discrete modal equation')
            rows.append(dict(shape_zyx=shape,mode_zyx=modes,operator_relative_error=error))
    state=load_checkpoint('C107')
    with torch.inference_mode():
        features=torch.from_numpy(arrays['features']);logits=state.network.coarse_logits+state.network.shape_head(features)
        initial_log_fraction=torch.nn.functional.logsigmoid(logits).numpy()
    summaries=[]
    for small in (True,False):
        for phase in (0,1):
            selection=(arrays['tier_small']==int(small))&(arrays['split']==phase)
            errors=initial_log_fraction[selection].astype('f8')-arrays['log_gain_fraction'][selection]
            summaries.append(dict(tier='small'if small else'large',split='train'if phase==0 else'development',geometries=int(selection.sum()),
                initial_C107_log_gain_rmse=float(np.sqrt(np.mean(errors**2))),
                initial_C107_median_relative_gain_error=float(np.median(np.abs(np.expm1(errors)))),
                initial_C107_maximum_relative_gain_error=float(np.max(np.abs(np.expm1(errors))))))
    write(args.output,dict(status='passed',source_sha256=sha(__file__),teacher_manifest_sha256=sha(bank/'manifest.json'),
        teacher_archive_sha256=sha(bank/'teacher.npz'),label_relative_conversion_bound=1e-6,
        measured_label_relative_conversion_error=relative,operator_relative_tolerance=1e-9,modal_checks=rows,
        initial_C107_teacher_diagnostics=summaries,optimizer_updates=0,
        control_or_teacher_only=True,candidate_numerical_pressure_solve=False,reserved_scenes_consumed=False))
    print('Qualified offline teacher labels against25actual discrete modal equations; zero fitting updates',flush=True)
