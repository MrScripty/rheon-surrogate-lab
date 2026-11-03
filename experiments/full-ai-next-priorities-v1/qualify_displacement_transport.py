"""Independent native qualification of training labels and the fixed readout."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from displacement_readout import convex_field_readout
from training_displacement_teacher import displacement_labels
from transport_teacher_support import carrier,native,packed_native_points,point_mask
from temporal_features_v2 import geometry
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh native transport qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=DOCS/'displacement-transport-qualification-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered transport source changed')
    cases=[];sources={};rng=np.random.Generator(np.random.PCG64(2026102031))
    for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
        if row['split']!='train':continue
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Native training receipt changed')
        receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=sha(path)
        for start in reg['starts']:
            states=[]
            for step in (start,start+1):
                spec=receipt['states'][step];source=path.parent/spec['path']
                if sha(source)!=spec['sha256']:raise ValueError('Native field changed')
                sources[str(source.relative_to(ROOT))]=sha(source)
                with np.load(source,allow_pickle=False)as arrays:states.append({n:arrays[n].copy()for n in ('vx','vy','vz','tracer')})
            cases.append((states[0],scene,dict(kind='captured',identifier=row['identifier'],start=start),states[1]['tracer']))
    for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
        nx,ny,nz=counts
        for cpm in (8,16):
            for pattern in ('zero','random'):
                fields=dict(vx=(.01*rng.normal(size=(nz,ny,nx+1))).astype('f4'),
                    vy=(.01*rng.normal(size=(nz,ny+1,nx))).astype('f4'),
                    vz=(.01*rng.normal(size=(nz+1,ny,nx))).astype('f4'),tracer=rng.random((nz,ny,nx)).astype('f4'))
                fields['vx'][:,:,[0,-1]]=0;fields['vy'][:,[0,-1],:]=0;fields['vz'][[0,-1],:,:]=0
                if pattern=='zero':
                    for name in ('vx','vy','vz'):fields[name].fill(0)
                fields={n:v.reshape(-1)for n,v in fields.items()}
                cases.append((fields,dict(counts=counts,cells_per_metre=cpm),dict(kind='fixture',pattern=pattern),None))
    rows=[]
    for start in range(0,len(cases),2):
        selected=cases[start:start+2];packets=[]
        for fields,scene,identity,target in selected:
            packets.append(dict(counts=scene['counts'],cell_size_m=1/scene['cells_per_metre'],dt_s=1/60,
                velocity=[fields[n].astype('f8').tolist()for n in ('vx','vy','vz')],tracer=fields['tracer'].astype('f8').tolist()))
        references=native(packets)
        for (fields,scene,identity,target),reference in zip(selected,references):
            counts=scene['counts'];nx,ny,nz=counts;input_tensor=carrier(fields,counts).to(args.device)
            shifts,points=displacement_labels(input_tensor[:,:3],counts,scene['cells_per_metre']/60)
            native_shifts=packed_native_points(reference,counts,'offsets_cells').to(args.device)
            native_points=packed_native_points(reference,counts,'velocity_samples').to(args.device)
            shift_error=float((shifts-native_shifts).abs().max());point_error=float((points-native_points).norm())/max(float(native_points.norm()),1e-30)
            if shift_error>reg['maximum_displacement_absolute_error_cells']or point_error>reg['point_velocity_relative_tolerance']:
                raise ValueError(f'Training-only label mismatch: shift={shift_error} point={point_error}')
            with torch.no_grad():
                predicted=convex_field_readout(input_tensor,native_shifts)
                predicted[:,:3]*=geometry(tuple(counts),args.device)[1][1:4]
                actuals=[predicted[0,0,:nz,:ny,:nx+1],predicted[0,1,:nz,:ny+1,:nx],predicted[0,2,:nz+1,:ny,:nx],predicted[0,3,:nz,:ny,:nx]]
                expected=[*reference['advected_velocity'],reference['advected_tracer']];differences={}
                for name,a,b in zip(('vx','vy','vz','tracer'),actuals,expected):
                    truth=torch.tensor(b,dtype=torch.float32,device=args.device).reshape_as(a)
                    error=float((a-truth).double().norm())/max(float(truth.double().norm()),1e-30)
                    if error>reg['native_readout_relative_tolerance']:raise ValueError('Native velocity/tracer readout mismatch')
                    differences[name]=error
                if target is not None and not np.array_equal(np.asarray(reference['advected_tracer'],dtype='f4').view('u4'),target.view('u4')):
                    raise ValueError('Native tracer staging changed')
                zero=input_tensor.clone();zero[:,3]=0
                constant=input_tensor.clone();constant[:,3]=.25
                if bool((convex_field_readout(zero,native_shifts)[:,3]!=0).any()):raise ValueError('Zero tracer changed')
                if not bool((convex_field_readout(constant,native_shifts)[:,3]==.25).all()):raise ValueError('Constant tracer changed')
            gradients=[]
            for repeat in range(2):
                value=input_tensor.detach().clone().requires_grad_(True);offset=native_shifts.detach().clone().requires_grad_(True)
                convex_field_readout(value,offset).square().mean().backward()
                if not bool(torch.isfinite(value.grad).all())or not bool(torch.isfinite(offset.grad).all()):raise ValueError('Readout gradients nonfinite')
                gradients.append((value.grad.detach().cpu(),offset.grad.detach().cpu()))
            if any(not torch.equal(a,b)for a,b in zip(*gradients)):raise ValueError('Readout field/shift gradients not repeatable')
            rows.append(dict(**identity,counts_xyz=counts,cells_per_metre=scene['cells_per_metre'],
                displacement_absolute_error_cells=shift_error,point_velocity_relative_error=point_error,
                native_readout_relative_error=differences,deterministic_field_and_shift_gradients=True,zero_and_constant_tracer_exact=True))
        print('Qualified transport',args.device,min(start+2,len(cases)),'of',len(cases),flush=True)
    write(args.output,dict(status='passed',device=args.device,rows=rows,source_sha256=reg['source_sha256'],source_evidence=sources,
        protocol_sha256=sha(protocol),native_build_receipt_sha256=sha(BASE/'transport-teacher-build-01/build-receipt.json'),
        maximum_displacement_absolute_error_cells=max(r['displacement_absolute_error_cells']for r in rows),
        maximum_point_velocity_relative_error=max(r['point_velocity_relative_error']for r in rows),
        maximum_native_readout_relative_error=max(v for r in rows for v in r['native_readout_relative_error'].values()),
        native_pressure_solves=0,native_steps_executed=0,optimizer_updates=0,
        teacher_labels_for_training_only=True,candidate_predictor_implemented=False,candidate_frontier_point=False,reserved_scenes_consumed=False))
    print('Qualified full face/cell label and fixed-readout path',args.device,len(rows),'cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
