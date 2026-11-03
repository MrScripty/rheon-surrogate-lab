"""Offline scalar-transport replay with reference velocity; never a forecast."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from offline_tracer_readout import readout,native
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def cell_velocity(fields,counts):
    nx,ny,nz=counts
    vx=torch.as_tensor(fields['vx']).double().reshape(nz,ny,nx+1)
    vy=torch.as_tensor(fields['vy']).double().reshape(nz,ny+1,nx)
    vz=torch.as_tensor(fields['vz']).double().reshape(nz+1,ny,nx)
    return torch.stack(((vx[:,:,:-1]+vx[:,:,1:])/2,(vy[:,:-1]+vy[:,1:])/2,(vz[:-1]+vz[1:])/2),0).unsqueeze(0)


def run(args):
    if args.output.exists():raise ValueError('Fresh reference-velocity replay required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    protocol=DOCS/'tracer-reference-velocity-replay-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered replay/readout source changed')
    rows=[];qualifications=[];sources={}
    with torch.inference_mode():
        for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
            if row['split']!='train':continue
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(path);scene=receipt['scene'];counts=scene['counts'];shape=(1,1,*counts[::-1]);states=[]
            sources[str(path.relative_to(ROOT))]=sha(path)
            for spec in receipt['states']:
                source=path.parent/spec['path']
                if sha(source)!=spec['sha256']:raise ValueError('Reference state changed')
                sources[str(source.relative_to(ROOT))]=sha(source)
                with np.load(source,allow_pickle=False)as arrays:states.append({n:arrays[n].copy()for n in ('tracer','vx','vy','vz')})
            pending=[]
            for start in (0,8,63,120):
                fields=states[start]
                packet=dict(counts=counts,cell_size_m=1/scene['cells_per_metre'],dt_s=1/60,
                    velocity=[fields[n].astype('f8').tolist()for n in ('vx','vy','vz')],tracers=[fields['tracer'].astype('f8').tolist()])
                pending.append(packet)
            for start,observed in zip((0,8,63,120),native(pending)):
                expected=torch.tensor(observed['center_velocity'],dtype=torch.float64).reshape(1,3,*counts[::-1])
                actual=cell_velocity(states[start],counts)
                error=float((actual-expected).norm())/max(float(expected.norm()),1e-30)
                if error>reg['center_velocity_native_relative_tolerance']:raise ValueError('Offline center interpolation differs from native sampler')
                qualifications.append(dict(identifier=row['identifier'],start=start,relative_difference=error))
            current=torch.from_numpy(states[0]['tracer']).reshape(shape)
            for start in range(128):
                shift=-cell_velocity(states[start],counts)*(scene['cells_per_metre']/60)
                if float(shift.abs().max())>reg['maximum_CFL']:raise ValueError('Subcell replay scope exceeded')
                current=readout(current,shift)
                truth=torch.from_numpy(states[start+1]['tracer']).reshape(shape)
                error=float((current-truth).double().norm())/max(float(truth.double().norm()),1e-30)
                rows.append(dict(identifier=row['identifier'],counts_xyz=counts,step=start+1,relative_tracer_error=error))
            print('Replayed scalar control with native velocity',row['identifier'],'128steps',flush=True)
    summaries=[]
    for regime in ('small','32cubed'):
        for interval in (12,48,128):
            selected=[r for r in rows if(max(r['counts_xyz'])<=16)==(regime=='small')and r['step']<=interval]
            summaries.append(dict(regime=regime,interval_steps=interval,outputs=len(selected),
                worst_relative_tracer_error=max(r['relative_tracer_error']for r in selected)))
    write(args.output,dict(status='completed_offline_reference_velocity_control',rows=rows,summaries=summaries,
        center_velocity_qualification=qualifications,maximum_center_velocity_native_relative_error=max(r['relative_difference']for r in qualifications),
        protocol_sha256=sha(protocol),source_sha256=reg['source_sha256'],source_evidence=sources,
        reference_velocity_supplied_at_every_step=True,scalar_own_history_steps=128,full_state_prediction=False,
        candidate_inference_implemented=False,candidate_frontier_point=False,CPU_latency_claim=False,new_optimizer_updates=0,
        native_pressure_solves=0,native_steps_executed=0,native_tracer_primitive_calls=64,reserved_scenes_consumed=False,
        original240step_requirement_unchanged=True))
    print('Completed reference-velocity scalar transport replay;2048outputs; not a full-state model',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
