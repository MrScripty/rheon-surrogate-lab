"""Native teacher qualification of learned coefficient sharing and pressure sums.

Future native pressure is used only in the explicitly offline cold controls.
No candidate prediction, optimizer update or pressure solve occurs here.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np
import torch
from tied_momentum_readout import index_readout
from temporal_features_v2 import force_features
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def vectors(value,counts):
    nx,ny,nz=counts
    return [value[0,0,:nz,:ny,:nx+1].reshape(-1).numpy(),
            value[0,1,:nz,:ny+1,:nx].reshape(-1).numpy(),
            value[0,2,:nz+1,:ny,:nx].reshape(-1).numpy()]


def run(args):
    if args.output.exists():raise ValueError('Fresh tied-readout qualification required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    reg=read(DOCS/'tied-momentum-qualification-registration-01.json')
    if sha(HERE/'tied_momentum_readout.py')!=reg['readout_source_sha256']:raise ValueError('Registered neural readout changed')
    build=BASE/'momentum-teacher-build-02';receipt=read(build/'build-receipt.json');binary=build/'rheon-momentum-teacher-probe'
    if sha(binary)!=receipt['binary_sha256']or sha(HERE/'momentum_teacher_probe.rs')!=receipt['source_sha256']:
        raise ValueError('Independent native teacher changed')
    assets=BASE/'temporal-pressure-expert-assets-01';provenance=read(assets/'request-initialization.json')
    if sha(assets/'request-weights.npz')!=provenance['weights_sha256']:raise ValueError('Managed learned flux changed')
    with np.load(assets/'request-weights.npz',allow_pickle=False)as arrays:
        weight=torch.from_numpy(arrays['flux.weight'].copy())
    cases=[];neural=[];identities=[];cold=[]
    rng=np.random.Generator(np.random.PCG64(2026101998))
    def append(counts,cpm,p,identity):
        p=np.asarray(p,dtype='<f8').reshape(counts[::-1]);p=p-p.reshape(-1)[0]
        packet=dict(counts=counts,cell_size_m=1/cpm,dt_s=1/60,rho=1000.,pressure=p.reshape(-1).tolist())
        cases.append(packet);identities.append(identity)
        raw=index_readout(torch.from_numpy(p.copy()).reshape(1,1,*p.shape),weight)*(cpm/(60*1000))
        neural.append(vectors(raw,counts));return len(cases)-1
    with torch.no_grad():
        for counts in ([16,16,16],[9,11,11],[15,6,8],[32,32,32]):
            for cpm in (8,16):
                for pattern in range(2):
                    append(counts,cpm,rng.normal(size=counts[::-1]),dict(kind='random_pressure',counts_xyz=counts,cells_per_metre=cpm,pattern=pattern))
        for row in read(BASE/'temporal-data-01/bank.json')['trajectories']:
            if row['split']!='train':continue
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:raise ValueError('Training native receipt changed')
            captured=read(path);scene=captured['scene'];counts=scene['counts'];cpm=scene['cells_per_metre']
            forces=force_features(scene,0,4,reg['velocity_scale'],'cpu').reshape(4,3,*[n+1 for n in counts[::-1]])*reg['velocity_scale']
            accumulated=np.zeros(counts[::-1],dtype='f8')
            for horizon in (1,2,3,4):
                record=captured['states'][horizon];source=path.parent/record['path']
                if sha(source)!=record['sha256']:raise ValueError('Native field changed')
                with np.load(source,allow_pickle=False)as arrays:
                    p=arrays['pressure'].copy().reshape(counts[::-1]);truth=[arrays[n].astype('f8').reshape(-1)for n in ('vx','vy','vz')]
                accumulated+=p
                prefix=append(counts,cpm,accumulated,dict(kind='cold_pressure_prefix',identifier=row['identifier'],horizon=horizon))
                endpoint=append(counts,cpm,horizon*p,dict(kind='cold_last_pressure_times_horizon',identifier=row['identifier'],horizon=horizon))
                known=vectors(forces[:horizon].double().sum(0,keepdim=True),counts)
                reference=np.concatenate(truth);den=max(float(np.linalg.norm(reference)),1e-30)
                errors={}
                for label,index in (('pressure_prefix',prefix),('last_pressure_times_horizon',endpoint)):
                    predicted=np.concatenate([f+v for f,v in zip(known,neural[index])])
                    errors[label]=float(np.linalg.norm(predicted-reference))/den
                cold.append(dict(identifier=row['identifier'],counts_xyz=counts,cells_per_metre=cpm,horizon=horizon,
                    relative_velocity_error=errors,nonlinear_transport_and_viscosity_omitted=True,reference_future_pressure=True))
    qualification=[]
    for start in range(0,len(cases),16):
        packet=json.dumps(dict(cases=cases[start:start+16]),allow_nan=False).encode()
        completed=subprocess.run([str(binary)],input=packet,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
        outputs=json.loads(completed.stdout)['cases']
        if len(outputs)!=len(cases[start:start+16]):raise ValueError('Native teacher omitted a case')
        for offset,output in enumerate(outputs):
            index=start+offset
            if output['pressure_solves']!=0 or output['native_steps_executed']!=0:raise ValueError('Primitive teacher scope changed')
            target=np.concatenate([np.asarray(v,dtype='<f4').astype('f8')for v in output['velocity']])
            actual=np.concatenate(neural[index]);relative=float(np.linalg.norm(actual-target))/max(float(np.linalg.norm(target)),1e-30)
            if relative>reg['native_relative_difference_tolerance']:raise ValueError(f'Tied learned readout differs from native teacher: {relative}')
            qualification.append(dict(**identities[index],native_relative_difference=relative))
    summaries=[]
    for horizon in (1,2,3,4):
        selected=[row for row in cold if row['horizon']==horizon]
        summaries.append(dict(horizon=horizon,queries=len(selected),
            **{label:dict(median=statistics.median(row['relative_velocity_error'][label]for row in selected),
                worst=max(row['relative_velocity_error'][label]for row in selected))for label in ('pressure_prefix','last_pressure_times_horizon')}))
    write(args.output,dict(status='passed_offline_learned_readout_qualification',qualification=qualification,cold_controls=cold,summaries=summaries,
        registration_sha256=sha(DOCS/'tied-momentum-qualification-registration-01.json'),
        source_sha256={n:sha(HERE/n)for n in('qualify_tied_momentum.py','tied_momentum_readout.py','momentum_teacher_probe.rs','build_momentum_teacher.py')},
        native_build_receipt_sha256=sha(build/'build-receipt.json'),learned_weight_provenance=provenance,
        primitive_maximum_native_relative_difference=max(row['native_relative_difference']for row in qualification),
        new_optimizer_updates=0,pressure_solves=0,native_steps_executed=0,
        candidate_inference_implemented=False,temporal_frontier_point=False,CPU_latency_claim=False,
        future_reference_pressure_only_offline=True,reserved_scenes_consumed=False))
    print('Qualified neural momentum coefficients derived from learned flux:',len(qualification),'native primitive comparisons',flush=True)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
