"""Train-only offline pressure/decoder ceilings; never a candidate prediction."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from pressure_expert_evaluation_support import load_model
from temporal_train_pressure_expert import inventory
from temporal_features_v2 import packed_state,auxiliary,geometry
from temporal_metrics import metrics,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh factorization diagnosis required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    protocol=DOCS/'temporal-pressure-expert-diagnostic-protocol-01.json';read(protocol)
    tracer_protocol=DOCS/'temporal-pressure-expert-tracer-probe-protocol-01.json';tracer_registration=read(tracer_protocol)
    if sha(ROOT/tracer_registration['native_source'])!=tracer_registration['native_source_sha256']:
        raise ValueError('Native passive-tracer contract changed')
    network,norm,identity=load_model(args.pair.resolve());before=inventory(network);k=network.horizons
    bank=read(BASE/'temporal-data-01/bank.json');rows=[];readouts=[]
    with torch.inference_mode():
        for row in bank['trajectories']:
            if row['split']!='train':continue
            receipt_path=ROOT/row['receipt']
            if sha(receipt_path)!=row['receipt_sha256']:raise ValueError('Training receipt changed')
            receipt=read(receipt_path);scene=receipt['scene'];counts=scene['counts'];nx,ny,nz=counts
            def state(step):
                record=receipt['states'][step];path=receipt_path.parent/record['path']
                if sha(path)!=record['sha256']:raise ValueError('Training reference changed')
                with np.load(path,allow_pickle=False)as arrays:return {name:arrays[name].copy()for name in (*FIELDS,'rhs')}
            for start in (0,8,63,120):
                current=state(start);physical={name:current[name]for name in FIELDS}
                targets=[state(start+h)for h in range(1,k+1)]
                normal=network.predict(physical,scene,start,norm)
                zero_tracer=network.predict(dict(physical,tracer=np.zeros_like(physical['tracer'])),scene,start,norm)
                anchor=targets[-1]if network.architecture=='endpoint'else targets[0]
                reference_pressure=torch.from_numpy(anchor['pressure'].copy()).reshape(1,1,nz,ny,nx).double()
                reference_pressure=reference_pressure-reference_pressure[:,:,:1,:1,:1]
                def oracle_pressure(module,arguments,output):return reference_pressure.to(output.device,output.dtype)
                observed=[]
                def velocity_readout(module,arguments,output):observed.append(output.detach().clone())
                hooks=[network.pressure_core.register_forward_hook(oracle_pressure),network.pressure_velocity.register_forward_hook(velocity_readout)]
                try:oracle=network.predict(physical,scene,start,norm)
                finally:
                    for hook in hooks:hook.remove()
                def gauge_pressure(module,arguments,output):return output+norm['scales']['pressure']
                hook=network.pressure_core.register_forward_hook(gauge_pressure)
                try:gauge=network.predict(physical,scene,start,norm)
                finally:hook.remove()
                for horizon,(actual,ceiling,shifted,target)in enumerate(zip(normal,oracle,gauge,targets),1):
                    normal_metrics=metrics(actual,target,target['rhs'],scene,norm,0.)
                    oracle_metrics=metrics(ceiling,target,target['rhs'],scene,norm,0.)
                    velocity_difference=sum(float((shifted[name]-actual[name]).double().square().sum())for name in ('vx','vy','vz'))**.5
                    target_norm=sum(float(np.square(target[name].astype('f8')).sum())for name in ('vx','vy','vz'))**.5
                    rows.append(dict(identifier=row['identifier'],start=start,horizon=horizon,counts_xyz=counts,
                        normal=normal_metrics['relative'],oracle_anchor_pressure=oracle_metrics['relative'],
                        pressure_gauge_shift_velocity_relative_change=velocity_difference/max(target_norm,1e-30),
                        zero_current_tracer_predicted_absolute_rms=float(zero_tracer[horizon-1]['tracer'].double().square().mean().sqrt())))
                current_tensor=packed_state(physical,tuple(counts),norm['scales'],'cpu')
                force=auxiliary(scene,start,k,norm['scales'],'cpu')[:,7:].reshape(1,k,3,*current_tensor.shape[2:])
                _,mask=geometry(tuple(counts),'cpu');h=1/scene['cells_per_metre']
                units=(1/60/(1000*h)*norm['scales']['pressure']/norm['scales']['velocity'])
                horizons=[k,*range(1,k)]if network.architecture=='endpoint'else list(range(1,k+1))
                if len(observed)!=k:raise ValueError('Every pressure-velocity readout must be observed')
                for horizon,basis in zip(horizons,observed):
                    target=packed_state({name:targets[horizon-1][name]for name in FIELDS},tuple(counts),norm['scales'],'cpu')
                    desired=(target[:,1:4]-current_tensor[:,1:4]-force[:,:horizon].sum(1))*mask[1:4]
                    basis=basis*units*mask[1:4];den=float(basis.double().square().sum());truth=float(desired.double().square().sum())
                    cross=float((basis.double()*desired.double()).sum())
                    readouts.append(dict(identifier=row['identifier'],start=start,horizon=horizon,
                        diagnostic_least_squares_scalar=cross/den if den>0 else None,
                        cosine=cross/max((den*truth)**.5,1e-30),readout_norm=den**.5,target_correction_norm=truth**.5,
                        scalar_applied_to_candidate=False,advection_and_viscosity_remain_in_target_correction=True))
    if inventory(network)!=before:raise ValueError('Offline diagnostic modified candidate coefficients')
    summaries=[]
    for phase in ('cold','warm'):
        selected=[row for row in rows if(row['start']==0)==(phase=='cold')]
        for mode in ('normal','oracle_anchor_pressure'):
            for field in ('pressure','equation','velocity','tracer'):
                values=[row[mode][field]for row in selected]
                summaries.append(dict(phase=phase,mode=mode,field=field,outputs=len(values),median=statistics.median(values),worst=max(values)))
    scalar_summaries=[]
    for phase in ('cold','warm'):
        for horizon in range(1,k+1):
            selected=[row for row in readouts if(row['start']==0)==(phase=='cold')and row['horizon']==horizon]
            scalars=[row['diagnostic_least_squares_scalar']for row in selected if row['diagnostic_least_squares_scalar']is not None]
            scalar_summaries.append(dict(phase=phase,horizon=horizon,median_scalar=statistics.median(scalars),
                median_cosine=statistics.median(row['cosine']for row in selected)))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,readouts=readouts,
        lead_time_readout_summaries=scalar_summaries,protocol_sha256=sha(protocol),source_sha256=sha(__file__),
        zero_tracer_probe_protocol_sha256=sha(tracer_protocol),
        inference_source_sha256=sha(HERE/'bundle_model_pressure_expert.py'),optimizer_updates=0,
        training_queries_only=True,native_future_pressure_used_only_in_offline_oracle=True,
        candidate_frontier_point=False,CPU_latency_claim=False,reserved_scenes_consumed=False,
        field_criteria_and_full_state_objective_unchanged=True))
    print('Completed train-only offline pressure/decoder and lead-time diagnostics; candidate unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
