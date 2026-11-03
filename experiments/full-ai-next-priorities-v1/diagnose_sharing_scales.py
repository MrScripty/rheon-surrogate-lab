"""Training-only increment scale and irrelevant-tracer sensitivity diagnosis.

The cold-start native pressure-gradient reconstruction is an offline teacher
control. It never changes a candidate, trains weights or joins a frontier.
"""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from sharing_evaluation_support import load_model
from temporal_features_v2 import packed_state,auxiliary
from temporal_objective import errors
from study_support import ROOT,HERE,BASE,read,write,sha


def rms(value):return float(value.double().square().mean().sqrt())


def run(args):
    if args.output.exists():raise ValueError('Fresh training scale diagnosis required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    network,norm,identity=load_model(args.pair.resolve());k=network.horizons
    bank=read(BASE/'temporal-data-01/bank.json');rows=[];cold=[]
    for row in bank['trajectories']:
        if row['split']!='train':continue
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Training scene changed')
        receipt=read(path);scene=receipt['scene'];nx,ny,nz=scene['counts']
        def state(step):
            spec=receipt['states'][step];source=path.parent/spec['path']
            if sha(source)!=spec['sha256']:raise ValueError('Training state changed')
            with np.load(source,allow_pickle=False)as arrays:
                return {name:arrays[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
        shapes=((nz,ny,nx),(nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz,ny,nx))
        slices=tuple((slice(0,a),slice(0,b),slice(0,c))for a,b,c in shapes)
        for start in (0,8,63,120):
            current=state(start);encoded=packed_state(current,scene['counts'],norm['scales'])
            feature=auxiliary(scene,start,k,norm['scales'])
            target=torch.stack([packed_state(state(start+h),scene['counts'],norm['scales'])for h in range(1,k+1)],dim=1)
            with torch.no_grad():
                prediction=network(encoded,feature)
                no_tracer=encoded.clone();no_tracer[:,4]=0
                ablated=network(no_tracer,feature)
                no_force=feature.clone();no_force[:,7:]=0
                unforced=network(encoded,no_force)
            variable=prediction.detach().requires_grad_(True)
            losses=errors(variable,target,scene['counts'])
            field_gradient={}
            for name in ('pressure','equation','velocity','tracer'):
                objective=torch.log1p(losses[name]/1e-6).mean()
                gradient=torch.autograd.grad(objective,variable,retain_graph=True)[0]
                field_gradient[name]=float(gradient.norm())
            fields={}
            for index,name in enumerate(('pressure','vx','vy','vz','tracer')):
                selection=(slice(None),slice(None),index,*slices[index])
                truth=target[selection];predicted=prediction[selection]
                ablated_field=ablated[selection];unforced_field=unforced[selection]
                if index==0:
                    truth=truth-truth.mean(dim=(-3,-2,-1),keepdim=True)
                    predicted=predicted-predicted.mean(dim=(-3,-2,-1),keepdim=True)
                    ablated_field=ablated_field-ablated_field.mean(dim=(-3,-2,-1),keepdim=True)
                    unforced_field=unforced_field-unforced_field.mean(dim=(-3,-2,-1),keepdim=True)
                base=encoded[:,None,index][(slice(None),slice(None),*slices[index])]
                delta=truth if index==0 else truth-base
                learned=predicted if index==0 else predicted-base
                fields[name]=dict(normalized_target_rms=rms(truth),normalized_increment_rms=rms(delta),
                    predicted_increment_rms=rms(learned),
                    prediction_relative_l2=float((predicted-truth).double().norm()/truth.double().norm().clamp_min(1e-30)),
                    tracer_ablation_change_relative_to_truth=float((predicted-ablated_field).double().norm()/truth.double().norm().clamp_min(1e-30)),
                    force_ablation_change_relative_to_truth=float((predicted-unforced_field).double().norm()/truth.double().norm().clamp_min(1e-30)))
            rows.append(dict(identifier=row['identifier'],start=start,fields=fields,
                input_rms=dict(current_velocity=rms(encoded[:,1:4]),tracer=rms(encoded[:,4]),known_force=rms(feature[:,7:])),
                output_gradient_norm_by_log_loss=field_gradient))
            if start==0:
                future=state(1);p=future['pressure'].astype('f8').reshape(nz,ny,nx)
                forces=feature[:,7:10]*norm['scales']['velocity'];predicted=[];reference=[]
                for component,(name,shape)in enumerate(zip(('vx','vy','vz'),shapes[1:4])):
                    value=forces[0,component,:shape[0],:shape[1],:shape[2]].double().numpy().copy()
                    interior=[slice(None)]*3;interior[2-component]=slice(1,-1)
                    value[tuple(interior)]-=scene['cells_per_metre']/(60*1000)*np.diff(p,axis=2-component)
                    predicted.append(value.ravel());reference.append(future[name].astype('f8').ravel())
                actual=np.concatenate(reference);control=np.concatenate(predicted)
                cold.append(dict(identifier=row['identifier'],
                    native_pressure_offline_velocity_relative_l2=float(np.linalg.norm(control-actual)/max(np.linalg.norm(actual),1e-30)),
                    reference_velocity_rms=float(np.sqrt(np.mean(actual**2))),
                    reference_pressure_rms=float(np.sqrt(np.mean((p-p.mean())**2)))))
    summaries=[]
    for phase in ('cold','warm'):
        selected=[row for row in rows if(row['start']==0)==(phase=='cold')]
        summaries.append(dict(phase=phase,queries=len(selected),
            median_input_rms={name:statistics.median(row['input_rms'][name]for row in selected)for name in selected[0]['input_rms']},
            fields={name:{key:statistics.median(row['fields'][name][key]for row in selected)
                          for key in selected[0]['fields'][name]}for name in selected[0]['fields']},
            median_output_gradient_norm={name:statistics.median(row['output_gradient_norm_by_log_loss'][name]for row in selected)
                                         for name in selected[0]['output_gradient_norm_by_log_loss']}))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,cold_offline_teacher_control=cold,
        source_sha256=sha(__file__),training_bank_sha256=sha(BASE/'temporal-data-01/bank.json'),
        training_scenes_only=True,pressure_metrics_mean_free=True,
        supersedes_pressure_gauge_diagnostics='temporal-sharing-scale-diagnosis-01.json; its raw pressure diagnostic mixed cell-zero and mean-free gauges, all other diagnostics unchanged',optimizer_updates=0,reference_used_only_for_metrics=True,
        offline_control_excluded_from_candidate_frontier=True,reserved_scenes_consumed=False))
    print('Diagnosed training increment scale, pressure/velocity tracer sensitivity and native cold-start teacher control',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
