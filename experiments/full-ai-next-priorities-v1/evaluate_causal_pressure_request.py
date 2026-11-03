"""Causal predicted-request pressure evaluation; reference future data is metrics only."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint
from pressure_request_inference import RequestPrediction
from precise_routed_tail_pressure import PreciseRoutedTailPrediction


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh causal pressure component evaluation required')
    args.output.mkdir(parents=True);os.sched_setaffinity(0,{23})
    component=RequestPrediction(args.entry);core=load_checkpoint('a-tail8');predict=PreciseRoutedTailPrediction(core)
    sources={};rows=[];probe_parts=[];traces=[]
    banks=[(BASE/'temporal-data-01/bank.json','train'),(BASE/'temporal-development-240-01/bank.json','development')]
    with torch.inference_mode():
        for bank_path,split in banks:
            bank=read(bank_path);sources[str(bank_path.relative_to(ROOT))]=sha(bank_path)
            selected=[row for row in bank['trajectories']if row['split']==split]
            if len(selected)!=(16 if split=='train'else 8):raise ValueError('All declared train/development scenes required')
            for parent in selected:
                path=ROOT/parent['receipt']
                if sha(path)!=parent['receipt_sha256']:raise ValueError('Native scene receipt changed')
                receipt=read(path);scene=receipt['scene'];counts=scene['counts'];shape=(1,1,*counts[::-1])
                sources[str(path.relative_to(ROOT))]=sha(path)
                starts=(0,8,63,120)if split=='train'else range(240)
                per_scene=[]
                for start in starts:
                    current=receipt['states'][start];current_path=path.parent/current['path']
                    if current['step']!=start or sha(current_path)!=current['sha256']:raise ValueError('Current native state changed')
                    with np.load(current_path,allow_pickle=False)as arrays:
                        fields={name:arrays[name].copy()for name in ('vx','vy','vz')}
                    # Prediction completes before the future reference file is opened.
                    normalized=component.predict_normalized(fields,scene,start)
                    q=normalized[:,0].double()*component.norm['scales']['q']
                    h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64)
                    rho=torch.tensor([1000.],dtype=torch.float64);dt=torch.tensor([1/60],dtype=torch.float64)
                    request=dict(base=torch.zeros(shape,dtype=torch.float64),native_input_residual=q*h.reshape(-1,1,1,1,1)/rho.reshape(-1,1,1,1,1),
                        cell_size_m=h,rho=rho,dt=dt)
                    pressure=predict(request)
                    future=receipt['states'][start+1];future_path=path.parent/future['path']
                    if future['step']!=start+1 or sha(future_path)!=future['sha256']:raise ValueError('Future metric reference changed')
                    with np.load(future_path,allow_pickle=False)as arrays:
                        rhs=torch.from_numpy(arrays['rhs'].copy().reshape(shape))
                        reference=torch.from_numpy(arrays['pressure'].copy().reshape(shape))
                    true_q=rhs*(1000.*scene['cells_per_metre']);reference=reference-reference.mean()
                    equation=core.model.full_operator(pressure,h,rho)-rhs
                    time_band='startup'if start==0 else'early'if start<=8 else'in_training_time'if start<=120 else'later_time'
                    row=dict(identifier=parent['identifier'],split=split,counts_xyz=counts,cells_per_metre=scene['cells_per_metre'],
                        start=start,horizon=1,step=start+1,time_band=time_band,
                        request_relative_l2=float((q-true_q).norm())/max(float(true_q.norm()),1e-30),
                        equation_relative_residual=float(equation.norm())/max(float(rhs.norm()),1e-30),
                        pressure_relative_l2=float((pressure-pressure.mean()-reference).norm())/max(float(reference.norm()),1e-30))
                    rows.append(row);per_scene.append(row)
                scene_output=args.output/(parent['identifier']+'.json')
                write(scene_output,dict(status='completed',rows=per_scene,scene=scene,reference_receipt_sha256=sha(path),
                    inference_inputs='Current velocities and known forcing only; future reference files opened after prediction',
                    pressure_input_keys=sorted(request),component_only=True,own_history_rollout=False))
                traces.append(dict(path=str(scene_output.relative_to(ROOT)),sha256=sha(scene_output)))
                print('Evaluated causal pressure component',parent['identifier'],len(per_scene),'windows',flush=True)
    if component.inventory!=component.helpers.inventory(component.network):raise ValueError('Evaluation changed component weights')
    groups=[]
    for split in ('train','development'):
        for regime in ('small','32cubed'):
            for band in ('all','startup','early','in_training_time','later_time'):
                selected=[row for row in rows if row['split']==split and(max(row['counts_xyz'])<=16)==(regime=='small')and(band=='all'or row['time_band']==band)]
                if not selected:continue
                groups.append(dict(split=split,regime=regime,time_band=band,queries=len(selected),
                    median_request_error=statistics.median(row['request_relative_l2']for row in selected),
                    worst_request_error=max(row['request_relative_l2']for row in selected),
                    median_pressure_error=statistics.median(row['pressure_relative_l2']for row in selected),
                    worst_pressure_error=max(row['pressure_relative_l2']for row in selected),
                    worst_equation_error=max(row['equation_relative_residual']for row in selected),
                    component_pressure_equation_4e_3_pass=all(max(row['pressure_relative_l2'],row['equation_relative_residual'])<=.004 for row in selected)))
    write(args.output/'summary.json',dict(status='completed',groups=groups,component_entry_sha256=sha(args.entry),
        pressure_core_entry=core.entry,source_sha256=sha(__file__),inference_source_sha256=sha(HERE/'pressure_request_inference.py'),
        pressure_execution_sha256=sha(HERE/'precise_routed_tail_pressure.py'),sources=sources,traces=traces,
        optimizer_updates=0,component_only=True,full_state_temporal_candidate=False,own_history_rollout=False,
        reserved_scenes_consumed=False,candidate_numerical_pressure_solve=False,model_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
