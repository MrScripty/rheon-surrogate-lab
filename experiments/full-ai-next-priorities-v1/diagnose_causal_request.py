"""Offline linear-response controls and learned-branch diagnosis, never candidates."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from torch.nn import functional as F
from study_support import ROOT,HERE,BASE,read,write,sha,load_checkpoint
from pressure_request_inference import RequestPrediction
from temporal_features_v2 import force_features
from precise_routed_tail_pressure import PreciseRoutedTailPrediction


def run(args):
    if args.output.exists():raise ValueError('Fresh causal fitting diagnosis required')
    os.sched_setaffinity(0,{23});component=RequestPrediction(args.entry);core=load_checkpoint('a-tail8')
    predict=PreciseRoutedTailPrediction(core);rows=[];norm=component.norm['scales'];network=component.network
    with torch.inference_mode():
        for bank_path,split in ((BASE/'temporal-data-01/bank.json','train'),(BASE/'temporal-development-240-01/bank.json','development')):
            for parent in read(bank_path)['trajectories']:
                if parent['split']!=split:continue
                path=ROOT/parent['receipt']
                if sha(path)!=parent['receipt_sha256']:raise ValueError('Native diagnostic receipt changed')
                receipt=read(path);scene=receipt['scene'];nx,ny,nz=scene['counts'];shape=(1,1,nz,ny,nx)
                starts=(0,8,63,120)if split=='train'else(0,8,63,120,200,239)
                for step in starts:
                    spec=receipt['states'][step];source=path.parent/spec['path']
                    if sha(source)!=spec['sha256']:raise ValueError('Current state changed')
                    with np.load(source,allow_pickle=False)as arrays:fields={name:arrays[name].copy()for name in ('vx','vy','vz')}
                    physical=[]
                    for name,grid in zip(('vx','vy','vz'),((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx))):
                        value=torch.from_numpy(fields[name]).reshape(1,1,*grid)
                        physical.append(F.pad(value,(0,nx+1-grid[2],0,ny+1-grid[1],0,nz+1-grid[0]),mode='replicate'))
                    physical=torch.cat(physical,dim=1);velocity=physical/norm['velocity']
                    known=force_features(scene,step,1,norm['velocity'],'cpu');values=torch.cat((velocity,known),dim=1)
                    h=1/scene['cells_per_metre'];units=1000*h/(1/60)*norm['velocity']
                    def center(value):
                        value=value[:,:,:nz,:ny,:nx].double()
                        return value-value.mean()
                    learned=component(fields,scene,step)[:,0]
                    left=network.left(values);right=network.right(values)
                    alpha=(1/60)/h*norm['velocity'];beta=.001/1000*(1/60)/h**2
                    linear=center(network.linear(values))*units
                    nonlinear=center(network.quadratic(torch.cat((left*right,left.abs()*right),dim=1)))*units*alpha
                    viscous=center(network.viscous(values))*units*beta
                    Fonly=torch.cat((torch.zeros_like(velocity),known),dim=1)
                    learned_force=center(network.linear(Fonly))*units
                    learned_velocity=linear-learned_force
                    def flux_q(value):
                        # Numerical forward operator is an offline control only.
                        result=torch.zeros(shape,dtype=torch.float64)
                        for d in range(3):
                            lower=value[:,d:d+1,:nz,:ny,:nx].double()
                            upper=(value[:,d:d+1,:nz,:ny,1:nx+1]if d==0 else
                                   value[:,d:d+1,:nz,1:ny+1,:nx]if d==1 else value[:,d:d+1,1:nz+1,:ny,:nx]).double()
                            result+=lower-upper
                        return result*(1000*h/(1/60))
                    force_q=flux_q((known*norm['velocity']).float())
                    current_plus_force=flux_q((physical.double()+known.double()*norm['velocity']).float())
                    future=receipt['states'][step+1];source=path.parent/future['path']
                    if sha(source)!=future['sha256']:raise ValueError('Future diagnostic reference changed')
                    with np.load(source,allow_pickle=False)as arrays:
                        rhs=torch.from_numpy(arrays['rhs'].copy().reshape(shape));reference=torch.from_numpy(arrays['pressure'].copy().reshape(shape))
                    truth=rhs*(1000/h);reference=reference-reference.mean();den=max(float(truth.norm()),1e-30)
                    branches={name:float(value.norm())/den for name,value in (
                        ('learned_linear_velocity',learned_velocity),('learned_linear_force',learned_force),
                        ('learned_nonlinear',nonlinear),('learned_viscous',viscous),
                        ('actual_nonlinear_plus_viscous_residual',truth-current_plus_force))}
                    controls=dict(learned=learned,force_only_numerical_control=force_q,
                        current_plus_force_numerical_control=current_plus_force,
                        exact_linear_with_learned_residual_control=current_plus_force+nonlinear+viscous,
                        true_future_request_oracle=truth)
                    metrics={}
                    for name,q in controls.items():
                        pressure=predict(dict(base=torch.zeros(shape,dtype=torch.float64),native_input_residual=q*(h/1000),
                            cell_size_m=torch.tensor([h],dtype=torch.float64),rho=torch.tensor([1000.],dtype=torch.float64),
                            dt=torch.tensor([1/60],dtype=torch.float64)))
                        error=core.model.full_operator(pressure,torch.tensor([h],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))-rhs
                        metrics[name]=dict(request_relative_l2=float((q-truth).norm())/den,
                            pressure_relative_l2=float((pressure-pressure.mean()-reference).norm())/max(float(reference.norm()),1e-30),
                            equation_relative_residual=float(error.norm())/max(float(rhs.norm()),1e-30))
                    rows.append(dict(identifier=parent['identifier'],split=split,counts_xyz=scene['counts'],step=step,
                        branches=branches,controls=metrics))
                print('Diagnosed causal request branches',parent['identifier'],flush=True)
    summaries=[]
    for split in ('train','development'):
        for regime in ('small','32cubed'):
            selected=[row for row in rows if row['split']==split and(max(row['counts_xyz'])<=16)==(regime=='small')]
            for name in rows[0]['controls']:
                values=[row['controls'][name]for row in selected]
                summaries.append(dict(split=split,regime=regime,control=name,queries=len(values),
                    median_request_error=statistics.median(value['request_relative_l2']for value in values),
                    worst_request_error=max(value['request_relative_l2']for value in values),
                    worst_pressure_error=max(value['pressure_relative_l2']for value in values),
                    worst_equation_error=max(value['equation_relative_residual']for value in values)))
    write(args.output,dict(status='completed',rows=rows,summaries=summaries,source_sha256=sha(__file__),
        component_entry_sha256=sha(args.entry),pressure_core_entry=core.entry,optimizer_updates=0,
        classification='Offline numerical controls and true-future-request oracle, not temporal candidates or frontier points',
        sampled_queries_only=True,reserved_scenes_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
