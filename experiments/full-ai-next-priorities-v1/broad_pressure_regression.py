"""Train/development residual families, boundary localization and stage spectra."""
import argparse
import math
import os
from pathlib import Path
from study_support import ROOT,read,write,sha,load_checkpoint,historical_cases


def run(args):
    if args.output.exists():raise ValueError('Fresh pressure residual diagnosis required')
    os.sched_setaffinity(0,{22});state=load_checkpoint(args.identifier);torch=state.torch
    torch.set_num_threads(1);project=ROOT/state.entry['project']
    import train,physical_residual,numerics
    data_project=ROOT/'runs/full-ai-next-priorities-v1/a-combined-project-01'
    slots={r['id']:ROOT/r['path']for r in read(data_project/'project.json')['data']}
    for slot,spec in read(data_project/'source-witness.json')['inputs'].items():
        if sha(slots[slot])!=spec['sha256']:raise ValueError('Original physical regression input changed')
    _,parents=state.data.load(tuple(slots[k].read_bytes()for k in ['dataset','dataset_extra',*train.VIEW_SLOTS]),read(slots['splits']))
    if any(p['entry']['split']!='train'for p in parents):raise ValueError('Train diagnosis requires the original train-only view')
    cases=[(dict(split='train',trajectory_id=parent['trajectory_id'],native_frame=parent['frames'][frame]['index'],
        counts_xyz=parent['counts_xyz'],cell_size_m=parent['cell_size_m']),state.data.sample(parent,frame))
        for parent in parents for frame in range(len(parent['frames']))]
    cases.extend((dict(identity,split='consumed_historical_development'),sample)for identity,sample in historical_cases(state))
    rows=[];inventory=state.data.helpers().state_inventory(state.network)
    with torch.inference_mode():
        for identity,sample in cases:
            pressures=[];physical_residual.predict_pressure(state.network,sample,state.normalization,state.model.owner,state.model.CORRECTION,pressures)
            boundary=torch.zeros_like(sample['rhs'],dtype=torch.bool)
            for axis in (2,3,4):
                face=[slice(None)]*5;face[axis]=0;boundary[tuple(face)]=True
                face[axis]=-1;boundary[tuple(face)]=True
            axes,_=numerics.geometry(tuple(sample['rhs'].shape[2:]),4)
            denominator=max(float(torch.linalg.vector_norm(sample['rhs'])),1e-30);stages=[]
            for stage,pressure in enumerate([sample['base'],*pressures]):
                residual=state.model.full_operator(pressure,sample['cell_size_m'],sample['rho'])-sample['rhs']
                norm=float(torch.linalg.vector_norm(residual));low=float(torch.linalg.vector_norm(numerics.coefficients(residual,axes)))
                stages.append(dict(stage=stage,equation_relative_residual=norm/denominator,
                    boundary_relative_residual=float(torch.linalg.vector_norm(residual[boundary]))/denominator,
                    interior_relative_residual=float(torch.linalg.vector_norm(residual[~boundary]))/denominator,
                    low_modes_norm=low,high_modes_norm=math.sqrt(max(0.,norm*norm-low*low)),
                    low_modes_squared_energy_fraction=low*low/max(norm*norm,1e-300)))
            rows.append(dict(identity,stages=stages))
    if state.data.helpers().state_inventory(state.network)!=inventory:raise ValueError('Diagnosis changed weights')
    write(args.output,dict(status='completed',model_entry=state.entry,rows=rows,source_sha256=sha(__file__),
        pressure_operator_source_sha256=sha(project/'portable_model.py'),
        low_modes='Four orthonormal cosine modes per axis including DC; projection only, no pressure inverse',
        optimizer_updates=0,new_reserved_cases_consumed=False,
        physical_data_project=str(data_project.relative_to(ROOT)),physical_data_witness_sha256=sha(data_project/'source-witness.json'),
        classification='Previously consumed C107 physical training/development cases; regression only for head calibration',
        train_requests=sum(r['split']=='train'for r in rows),historical_requests=sum(r['split']!='train'for r in rows)))
    print('Diagnosed pressure stages and boundary/low/high residual families',args.identifier,len(rows),'requests',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
