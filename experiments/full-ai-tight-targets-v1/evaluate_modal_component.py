"""Evaluate exact restored learned gains on consumed native RHS, offline only."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from learned_modal_pressure import LearnedModalPressure
from modal_basis import bases, transform
from fftw_pressure_transform_v2 import MeasuredDCTMultiplier
from train_modal_pressure import inventory, load
from numerics import fine_action
import numpy as np
import torch


def relative(value, reference):
    return float((value-reference).norm()/reference.norm().clamp_min(1e-30))


def run(args):
    if args.output.exists(): raise ValueError('Fresh offline component evaluation required')
    reg=read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected: raise ValueError('Registered model/evaluation source changed: '+path)
    if digest(reg['library_path'])!=reg['library_sha256']: raise ValueError('Registered FFTW binary changed')
    pair=ROOT/reg['pair']; receipt=read(pair/'receipt.json'); restored=read(pair/'restore/receipt.json')
    trained=read(pair/'train/receipt.json')
    if receipt['status']!='completed' or not all(receipt['artifact_parity'].values()) or trained['optimizer_updates']!=800:
        raise ValueError('One exact complete managed CUDA fit/CPU restore required')
    if restored['optimizer_updates']!=0 or restored['execution']['cuda_initialized'] or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Fresh typed double CPU probe/weights must verify first')
    for phase in ('train','restore'):
        terminal=read(pair/('terminal-'+receipt[phase+'ing_run_id' if phase=='train' else 'restore_run_id']+'.json'))
        if terminal['state']!='completed' or terminal['custody_phase']!='drained': raise ValueError('Managed owners must complete and drain before CPU evaluation')
    assembly=read(ROOT/receipt['source_project']/'assembly.json')
    if digest(ROOT/receipt['source_project']/'assembly.json')!=receipt['source_assembly_sha256']: raise ValueError('Captured training Project changed')
    for source in assembly['sources']:
        if digest(ROOT/source['path'])!=source['sha256']: raise ValueError('Captured inference source changed')
    manifest=read(ROOT/assembly['static_inputs']['manifest']['path'])
    model=LearnedModalPressure(manifest['geometries'])
    load(model,pair/'restore/weights.npz',restored['final_inventory']); model.eval().requires_grad_(False)
    before=inventory(model)
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True); rows=[]; sources={}; owners={}; gains={}; axes={}; outputs=[]
    with torch.inference_mode():
        try:
            for row in read(ROOT/reg['bank'])['trajectories']:
                path=ROOT/row['receipt']
                if digest(path)!=row['receipt_sha256']:raise ValueError('Consumed development receipt changed')
                scene_receipt=read(path);scene=scene_receipt['scene'];shape=tuple(reversed(scene['counts']))
                sources[str(path.relative_to(ROOT))]=digest(path); key='g'+'_'.join(map(str,shape))
                if key not in owners:
                    owners[key]=MeasuredDCTMultiplier((1,1,*shape),reg['library_path'])
                    gains[key]=model.gains(key).reshape(1,1,*shape).contiguous()
                    axes[key]=bases(shape)
                h=torch.tensor([1./scene['cells_per_metre']],dtype=torch.float64)
                rho=torch.tensor([1000.],dtype=torch.float64); predictions=[]
                for step in range(reg['first_state'],reg['last_state']+1):
                    spec=scene_receipt['states'][step];state=path.parent/spec['path']
                    if digest(state)!=spec['sha256'] or spec['step']!=step:raise ValueError('Native timestamp/pressure/RHS changed')
                    sources[str(state.relative_to(ROOT))]=digest(state)
                    with np.load(state,allow_pickle=False)as archive:
                        truth=torch.from_numpy(archive['pressure'].copy()).double().reshape(1,1,*shape)
                        rhs=torch.from_numpy(archive['rhs'].copy()).double().reshape_as(truth)
                    q=(rhs*(1000.*scene['cells_per_metre'])).contiguous()
                    pressure=owners[key](q,gains[key]);pressure=pressure-pressure[:,:,:1,:1,:1]
                    if not bool(torch.isfinite(pressure).all()):raise ValueError('Nonfinite actual learned pressure')
                    fftw_residual=relative(fine_action(pressure,h,rho),rhs)
                    fftw_pressure=relative(pressure-pressure.mean(),truth-truth.mean())
                    coefficients=transform(q[:,0],axes[key]);p=model(coefficients,key)
                    control=transform(p,axes[key],inverse=True)[:,None];control=control-control[:,:,:1,:1,:1]
                    parity=relative(pressure,control)
                    native_residual=relative(fine_action(truth,h,rho),rhs)
                    if parity>reg['maximum_execution_relative_error'] or native_residual>reg['maximum_native_control_residual']:
                        raise ValueError('Actual multiplier execution or native operator qualification failed')
                    rows.append(dict(identifier=row['identifier'],state=step,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
                        equation_relative_l2=fftw_residual,mean_free_pressure_relative_l2=fftw_pressure,
                        native_control_equation_relative_l2=native_residual,torch_fftw_pressure_relative_l2=parity,
                        both_pressure_criteria_pass=(fftw_residual<=reg['maximum_error'] and fftw_pressure<=reg['maximum_error'])))
                    predictions.append(pressure.numpy()[0,0].copy())
                output=args.output/(row['identifier']+'.npz')
                np.savez_compressed(output,pressure=np.stack(predictions).astype('<f8'))
                outputs.append(dict(identifier=row['identifier'],path=output.name,sha256=digest(output),bytes=output.stat().st_size,
                    counts_xyz=scene['counts'],first_state=reg['first_state'],last_state=reg['last_state'],dtype='<f8'))
                selected=[value for value in rows if value['identifier']==row['identifier']]
                print(row['identifier'],'worst Eq/P',max(v['equation_relative_l2']for v in selected),max(v['mean_free_pressure_relative_l2']for v in selected),flush=True)
        finally:
            for owner in owners.values():owner.close()
    if inventory(model)!=before or torch.cuda.is_initialized():raise ValueError('CPU evaluation changed parameters or initialized CUDA')
    summaries=[]
    for regime in ('small','32cubed','all'):
        selected=[r for r in rows if regime=='all' or (max(r['counts_xyz'])<=16)==(regime=='small')]
        summaries.append(dict(regime=regime,physical_pressure_states=len(selected),
            worst_equation_relative_l2=max(r['equation_relative_l2']for r in selected),
            worst_mean_free_pressure_relative_l2=max(r['mean_free_pressure_relative_l2']for r in selected),
            worst_execution_relative_l2=max(r['torch_fftw_pressure_relative_l2']for r in selected),
            component_pressure_accuracy_pass=all(r['both_pressure_criteria_pass']for r in selected)))
    write(args.output/'results.json',dict(status='completed_offline_learned_pressure_component_evaluation',summaries=summaries,
        rows=rows,pressure_outputs=outputs,source_evidence=sources,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],
        training_run_id=receipt['training_run_id'],restore_run_id=receipt['restore_run_id'],weights_sha256=digest(pair/'restore/weights.npz'),
        optimizer_updates=0,weights_unchanged=True,cuda_initialized=False,teacher_rhs_used=True,
        causal_full_state_generator=False,own_history_rollout=False,fresh_holdout_consumed=False,
        candidate_frontier_point=False,all_plan_owners_closed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
