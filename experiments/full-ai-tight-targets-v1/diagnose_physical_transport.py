"""Test two frozen physical-arithmetic hypotheses on causal complete blocks."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from application_model_support import load_model
from restored_modal_core import RestoredModalCore
from physical_transport_composition import PhysicalTransportComposition
from diagnose_causal_composition import inventory,measure,FIELDS
from numerics import fine_action
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh physical arithmetic diagnosis required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered physical arithmetic source changed: '+path)
    if digest(reg['library_path'])!=reg['library_sha256']:raise ValueError('Registered FFTW binary changed')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);registry=read(ROOT/reg['models'])['methods'];bank=read(ROOT/reg['bank'])
    rows=[];outputs=[];sources={}
    def native(path,receipt,step):
        entry=receipt['states'][step];state=path.parent/entry['path']
        if digest(state)!=entry['sha256']:raise ValueError('Current/reference physical state changed')
        sources[str(state.relative_to(ROOT))]=digest(state)
        with np.load(state,allow_pickle=False)as archive:return {key:archive[key].copy()for key in (*FIELDS,'rhs')}
    with torch.inference_mode():
        for identifier in reg['identifiers']:
            spec=next(row for row in registry if row['identifier']==identifier)
            parent,norm,identity=load_model(spec['model']);original=inventory(parent);old_core=parent.pressure_core
            modal=RestoredModalCore(ROOT/reg['pressure_pair'],reg['library_path'],capture=True);modal_before=inventory(modal)
            parent.pressure_core=modal
            try:
                for mode in reg['modes']:
                    model=PhysicalTransportComposition(parent,mode)
                    for trajectory in bank['trajectories']:
                        path=ROOT/trajectory['receipt']
                        if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Development receipt changed')
                        receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=digest(path)
                        for start in reg['starts']:
                            current=native(path,receipt,start);modal.reset()
                            emitted=model.predict({name:current[name]for name in FIELDS},scene,start,norm)
                            if len(emitted)!=parent.horizons or len(modal.observations)!=parent.horizons:
                                raise ValueError('Every emitted physical step and learned pressure call required')
                            arrays={}
                            for lead,(state,observed)in enumerate(zip(emitted,modal.observations),1):
                                truth=native(path,receipt,start+lead);metrics=measure(state,truth,scene,norm)
                                q=observed['q'];pressure=observed['pressure'];h=1./scene['cells_per_metre']
                                native_q=torch.from_numpy(truth['rhs']).double().reshape_as(q)*(1000./h)
                                action=fine_action(pressure,torch.tensor([h],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))*(1000./h)
                                rows.append(dict(parent=identifier,mode=mode,identifier=trajectory['identifier'],start=start,lead=lead,
                                    counts_xyz=scene['counts'],metrics=metrics,
                                    generated_q_relative_error=float((q-native_q).norm()/native_q.norm().clamp_min(1e-30)),
                                    pressure_self_q_residual=float((action-q).norm()/q.norm().clamp_min(1e-30))))
                                for name in FIELDS:arrays[f'lead{lead}_{name}']=state[name].numpy().copy()
                                arrays[f'lead{lead}_generated_q']=q.numpy().copy()
                            output=args.output/(identifier+'-'+mode+'-'+trajectory['identifier']+f'-start{start}.npz')
                            np.savez_compressed(output,**arrays)
                            outputs.append(dict(parent=identifier,mode=mode,identifier=trajectory['identifier'],start=start,
                                horizons=parent.horizons,path=output.name,sha256=digest(output),bytes=output.stat().st_size))
                retained={key:value for key,value in inventory(parent).items()if not key.startswith('pressure_core.')}
                if retained!={key:value for key,value in original.items()if not key.startswith('pressure_core.')}or inventory(modal)!=modal_before:
                    raise ValueError('Physical arithmetic experiment changed a learned tensor')
            finally:parent.pressure_core=old_core;modal.close()
            if inventory(parent)!=original:raise ValueError('Parent checkpoint restoration failed')
            print(identifier,'both physical arithmetic hypotheses completed',flush=True)
    summaries=[]
    for identifier in reg['identifiers']:
        for mode in reg['modes']:
            for regime in ('small','32cubed'):
                selected=[row for row in rows if row['parent']==identifier and row['mode']==mode and (max(row['counts_xyz'])<=16)==(regime=='small')]
                summaries.append(dict(parent=identifier,mode=mode,regime=regime,physical_outputs=len(selected),
                    worst={field:max(row['metrics']['relative'][field]for row in selected)for field in ('equation','pressure','velocity','tracer')},
                    worst_generated_q_relative_error=max(row['generated_q_relative_error']for row in selected),
                    worst_pressure_self_q_residual=max(row['pressure_self_q_residual']for row in selected),
                    all_four_fields_below_1e5=all(row['metrics']['all_below_1e5']for row in selected)))
    if torch.cuda.is_initialized():raise ValueError('CPU diagnosis initialized CUDA')
    write(args.output/'results.json',dict(status='completed_causal_physical_arithmetic_diagnostic',rows=rows,summaries=summaries,
        physical_outputs=outputs,source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
        optimizer_updates=0,all_learned_tensors_unchanged=True,all_plan_owners_closed=True,cuda_initialized=False,
        truth_used_only_after_causal_prediction=True,every_intermediate_horizon_scored=True,own_history_240_steps=False,
        fresh_holdout_consumed=False,candidate_frontier_point=False,latency_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
