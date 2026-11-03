"""Finite full-state causal comparisons after replacing only the pressure branch."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from profile_frozen_components import ROOT, read, write, digest
from application_model_support import load_model
from restored_modal_core import RestoredModalCore
from numerics import fine_action
import numpy as np
import torch

FIELDS=('pressure','vx','vy','vz','tracer')


def inventory(model):
    return {name:dict(dtype=value.detach().cpu().numpy().dtype.str,shape=list(value.shape),
                     sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
            for name,value in model.state_dict().items()}


def measure(output,truth,scene,norm):
    fields={name:torch.as_tensor(output[name]).double().flatten()for name in FIELDS}
    reference={name:torch.from_numpy(truth[name]).double().flatten()for name in FIELDS}
    errors={};absolute={}
    for name,components in {'pressure':['pressure'],'velocity':['vx','vy','vz'],'tracer':['tracer']}.items():
        p=torch.cat([fields[k]for k in components]);t=torch.cat([reference[k]for k in components])
        if name=='pressure':p=p-p.mean();t=t-t.mean()
        rms=float((p-t).square().mean().sqrt());reference_rms=float(t.square().mean().sqrt())
        floor=norm['relative_denominator_rms_floor'][name]
        errors[name]=rms/max(reference_rms,floor);absolute[name]=rms
    shape=(1,1,*reversed(scene['counts']))
    h=torch.tensor([1./scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
    action=fine_action(fields['pressure'].reshape(shape),h,rho).flatten();rhs=torch.from_numpy(truth['rhs']).double().flatten()
    errors['equation']=float((action-rhs).norm()/rhs.norm().clamp_min(1e-30))
    if not all(np.isfinite(v)for v in errors.values()):raise ValueError('Nonfinite full-state diagnostic')
    return dict(relative=errors,absolute_rms=absolute,all_below_1e5=all(v<=1e-5 for v in errors.values()))


def run(args):
    if args.output.exists():raise ValueError('Fresh causal composition diagnosis required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered composition source/evidence changed: '+path)
    if digest(reg['library_path'])!=reg['library_sha256']:raise ValueError('Registered double FFTW changed')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True);registry=read(ROOT/reg['models'])['methods'];bank=read(ROOT/reg['bank'])
    all_rows=[];sources={};identities=[];saved=[]
    def native(receipt_path,receipt,step):
        spec=receipt['states'][step];path=receipt_path.parent/spec['path']
        if digest(path)!=spec['sha256']:raise ValueError('Native current/reference state changed')
        sources[str(path.relative_to(ROOT))]=digest(path)
        with np.load(path,allow_pickle=False)as archive:return {key:archive[key].copy()for key in (*FIELDS,'rhs')}
    with torch.inference_mode():
        for identifier in reg['identifiers']:
            method=next(row for row in registry if row['identifier']==identifier)
            network,norm,identity=load_model(method['model']);before=inventory(network)
            original_core=network.pressure_core
            modal=RestoredModalCore(ROOT/reg['pressure_pair'],reg['library_path'],capture=True)
            pressure_before=inventory(modal)
            retained_before={key:value for key,value in before.items()if not key.startswith('pressure_core.')}
            identities.append(dict(parent=identity,modal=modal.identity))
            try:
                for row in bank['trajectories']:
                    receipt_path=ROOT/row['receipt']
                    if digest(receipt_path)!=row['receipt_sha256']:raise ValueError('Development scene changed')
                    receipt=read(receipt_path);scene=receipt['scene'];sources[str(receipt_path.relative_to(ROOT))]=digest(receipt_path)
                    for start in reg['starts']:
                        current=native(receipt_path,receipt,start);fields={key:current[key]for key in FIELDS}
                        network.pressure_core=original_core
                        baseline=network.predict(fields,scene,start,norm)
                        network.pressure_core=modal;modal.reset();tick=time.perf_counter()
                        emitted=network.predict(fields,scene,start,norm);elapsed=time.perf_counter()-tick
                        if len(emitted)!=network.horizons or len(modal.observations)!=network.horizons:
                            raise ValueError('Every actual pressure call/horizon must be captured')
                        arrays={};block_rows=[]
                        for lead,(base,quantized,observed)in enumerate(zip(baseline,emitted,modal.observations),1):
                            # The double output is the same actual learned
                            # pressure used by momentum, before legacy codec.
                            precise=dict(quantized,pressure=observed['pressure'].flatten())
                            truth=native(receipt_path,receipt,start+lead)
                            for policy,predicted in [('parent',base),('modal_legacy_pressure_codec',quantized),('modal_double_pressure_output',precise)]:
                                value=measure(predicted,truth,scene,norm)
                                record=dict(parent=identifier,horizons=network.horizons,identifier=row['identifier'],counts_xyz=scene['counts'],
                                    start=start,lead=lead,policy=policy,metrics=value)
                                if policy!='parent':
                                    q=observed['q'];p=observed['pressure'];h=1./scene['cells_per_metre']
                                    action=fine_action(p,torch.tensor([h],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))*(1000./h)
                                    native_q=torch.from_numpy(truth['rhs']).double().reshape_as(q)*(1000./h)
                                    record['generated_q_relative_error']=float((q-native_q).norm()/native_q.norm().clamp_min(1e-30))
                                    record['pressure_self_q_residual']=float((action-q).norm()/q.norm().clamp_min(1e-30))
                                all_rows.append(record);block_rows.append(record)
                                for key in FIELDS:arrays[f'{policy}_lead{lead}_{key}']=predicted[key].cpu().numpy().copy()
                            arrays[f'lead{lead}_generated_q']=observed['q'].numpy().copy()
                            if any(not torch.equal(quantized[key],precise[key])for key in FIELDS if key!='pressure'):
                                raise ValueError('Pressure output precision changed any emitted carrier')
                        path=args.output/(identifier+'-'+row['identifier']+f'-start{start}.npz')
                        np.savez_compressed(path,**arrays)
                        saved.append(dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size,
                            parent=identifier,identifier=row['identifier'],start=start,horizons=network.horizons,
                            instrumented_modal_block_seconds=elapsed,not_benchmark_latency=True))
                retained_after={key:value for key,value in inventory(network).items()if not key.startswith('pressure_core.')}
                if retained_after!=retained_before or inventory(modal)!=pressure_before:
                    raise ValueError('Composition changed retained temporal or restored pressure tensors')
            finally:
                network.pressure_core=original_core;modal.close()
            if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Restoring parent changed tensors or initialized CUDA')
            print(identifier,'all full-state causal block diagnostics completed',flush=True)
        summaries=[]
        for identifier in reg['identifiers']:
            for policy in ('parent','modal_legacy_pressure_codec','modal_double_pressure_output'):
                for regime in ('small','32cubed'):
                    selected=[row for row in all_rows if row['parent']==identifier and row['policy']==policy and (max(row['counts_xyz'])<=16)==(regime=='small')]
                    summaries.append(dict(parent=identifier,policy=policy,regime=regime,physical_outputs=len(selected),
                        worst={field:max(row['metrics']['relative'][field]for row in selected)for field in ('equation','pressure','velocity','tracer')},
                        all_four_fields_below_1e5=all(row['metrics']['all_below_1e5']for row in selected),
                        worst_generated_q_relative_error=max((row.get('generated_q_relative_error',0.)for row in selected)),
                        worst_pressure_self_q_residual=max((row.get('pressure_self_q_residual',0.)for row in selected))))
    write(args.output/'results.json',dict(status='completed_causal_full_state_composition_diagnostic',summaries=summaries,rows=all_rows,
        models=identities,physical_outputs=saved,source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
        optimizer_updates=0,retained_parameter_tensors_unchanged=True,restored_pressure_tensors_unchanged=True,
        all_plan_owners_closed=True,cuda_initialized=False,truth_used_only_after_causal_prediction=True,
        every_intermediate_horizon_scored=True,own_history_240_steps=False,fresh_holdout_consumed=False,
        candidate_frontier_point=False,instrumented_times_are_not_acceptance_benchmarks=True,
        parent_flux_already_uses_float64_convolution=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
