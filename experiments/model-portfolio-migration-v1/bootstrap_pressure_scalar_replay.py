"""Original two GN pressure fits: matched native banks and applicable water cases.

No prediction arrays are saved. Phase0 retains full scalar diagnostics; phase1
reverses case order and retains hashes/metrics to verify them independently.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from pressure_scalar_diagnostics import field_summary, error_summary, equation


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def run(protocol_path,fit_id,phase):
    protocol = read(protocol_path)
    require(phase in (0,1) and os.environ.get('TULDOK_TORCH_DEVICE')=='cuda', 'Explicit CUDA accuracy phase required')
    folder = ROOT/protocol['output_root']/(fit_id+'-phase'+str(phase))
    require(not folder.exists(), 'Fresh bootstrap qualification owner required')
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path'])==spec, 'Preregistered dependency changed')
    folder.mkdir(parents=True)
    result = dict(status='starting_bootstrap_pressure_CUDA_qualification',pid=os.getpid(),fit_item_id=fit_id,phase=phase,
        protocol=binding(protocol_path),rows=[],CPU_neural_forwards=0,latency_qualified=False,primary_timing_samples=0,
        reference_labels_forwarded=False,new_parameter_fits=0,new_reference_generation=0,prediction_arrays_saved=False)
    save(folder/'result.json',result)
    began = time.monotonic()
    try:
        os.sched_setaffinity(0,{protocol['host_cpu']})
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import numpy as np
        import torch
        from bootstrap_pressure_cuda import restore
        from python_source_coverage import collect
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        require(torch.cuda.get_device_name()==protocol['expected_GPU'],'Expected GPU required')
        material = read(ROOT/protocol['material']['path'])
        entries = {m['fit_item_id']:m for m in material['models']}
        selected = entries[fit_id]
        cases = []
        for cohort in protocol['fit_ids']:
            entry = entries[cohort]
            for spec in entry['inputs'].values():
                require(binding(ROOT/spec['path'])['sha256']==spec['sha256'],'Original native cohort changed')
            splits = read(ROOT/entry['inputs']['splits']['path'])
            for slot,name in (('train0','train-0.npz'),('train1','train-1.npz'),('validation','validation.npz')):
                shard = splits['shards'][name]
                with np.load(ROOT/entry['inputs'][slot]['path'],allow_pickle=False) as archive:
                    bank = {n:archive[n] for n in archive.files}
                require(set(bank)==set(shard['arrays']),'Closed original packet array inventory required')
                for name,value in bank.items():
                    spec = shard['arrays'][name]
                    require(value.dtype.str==spec['dtype'] and list(value.shape)==spec['shape'] and digest(value)==spec['sha256'],
                        'Original native typed array differs')
                for index,row in enumerate(shard['rows']):
                    for name in ('base','reference','rhs'):
                        require(digest(bank[name][index])==row['lineage'][name]['sha256'],'Original native per-frame lineage differs')
                    sample = {n:bank[n][index] for n in ('base','native_input_residual','reference','rhs','spacing','rho','dt')}
                    identity = dict(population='bootstrap_native_union768',cohort_fit_item_id=cohort,
                        original_split=row['split'],training_exposure=('own_' if cohort==fit_id else 'other_cohort_')+row['split'],
                        row_id=row['row_id'],trajectory_id=row['trajectory_id'],frame=row['frame'],time_s=row['time_s'],
                        counts_xyz=[16,16,16],spacing_xyz_m=sample['spacing'].tolist(),
                        cells_per_metre_xyz=(1/sample['spacing']).tolist(),fluid_density_kg_m3=float(sample['rho']),input_dt_s=float(sample['dt']),
                        forcing_regime=row['forcing_regime'],amplitude_m_s2=row['amplitude_m_s2'],seed=row['seed'])
                    cases.append((identity,sample))
        require(len(cases)==768,'Both384frame native cohorts required')
        import pressure_cases
        water = pressure_cases.load(torch,np)
        unsupported = []
        for index,(identity,sample) in enumerate(water):
            if tuple(identity['counts_xyz']) not in ((16,16,16),(32,32,32)):
                unsupported.append(dict(case_index=index,counts_xyz=identity['counts_xyz'],reason='Original forward requires exact16/32cubes'))
                continue
            converted = {n:sample[n].numpy().reshape(-1) for n in ('base','native_input_residual','reference','rhs')}
            converted.update(spacing=np.full(3,float(sample['cell_size_m'][0]),dtype='<f8'),
                rho=float(sample['rho'][0]),dt=float(sample['dt'][0]))
            converted = {**converted,**{n:converted[n].reshape(tuple(reversed(identity['counts_xyz']))) for n in ('base','native_input_residual','reference','rhs')}}
            ident = dict(identity,population='common_water128_applicable_subset',case_index=index,
                spacing_xyz_m=converted['spacing'].tolist(),fluid_density_kg_m3=converted['rho'],input_dt_s=converted['dt'],
                exposure='Consumed diagnostic;mass density1000 is outside original1.25training cohort,not independent holdout')
            cases.append((ident,converted))
        require(len(cases)==770 and len(unsupported)==126,'Exactly2applicable water requests and126explicit unsupported geometries required')
        scalars_path = folder/'statistics.jsonl.gz'
        own_losses = []
        handle = scalars_path.open('xb') if phase==0 else None
        compressed = gzip.GzipFile(filename='',mode='wb',fileobj=handle,compresslevel=6,mtime=0) if handle else None
        try:
            with restore(selected) as state:
                proof = state.proof
                sequence = range(len(cases)) if phase==0 else reversed(range(len(cases)))
                for case_index in sequence:
                    require(time.monotonic()-began < protocol['worker_deadline_seconds'],'Bootstrap worker deadline exceeded')
                    identity,sample = cases[case_index]
                    shape = tuple(reversed(identity['counts_xyz']))
                    tensor = lambda name:torch.from_numpy(np.asarray(sample[name],dtype='<f8').reshape(1,1,*shape))
                    base,residual = tensor('base'),tensor('native_input_residual')
                    spacing=torch.from_numpy(np.asarray(sample['spacing'],dtype='<f8').reshape(1,3))
                    rho=torch.tensor([sample['rho']],dtype=torch.float64)
                    dt=torch.tensor([sample['dt']],dtype=torch.float64)
                    input_hashes = {n:digest(np.asarray(sample[n])) for n in ('base','native_input_residual','spacing','rho','dt')}
                    calls_before = sum(state.observer.calls.values())
                    with torch.inference_mode():
                        p = state.module.predict_pressure(state.bridge,residual,base,spacing,rho,dt,state.normalization)
                        torch.cuda.synchronize()
                        require(p.device.type=='cpu' and p.dtype==torch.float64 and torch.isfinite(p).all(), 'Original complete CPU F64 pressure output required')
                        loss = float(state.module.pressure_loss(p,tensor('reference'),tensor('rhs'),spacing,rho,state.normalization)[0])
                    require(input_hashes=={n:digest(np.asarray(sample[n])) for n in input_hashes},'Prediction mutated original inputs')
                    predicted=p.numpy().reshape(shape);reference=np.asarray(sample['reference']).reshape(shape)
                    q=np.asarray(sample['rhs']).reshape(shape);base_array=np.asarray(sample['base']).reshape(shape)
                    difference=predicted-reference;difference-=difference.mean();target=reference-reference.mean()
                    pe=error_summary(difference,target)
                    eq=error_summary(equation(predicted,sample['spacing'],float(sample['rho']))-q,q)
                    base_error=base_array-reference;base_error-=base_error.mean()
                    be=error_summary(base_error,target)
                    metrics=dict(pressure_mean_free_relative_l2=pe['relative_l2'],pressure_absolute_rms=pe['rms'],
                        pressure_absolute_max=pe['maximum_absolute'],equation_relative_l2=eq['relative_l2'],
                        original_gauged_normalized_objective=loss)
                    row=dict(case_index=case_index,identity=identity,metrics=metrics,input_hashes=input_hashes,
                        reference_pressure_sha256=digest(reference),reference_rhs_sha256=digest(q),output_sha256=digest(predicted),
                        CUDA_module_calls=sum(state.observer.calls.values())-calls_before,
                        original_zero_residual_bypass=not np.any(sample['native_input_residual']))
                    result['rows'].append(row)
                    if identity['population']=='bootstrap_native_union768' and identity['cohort_fit_item_id']==fit_id and identity['original_split']=='validation':
                        own_losses.append(loss)
                    if compressed:
                        values=dict(prediction_pressure=field_summary(predicted),reference_pressure=field_summary(reference),
                            input_base=field_summary(base_array),input_residual=field_summary(sample['native_input_residual']),
                            pressure_mean_free_error=pe,equation_residual=eq,supplied_base_pressure_error=be,
                            prediction_minus_supplied_base=field_summary((predicted-base_array)-(predicted-base_array).mean()),
                            pressure_gauge_offset=float(predicted.mean()-reference.mean()),
                            pressure_RMS_improvement_vs_supplied_base=be['rms']-pe['rms'])
                        compressed.write((json.dumps(dict(fit_item_id=fit_id,case_index=case_index,statistics=values),
                            separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode())
                    if len(result['rows'])%128==0:
                        save(folder/'result.json',result)
                state.verify_state()
                observation=state.observer
            require(observation.closed and not observation.forbidden_attempts and len(own_losses)==96,'Guarded CUDA lifecycle/full validation population required')
        finally:
            if compressed:compressed.close()
            if handle:handle.close()
        history=read(ROOT/selected['outputs']['history.json']['path'])
        mean_loss=math.fsum(own_losses)/len(own_losses)
        historical=history['validation']['final']
        require(abs(mean_loss-historical)<=protocol['CPU_history_absolute_tolerance']+protocol['CPU_history_relative_tolerance']*abs(historical),
            'CUDA validation objective does not reproduce historical CPU checkpoint within preregisteredF32tolerance')
        source_coverage=collect(sys.modules,ROOT)
        for name in source_coverage['lab_python']:
            require(name in protocol['admitted_lab_python'] and binding(ROOT/name)['sha256']==protocol['admitted_lab_python'][name],
                'Unbound actual bootstrap source: '+name)
        result.update(status='bootstrap_pressure_CUDA_all_applicable_cases_qualified',cases=770,native_union_cases=768,
            water_applicable_cases=2,water_unsupported=unsupported,declared32cube_support_not_yet_exercised=True,
            device_observation=observation.evidence(),source_coverage=source_coverage,forward_variant=proof,
            selected_weights=selected['outputs']['weights.npz'],selected_parameters_unchanged=True,
            historical_CPU_own_validation_objective=historical,current_CUDA_own_validation_objective=mean_loss,
            historical_objective_absolute_difference=abs(mean_loss-historical),
            scalar_statistics=dict(**binding(scalars_path),bytes=scalars_path.stat().st_size) if phase==0 else None,
            CPU_history_bitwise_equivalence_claim=False,execution_variant='original_CPU_physical_features_CUDA_original_GroupNorm_CNN_CPU_F64_pressure',
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),GPU=torch.cuda.get_device_name(),
            torch_version=torch.__version__,CUDA_version=torch.version.cuda,wall_seconds=time.monotonic()-began,
            exposure='Posthoc consumed diagnostics on both original cohorts;whole-parent split retained,no independent confirmation claim',
            full_simulation_step=False,own_history_drift=False,native_mutations=0)
    except BaseException as error:
        result.update(status='bootstrap_pressure_CUDA_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(folder/'result.json',result)
    print(fit_id,'phase',phase,'770applicable requests qualified;126water grid rejections explicit,no arrays saved.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    parser.add_argument('--phase',type=int,required=True)
    args=parser.parse_args();run(args.protocol.resolve(),args.fit_id,args.phase)
