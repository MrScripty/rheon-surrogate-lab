"""One original train checkpoint on the frozen128case pressure bank, CUDA only.

This is accuracy/adapter qualification. No latency, full simulation step,
parameter update, independent holdout or own-history drift claim is made.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,read,save,binding,require

UI = DOC/'reports/native-matched-portfolio-UI-01.json'
CUSTODY = DOC/'reports/native-matched-generator-outcome-adoption-01.json'
INPUTS = ('base','native_input_residual','cell_size_m','rho','dt')
METRICS = ('common_equation_relative_l2','pressure_mean_free_relative_l2','pressure_absolute_rms','pressure_absolute_max')


def publication_gate(ui_path=UI,custody_path=CUSTODY,header_path=PROJECT/'.tuldok/project.json'):
    require(ui_path.is_file() and custody_path.is_file() and header_path.is_file(),
            'Accepted native browser and actual image inspection required before CUDA qualification')
    ui,custody = read(ui_path),read(custody_path)
    raw = header_path.read_bytes()
    current = json.loads(raw)
    require(ui['status'] == 'matched39configurations_and35outcome_notes_actual_native_UI_accepted'
            and ui['browser_stopped'] and len(ui['captures_actually_viewed']) >= 10
            and ui['requested_chart_comparisons'] == 40 and ui['displayed_numeric_presence_assertions'] == 390,
            'Complete real chart/summary/image acceptance required')
    require(custody['status'] == 'native38matched_outcome_notes_exact_and_scientific_records_unchanged'
            and current['checkpoint'] == ui['checkpoint'] == custody['checkpoint']
            and hashlib.sha256(raw).hexdigest() == custody['canonical_header_sha256'],
            'Current native project differs from accepted publication checkpoint')
    return ui


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def request_binding(row):
    return {k:row[k] for k in ('trajectory_id','parent','native_frame','counts_xyz','cell_size_m',
        'input_sha256','reference_pressure_sha256','reference_rhs_sha256')}


def matching_archive(protocol,fit_id,rows,weights_sha256):
    """Reuse only a retained archive whose complete original case hashes match."""
    compared = []
    for candidate in protocol.get('prior_controls',{}).get(fit_id,[]):
        spec = candidate['result']
        require(binding(ROOT/spec['path']) == spec,'Prior control result changed')
        prior = read(ROOT/spec['path'])
        require(prior['weights_sha256'] == weights_sha256 and len(prior['rows']) == len(rows) == 128,
                'Prior same-weight control population differs')
        require(prior['status'] == 'completed_captured_pressure_GPU_accuracy'
                and prior['neural_module_forward_calls']['cpu'] == 0 and prior['learned_parameters_unchanged']
                and prior['architecture_unchanged'],'Previously qualified original CUDA control required')
        exact = all(request_binding(a) == request_binding(b) and a['output_sha256'] == b['output_sha256']
            for a,b in zip(rows,prior['rows'],strict=True))
        compared.append(dict(identifier=candidate['identifier'],all128_request_reference_and_output_hashes_exact=exact,
            result=spec,general_policy_equivalence_qualified=False))
        archive = candidate.get('archive')
        if exact and archive is not None:
            require(ROOT/archive['path'] == (ROOT/spec['path']).parent/prior['output_archive']['path']
                    and archive['sha256'] == prior['output_archive']['sha256'],'Archive must belong to the matched original result')
            require(binding(ROOT/archive['path']) == {k:archive[k] for k in ('path','sha256')},'Original control archive changed')
            return dict(kind='existing_original_prediction_archive',**archive),compared
    return None,compared


def run(protocol_path,fit_id):
    publication_gate()
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rheon.original-train-pressure-CUDA-qualification.v1'
            and protocol['device'] == 'cuda' and protocol['case_count'] == 128
            and fit_id in protocol['fit_ids'] and protocol['latency_claim'] is False
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda','Explicit frozen CUDA accuracy protocol required')
    for path,expected in protocol['source_sha256'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected,'Preregistered pressure qualification source changed')
    material_path = ROOT/protocol['material']['path']
    require(binding(material_path) == protocol['material'],'Original train material changed')
    material = read(material_path)
    output = ROOT/protocol['output_root']/fit_id
    require(not output.exists() and output.is_relative_to(RUN),'Fresh task-owned worker output required; no retry')
    os.sched_setaffinity(0,{protocol['host_cpu']})
    output.mkdir(parents=True)
    result = dict(status='starting_original_train_pressure_CUDA_qualification',pid=os.getpid(),created_unix=time.time(),fit_item_id=fit_id,
        protocol=binding(protocol_path),material=protocol['material'],rows=[],active_case=None,
        new_parameter_updates=0,latency_claim=False,full_simulation_step=False,independent_holdout=False,own_history_drift=False,
        reference_labels_forwarded=False,CPU_neural_fallback=False,full_portfolio_complete=False)
    save(output/'result.json',result)
    began = time.monotonic()
    observed = None
    try:
        from original_train_pressure_cuda import restore
        import pressure_cases
        with restore(material,fit_id) as state:
            torch = state.torch
            observed = state.device_observation
            # This is the same accepted common finite-volume operator used for
            # all127named checkpoints; it executes CPU scalar/tensor math only.
            import score_pressure_portfolio as common
            common.operator_check()
            cases = pressure_cases.load(torch,state.np)
            require(len(cases) == 128 and all(v.device.type == 'cpu' for _,s in cases for v in s.values()),
                    'Complete original owned CPU input/reference population required')
            require(torch.cuda.get_device_name() == protocol['expected_GPU']
                    and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                    'Preregistered GPU and unchanged explicit precision settings required')
            result.update(GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
                original_receipt_mode='train',architecture=state.model.ARCHITECTURE,
                weights_sha256=state.entry['outputs']['weights.npz']['sha256'],host_cpu=protocol['host_cpu'],
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
            arrays = {}
            for index,(identity,sample) in enumerate(cases):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'],'Original accuracy worker deadline exceeded')
                require(torch.cuda.max_memory_allocated() <= protocol['max_CUDA_allocated_bytes'],'Original accuracy worker exceeds CUDA allocation bound')
                result['active_case'] = dict(index=index,**identity)
                host_inputs = {name:sample[name] for name in INPUTS}
                original_inputs = {name:tensor_hash(value) for name,value in host_inputs.items()}
                forwarded = {name:value.to('cuda') for name,value in host_inputs.items()}
                with torch.device('cuda'):
                    prediction = state.model.predict_pressure(state.network,forwarded,state.normalization)
                require(prediction.device.type == 'cuda' and prediction.dtype == torch.float64
                        and tuple(prediction.shape) == tuple(sample['base'].shape),'Original complete CUDA F64 pressure field required')
                pressure = prediction.detach().cpu().contiguous()
                torch.cuda.synchronize()
                require(bool(torch.isfinite(pressure).all())
                        and all(tensor_hash(v) == original_inputs[name] for name,v in host_inputs.items()),
                        'Finite prediction and unchanged caller-owned CPU inputs required')
                residual = common.equation(pressure,sample['cell_size_m'],sample['rho'])-sample['rhs']
                captured = state.model.full_operator(pressure,sample['cell_size_m'],sample['rho'])-sample['rhs']
                difference = pressure-sample['reference']
                difference = difference-difference.mean()
                reference = sample['reference']-sample['reference'].mean()
                rhs_norm,reference_norm = float(torch.linalg.vector_norm(sample['rhs'])),float(torch.linalg.vector_norm(reference))
                values = dict(common_equation_relative_l2=float(torch.linalg.vector_norm(residual))/max(rhs_norm,1e-30),
                    pressure_mean_free_relative_l2=float(torch.linalg.vector_norm(difference))/max(reference_norm,1e-30),
                    pressure_absolute_rms=float(difference.square().mean().sqrt()),pressure_absolute_max=float(difference.abs().max()),
                    captured_common_equation_max_abs_difference=float((captured-residual).abs().max()),
                    equation_rhs_l2=rhs_norm,pressure_reference_mean_free_l2=reference_norm,
                    equation_floor_active=rhs_norm<1e-30,pressure_floor_active=reference_norm<1e-30)
                require(all(math.isfinite(v) for v in values.values()),'Finite pressure metrics required')
                result['rows'].append(dict(**identity,metrics=values,input_sha256=original_inputs,output_sha256=tensor_hash(pressure),
                    reference_pressure_sha256=tensor_hash(sample['reference']),reference_rhs_sha256=tensor_hash(sample['rhs'])))
                arrays[f'case{index:03d}_pressure'] = pressure.numpy().copy()
                if index%32 == 31:
                    save(output/'result.json',result)
                    print(fit_id,index+1,'original CUDA pressure cases scored',flush=True)
            archive,comparisons = matching_archive(protocol,fit_id,result['rows'],result['weights_sha256'])
            if archive is None:
                path = output/'pressure-outputs.npz'
                state.np.savez_compressed(path,**arrays)
                archive = dict(kind='new_original_fit_predictions',**binding(path),bytes=path.stat().st_size)
            result.update(output_archive=archive,same_weight_control_comparisons=comparisons)
        require(observed['scoped_guard_removed'] and observed['CPU_neural_forwards'] == 0
                and observed['forbidden_forward_attempts'] == 0,'Complete CUDA guard lifecycle required')
        for path,expected in protocol['source_sha256'].items():
            require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected,'Qualification source changed during execution')
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT)
        imported = coverage['lab_python']
        result['actual_bound_lab_Python_modules'] = imported
        result['runtime_virtual_modules'] = coverage['runtime_virtual_modules']
        result['unbound_lab_Python_modules'] = [path for path in imported if path not in protocol['source_sha256']]
        save(output/'result.json',result)
        require(not result['unbound_lab_Python_modules'],'Actual imported lab Python source absent from frozen protocol')
        result.update(status='original_train_pressure_CUDA_accuracy_qualified',finished_unix=time.time(),active_case=None,
            cases=128,learned_tensors_exact_and_unchanged=True,captured_architecture_unchanged=True,device_observation=observed,
            worst={name:max(row['metrics'][name] for row in result['rows']) for name in METRICS},
            maximum_captured_common_equation_difference=max(row['metrics']['captured_common_equation_max_abs_difference'] for row in result['rows']),
            peak_CUDA_allocated_bytes=state.torch.cuda.max_memory_allocated(),diagnostic_wall_seconds=time.monotonic()-began)
        result['actual_bound_lab_Python_modules'] = imported
    except BaseException as error:
        result.update(status='original_train_pressure_adapter_gap',error_type=type(error).__name__,error=str(error),
            device_observation=observed,scientific_accuracy_failure_claim=False)
        raise
    finally:
        save(output/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.fit_id)
