"""Exact original pressure policies; hook-free complete-call GPU timing.

Primary latency is milliseconds per pressure request, never per simulation step.
Device observation covers construction, setup, first calls, warmups and the on
side of separately paired calibration. It is removed for primary/off samples.
"""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from import_common_generator_measurements import ROOT,RUN,read,save,binding,require
from original_train_pressure_accuracy_02 import publication_gate,tensor_hash,request_binding
from original_pressure_complete_call import FIELDS,make_complete,check_output
from pressure_portfolio_gpu_timing import resources


def verify_sources(protocol):
    for name,expected in protocol['source_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,
                'Preregistered pressure timing source changed: '+name)


def run(protocol_path,fit_id,phase):
    publication_gate()
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rheon.original-train-pressure-GPU-timing.v1'
            and protocol['device'] == 'cuda' and protocol['cases_per_fit'] == 128
            and fit_id in protocol['fit_ids'] and phase in (0,1)
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda',
            'Explicit original-training CUDA timing protocol required')
    verify_sources(protocol)
    for spec in [protocol['material'],protocol['accuracy_reduction'],protocol['runtime_manifest'],protocol['frozen_cases'],
                 protocol['publication_acceptance'],protocol['accuracy'][fit_id]]:
        require(binding(ROOT/spec['path']) == spec,'Frozen timing material/evidence changed')
    material = read(ROOT/protocol['material']['path'])
    accuracy = read(ROOT/protocol['accuracy'][fit_id]['path'])
    require(accuracy['status'] == 'original_train_pressure_CUDA_accuracy_qualified' and accuracy['cases'] == 128,
            'Independent original-training pressure accuracy required')
    frozen = read(ROOT/protocol['frozen_cases']['path'])
    output = ROOT/protocol['output_root']/fit_id/f'phase-{phase}'
    require(output.is_relative_to(RUN) and not output.exists(),'Fresh timing phase required; no replay')
    os.sched_setaffinity(0,{protocol['host_cpu']})
    output.mkdir(parents=True)
    result = dict(schema='rheon.original-train-pressure-GPU-timing-worker.v1',status='starting',pid=os.getpid(),
        created_unix=time.time(),fit_item_id=fit_id,phase=phase,protocol=binding(protocol_path),rows=[],active_case=None,
        accuracy=protocol['accuracy'][fit_id],resource_observations=[],inference_device='cuda',timing_units='ms/pressure_request',
        timing_scope=protocol['timing_scope'],timed_module_hooks_enabled=False,calibration_pairs_per_case=protocol['calibration_pairs_per_case'],
        new_parameter_updates=0,new_reference_generation=False,full_simulation_step=False,independent_holdout=False,
        own_history_drift=False,CPU_neural_fallback=False,reference_labels_forwarded=False,full_portfolio_complete=False)
    save(output/'result.json',result)
    began = time.monotonic()
    observer = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from restore_original_pressure_for_timing import restore
        import pressure_cases
        with restore(material,fit_id) as state:
            observer = state.observer
            require(torch.cuda.get_device_name() == protocol['expected_GPU']
                    and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                    'Registered GPU and exact precision settings required')
            require(accuracy['weights_sha256'] == state.entry['outputs']['weights.npz']['sha256'],
                    'Pressure timing checkpoint differs from accuracy')
            cases = pressure_cases.load(torch,state.np)
            require(len(cases) == len(accuracy['rows']) == len(frozen['cases']) == 128
                    and all(value.device.type == 'cpu' for _,sample in cases for value in sample.values()),
                    'Complete same128 owned CPU requests required')
            result.update(GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
                host_cpu=protocol['host_cpu'],threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                original_architecture=state.model.ARCHITECTURE,weights_sha256=accuracy['weights_sha256'],
                setup_wall_seconds=time.monotonic()-began)
            complete = make_complete(state)
            indices = list(range(128))
            if phase:
                indices.reverse()
            result['case_execution_order'] = indices

            def sample_call(native,expected):
                started = time.perf_counter_ns()
                value = complete(native)
                elapsed = time.perf_counter_ns()-started
                check_output(state.np,value,expected,native)
                require(elapsed > 0,'Positive complete-call timing required')
                return elapsed

            for execution_index,index in enumerate(indices):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'],'Pressure timing phase deadline exceeded')
                identity,sample = cases[index]
                baseline,case = accuracy['rows'][index],frozen['cases'][index]
                result['active_case'] = dict(case_index=index,execution_index=execution_index,**identity)
                hashes = {name:tensor_hash(sample[name]) for name in FIELDS}
                require(hashes == baseline['input_sha256'] == case['original_case']['input_sha256']
                        and request_binding(baseline) == case['original_case']
                        and all(baseline[name] == value for name,value in identity.items()),
                        'Exact original request/reference identity required')
                require(tensor_hash(sample['reference']) == baseline['reference_pressure_sha256']
                        and tensor_hash(sample['rhs']) == baseline['reference_rhs_sha256'],'Reference labels differ')
                native = {name:sample[name].numpy()[0,0].copy() for name in FIELDS[:2]}
                native.update({name:float(sample[name].item()) for name in FIELDS[2:]})
                if execution_index%32 == 0:
                    observation = resources()
                    result['resource_observations'].append(observation)
                    require(not observation['other_compute_pids'],'Other GPU compute observed; preserve as contended')
                observer.set_phase('first_call')
                first_ns = sample_call(native,baseline['output_sha256'])
                observer.set_phase('warmup')
                warmups = [sample_call(native,baseline['output_sha256']) for _ in range(protocol['warmups_per_case'])]
                before = dict(observer.calls)
                with observer.unobserved('primary'):
                    require(observer.handle is None,'Primary must have no device observer callback')
                    samples = [sample_call(native,baseline['output_sha256']) for _ in range(protocol['samples_per_case'])]
                    require(dict(observer.calls) == before,'Primary incorrectly included device observer')
                calibration = []
                for pair_index in range(protocol['calibration_pairs_per_case']):
                    order = ['on','off'] if (phase+index+pair_index)%2 == 0 else ['off','on']
                    pair = dict(pair_index=pair_index,order=order)
                    for mode in order:
                        if mode == 'on':
                            observer.set_phase('calibration_on')
                            pair[mode+'_ns'] = sample_call(native,baseline['output_sha256'])
                        else:
                            before = dict(observer.calls)
                            with observer.unobserved('calibration_off'):
                                pair[mode+'_ns'] = sample_call(native,baseline['output_sha256'])
                                require(dict(observer.calls) == before,'Off-calibration incorrectly observed')
                    pair['on_minus_off_ns'] = pair['on_ns']-pair['off_ns']
                    calibration.append(pair)
                require(all(hashlib.sha256(native[name].tobytes()).hexdigest() == hashes[name] for name in FIELDS[:2])
                        and all(tensor_hash(sample[name]) == hashes[name] for name in FIELDS),
                        'Caller-owned inputs changed')
                result['rows'].append(dict(case_index=index,case_id=case['id'],**identity,
                    cells_per_metre=case['cells_per_metre'],input_dt_s=case['input_dt_s'],input_sha256=hashes,
                    reference_pressure_sha256=baseline['reference_pressure_sha256'],reference_rhs_sha256=baseline['reference_rhs_sha256'],
                    output_sha256=baseline['output_sha256'],raw_wall_nanoseconds=samples,median_ms=statistics.median(samples)/1e6,
                    instrumented_first_call_ns=first_ns,instrumented_warmup_ns=warmups,calibration=calibration,
                    output_independent_of_inputs=True))
                require(torch.cuda.max_memory_allocated() <= protocol['maximum_Torch_GPU_allocated_bytes']
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['maximum_worker_RSS_bytes'],
                        'Preregistered pressure timing resource bound exceeded')
                if execution_index%32 == 31:
                    save(output/'result.json',result)
                    print(fit_id,phase,execution_index+1,'original pressure requests timed',flush=True)
            observation = resources()
            result['resource_observations'].append(observation)
            require(not observation['other_compute_pids'],'Other GPU compute observed; preserve as contended')
        require(observer.closed and observer.handle is None and not observer.forbidden_attempts,
                'Timing observer must close cleanly with no forbidden forwards')
        verify_sources(protocol)
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT)
        result.update(actual_bound_lab_Python_modules=coverage['lab_python'],runtime_virtual_modules=coverage['runtime_virtual_modules'],
            unbound_lab_Python_modules=[name for name in coverage['lab_python'] if name not in protocol['source_sha256']])
        require(not result['unbound_lab_Python_modules'],'Imported pressure timing source absent from protocol')
        result.update(status='original_train_pressure_GPU_timing_qualified',finished_unix=time.time(),active_case=None,cases=128,
            task_calls_timed=128*protocol['samples_per_case'],logical_calls=128*(1+protocol['warmups_per_case']+
                protocol['samples_per_case']+2*protocol['calibration_pairs_per_case']),device_observation=observer.evidence(),
            all_output_hashes_exact=True,learned_tensors_exact_and_unchanged=True,captured_architecture_unchanged=True,
            primary_observer_calls=0,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            case_median_median_ms=statistics.median(row['median_ms'] for row in result['rows']),
            worst_case_median_ms=max(row['median_ms'] for row in result['rows']),wall_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(status='original_train_pressure_timing_adapter_gap',error_type=type(error).__name__,error=str(error),
            device_observation=observer.evidence() if observer else None,scientific_accuracy_failure_claim=False)
        raise
    finally:
        save(output/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.fit_id,args.phase)
