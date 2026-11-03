"""Unchanged F64 cell native entry on the35generators'64legacy-reference cases.

Every output advances1/60second through one original N1 stage. The N8-only
coordinate sampler is not exercised by this regime. No future field enters
preparation or prediction. All native fields return in fresh owned CPU storage.
"""
import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import time

from complete_generator_gpu_benchmark import sample
from full_state_gpu_timing import check_owned, cpu_views, field_sizes, hashes, numpy_outputs, resources
from joint_generator_gpu_benchmark import score
from review_water_learning_sequence import ROOT, binding, canonical, read, require

FIELDS = ('pressure','vx','vy','vz','tracer')


def check_inputs(protocol):
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Preregistered F64 input changed: '+spec['path'])


def imported_sources(protocol):
    admitted = {s['path']:s for s in protocol['dependencies']}
    paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name is None:
            continue
        path = Path(name).resolve()
        if any(path.is_relative_to(ROOT/root) for root in ('experiments','docs')):
            relative = str(path.relative_to(ROOT))
            require(relative in admitted and binding(path) == admitted[relative],
                    'Imported project source was not preregistered: '+relative)
            paths.add(relative)
    return sorted(paths)


def run(protocol_path,k,phase,output):
    require(not output.exists() and output.is_relative_to(ROOT/'runs/model-portfolio-migration-v1')
            and k in (1,2,3,4) and phase in (0,1) and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda',
            'Fresh admitted CUDA worker required')
    protocol = read(protocol_path)
    check_inputs(protocol)
    prepared = read(ROOT/protocol['cases']['path'])
    before = resources()
    require(not before['other_compute_pids'],'Another CUDA compute process is active')
    os.sched_setaffinity(0,{protocol['host_cpu']})
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    begun = time.monotonic()
    result = dict(status='original_F64_worker_running',pid=os.getpid(),K=k,phase=phase,
        assembly_execution_id='original_cell_coordinate_N1_K'+str(k),protocol=binding(protocol_path),
        resources_before=before,rows=[],CPU_neural_forwards=0,guarded_module_calls=0,forbidden_forward_attempts=0,
        primary_validation_hook_active=False,new_reference_generation=False,new_fit=False,native_mutations=0,
        output_arrays_written=False,reference_fields_passed_to_model=False,full_portfolio_complete=False)
    model = None

    def guard(owner,args):
        tensors = [v for v in args if isinstance(v,torch.Tensor)]
        if not tensors or any(v.device.type != 'cuda' for v in tensors):
            result['forbidden_forward_attempts'] += 1
            raise ValueError('CPU/unclassified neural forward forbidden')
        result['guarded_module_calls'] += 1

    @contextmanager
    def guarded_setup():
        handle = torch.nn.modules.module.register_module_forward_pre_hook(guard)
        try:
            yield
        finally:
            handle.remove()

    try:
        sys.path.insert(0,str(ROOT/'experiments/full-ai-data-diversity-v1'))
        import support
        from diagnose_causal_composition import inventory
        module = importlib.import_module('cell_coordinate_pressure_graph_v1')
        args = protocol['constructor_arguments']
        with guarded_setup():
            model = module.CellCoordinatePressureNativeEntry(*args)
            torch.cuda.synchronize()
        parameters = inventory(model)
        result.update(setup_wall_seconds=time.monotonic()-begun,model_identity=model.asset_identity,
            original_parameter_inventory=parameters,imported_project_sources=imported_sources(protocol))
        torch.cuda.reset_peak_memory_stats()
        for case in (prepared['cases'] if phase == 0 else list(reversed(prepared['cases']))):
            require(time.monotonic()-begun < protocol['worker_deadline_seconds'],'Admitted F64 worker deadline exceeded')
            path = ROOT/case['current']['path']
            require(binding(path)['sha256'] == case['current']['sha256'],'Original current state changed')
            with np.load(path,allow_pickle=False) as archive:
                native = {n:archive[n].copy() for n in FIELDS}
            sizes = field_sizes(case['scene']['counts'])
            input_hashes = hashes(native)
            tick = time.monotonic()
            with guarded_setup():
                model.prepare_native(cpu_views(np,torch,native,sizes),case['scene'],case['start'],k,1)
                torch.cuda.synchronize()
            setup_seconds = time.monotonic()-tick
            previous,expected,metrics = None,None,None

            def complete():
                with torch.inference_mode():
                    returned = model.predict_native(cpu_views(np,torch,native,sizes),case['scene'],case['start'],k,1)
                    return numpy_outputs(torch,returned,k,sizes)

            def measure(hook=None):
                nonlocal previous,expected,metrics
                elapsed,returned = sample(complete,torch,hook)
                require(hashes(native) == input_hashes and all(np.isfinite(r[n]).all() for r in returned for n in FIELDS),
                        'Mutated input or nonfinite F64 output')
                check_owned(np,returned,native,previous)
                actual = [hashes(r) for r in returned]
                if expected is None:
                    expected,metrics = actual,[]
                    for lead,row in enumerate(returned,1):
                        ref = case['reference_leads'][lead-1]
                        target = ROOT/ref['path']
                        require(binding(target)['sha256'] == ref['sha256'],'Original future reference changed')
                        with np.load(target,allow_pickle=False) as archive:
                            truth = {n:archive[n] for n in (*FIELDS,'rhs')}
                            metrics.append(dict(lead=lead,time_s=ref['time_s'],values=score(np,row,truth,case['scene'])))
                else:
                    require(actual == expected,'Repeated original F64 output bytes changed')
                previous = returned
                return elapsed

            first = measure(guard)
            warmups = [measure(guard) for _ in range(protocol['warmups'])]
            primary = [measure() for _ in range(protocol['samples'])]
            result['rows'].append(dict(identifier=case['identifier'],start=case['start'],K=k,
                counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],
                output_dt_s=1/60,block_advance_s=k/60,actual_internal_advances=k,
                graph_geometry_setup_seconds=setup_seconds,first_call_block_ms=first,warmup_block_ms=warmups,
                block_ms=primary,per_output_ms=[v/k for v in primary],metrics=metrics,output_hashes=expected,
                repeat_and_ownership_checks_passed=True))
        require(len(result['rows']) == 64 and inventory(model) == parameters,'Complete64cases and unchanged learned tensors required')
        result.update(status='all64_original_cases_F64_N1_GPU_measured',GPU=torch.cuda.get_device_name(),
            torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),imported_project_sources=imported_sources(protocol),
            original_reference_binary_sha256=prepared['reference_native_probe_binary_sha256'])
        require(result['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes'],'Admitted F64 CUDA allocation exceeded')
        check_inputs(protocol)
        print('Original F64 cell K',k,'phase',phase,'all64native cases complete.',flush=True)
    except BaseException as error:
        result.update(status='original_F64_worker_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if model is not None:
                model.close()
            result['cleanup_complete'] = True
        finally:
            result['elapsed_seconds'] = time.monotonic()-begun
            result['resources_after'] = resources()
            output.write_bytes(canonical(result)+b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--K',type=int,required=True)
    parser.add_argument('--phase',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.K,args.phase,args.output.resolve())
