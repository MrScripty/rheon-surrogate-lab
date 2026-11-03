"""Serial CUDA-only construction and untimed common-case accuracy qualification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from full_state_gpu_timing import check_owned, cpu_views, field_sizes, hashes, numpy_outputs, resources
from inherited_generator_cuda import restore, route
from joint_generator_gpu_benchmark import score
from reduce_sharing_generator_gpu_benchmark import saved_probe_check
from review_water_learning_sequence import ROOT, binding, canonical, read, require
from run_joint_generator_gpu_benchmark import save, stop

RUN = ROOT/'runs/model-portfolio-migration-v1'
FIELDS = ('pressure','vx','vy','vz','tracer')


def inputs(protocol_path):
    protocol = read(protocol_path)
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Admitted constructor qualification input changed')
    require(binding(ROOT/protocol['cases']['path']) == protocol['cases'],'Original common cases changed')
    return protocol,read(ROOT/protocol['cases']['path'])


def worker(protocol_path,identifier,output):
    require(not output.exists() and output.is_relative_to(RUN)
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda','Fresh admitted CUDA worker required')
    protocol,prepared = inputs(protocol_path)
    selected = [m for m in protocol['models'] if m['identifier'] == identifier]
    require(len(selected) == 1,'Exactly one source-bound inherited model required')
    selected = selected[0]
    before = resources()
    require(not before['other_compute_pids'],'Another GPU compute process is active; qualification not dispatched')
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
    torch.manual_seed(read(ROOT/selected['configuration']['path'])['seed'])
    started = time.monotonic()
    result = dict(status='original_inherited_qualification_running',pid=os.getpid(),created_unix=time.time(),
        identifier=identifier,fit_item_id=selected['fit_item_id'],K=selected['K'],protocol=binding(protocol_path),
        resources_before=before,rows=[],CPU_neural_forwards=0,new_reference_generation=False,
        output_arrays_written=False,native_mutations=0,new_fit=False,inference_latency_qualified=False,
        concurrent_native_metadata_work=True,full_portfolio_complete=False)
    evidence = None
    try:
        require(len(prepared['cases']) == 64,'All64original physical cases required')
        norm = read(ROOT/selected['normalization']['path'])
        with restore(selected) as (network,evidence):
            for case in prepared['cases']:
                require(time.monotonic()-started < protocol['worker_deadline_seconds'],'Original worker deadline exceeded')
                sizes = field_sizes(case['scene']['counts'])
                path = ROOT/case['current']['path']
                require(binding(path)['sha256'] == case['current']['sha256'],'Original current native state changed')
                with np.load(path,allow_pickle=False) as arrays:
                    native = {n:arrays[n].copy() for n in FIELDS}
                initial = hashes(native)
                branch = route(network.pressure_core,case['scene']['counts'])
                previous,expected,metrics = None,None,[]
                for repetition in range(2):
                    with torch.inference_mode():
                        generated = network.predict(cpu_views(np,torch,native,sizes),case['scene'],case['start'],norm)
                        owned = [{n:r[n].to('cpu',copy=True).contiguous() for n in FIELDS} for r in generated]
                        returned = numpy_outputs(torch,owned,selected['K'],sizes)
                        torch.cuda.synchronize()
                    require(hashes(native) == initial and all(np.isfinite(r[n]).all() for r in returned for n in FIELDS),
                            'Prediction mutated input or returned nonfinite fields')
                    check_owned(np,returned,native,previous)
                    actual = [hashes(r) for r in returned]
                    if expected is not None:
                        require(actual == expected,'Repeated complete physical fields changed bytes')
                    else:
                        expected = actual
                        for lead,row in enumerate(returned,1):
                            reference = case['reference_leads'][lead-1]
                            path = ROOT/reference['path']
                            require(binding(path)['sha256'] == reference['sha256'],'Original future reference changed')
                            with np.load(path,allow_pickle=False) as arrays:
                                truth = {n:arrays[n] for n in (*FIELDS,'rhs')}
                                metrics.append(dict(lead=lead,time_s=reference['time_s'],values=score(np,row,truth,case['scene'])))
                    previous = returned
                result['rows'].append(dict(identifier=case['identifier'],start=case['start'],K=selected['K'],
                    counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],
                    output_dt_s=1/60,block_advance_s=selected['K']/60,pressure_route=branch,
                    complete_output_hashes=expected,metrics=metrics,repeat_and_ownership_checks_passed=True))
            require(len(result['rows']) == 64,'Complete common-case population required')
            result['original_saved_probe_metric_comparison'] = saved_probe_check(selected,prepared,result)
        require(evidence['scoped_hooks_removed'] and evidence['CPU_neural_forwards'] == evidence['forbidden_forward_attempts'] == 0
                and evidence['neural_calls_by_phase']['prediction']['BundleNet'] == 128,
                'CUDA-only setup/prediction,exact work and hook cleanup required')
        result.update(status='all64_cases_and_original_inherited_CUDA_constructor_qualified',
            constructor=evidence,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),resources_after=resources(),
            GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            common_reference_method=prepared['reference_method'],reference_binary_sha256=prepared['reference_native_probe_binary_sha256'],
            exposure='All16consumed fitting scenes,four starts;no independent holdout or own-history drift claim.')
        inputs(protocol_path)
        print(identifier,'CUDA setup and all64native cases qualified;no inference-speed claim.',flush=True)
    except BaseException as error:
        result.update(status='original_inherited_worker_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        if evidence is not None:
            result['constructor'] = evidence
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        output.write_bytes(canonical(result)+b'\n')


def family(protocol_path):
    protocol,_ = inputs(protocol_path)
    output = ROOT/protocol['output']
    require(not output.exists() and output.parent == RUN,'Fresh family directory required;reconcile any existing owner')
    before = resources()
    require(not before['other_compute_pids'],'Another GPU compute process is active;family not dispatched')
    runtime = read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='starting_original_family',pid=os.getpid(),created_unix=time.time(),protocol=binding(protocol_path),
        active=None,completed=[],resources_before=before,CPU_neural_forwards=0,inference_latency_qualified=False,
        native_mutations=0,full_portfolio_complete=False)
    status = output/'status.json'
    save(status,state)
    child = None
    try:
        for model in protocol['models']:
            path = output/(model['identifier']+'.json')
            log = path.with_suffix('.log')
            require(not path.exists() and not log.exists(),'Preserve original fit result and log')
            with log.open('xb') as stream:
                child = subprocess.Popen([runtime['python'],'-B',str(Path(__file__)),'worker','--protocol',str(protocol_path),
                    '--identifier',model['identifier'],'--output',str(path)],cwd=ROOT,env=environment(runtime,'cuda'),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='original_worker_running',active=dict(identifier=model['identifier'],pid=child.pid,
                    result=str(path.relative_to(ROOT)),log=str(log.relative_to(ROOT))),updated_unix=time.time())
                save(status,state)
                print('Started',model['identifier'],'PID',child.pid,flush=True)
                try:
                    code = child.wait(timeout=protocol['worker_deadline_seconds']+30)
                finally:
                    stop(child)
            require(code == 0 and path.exists(),'Original worker failed;inspect retained evidence without replay')
            result = read(path)
            require(result['status'] == 'all64_cases_and_original_inherited_CUDA_constructor_qualified'
                    and result['protocol'] == state['protocol'],'Original worker outcome or protocol differs')
            state['completed'].append(dict(identifier=model['identifier'],result=binding(path)))
            state.update(status='original_worker_completed',active=None,updated_unix=time.time())
            save(status,state)
            child = None
        state.update(status='all16_original_workers_complete_reduction_pending',active=None,finished_unix=time.time())
        save(status,state)
    except BaseException as error:
        if child is not None:
            stop(child)
        state.update(status='original_family_requires_reconciliation',active=None,error_type=type(error).__name__,error=str(error),updated_unix=time.time())
        save(status,state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('family','worker'))
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--identifier')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.mode == 'family':
        family(args.protocol.resolve())
    else:
        worker(args.protocol.resolve(),args.identifier,args.output.resolve())
