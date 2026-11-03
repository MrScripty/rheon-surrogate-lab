"""One common uninstrumented CPU-input/all-K-CPU-output CUDA generator boundary.

Each worker owns one original fit and process phase. CUDA guards cover setup,
first calls and warmups; primary samples have no validation forward hook.
Separate alternating hook-on/off pairs measure instrumentation in the same call.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

from full_state_gpu_timing import check_owned, cpu_views, field_sizes, hashes, numpy_outputs, resources
from inherited_generator_cuda import restore as inherited_restore, route
from joint_generator_gpu_benchmark import score
from reduce_sharing_generator_gpu_benchmark import saved_probe_check
from review_water_learning_sequence import ROOT, binding, canonical, read, require

FIELDS = ('pressure','vx','vy','vz','tracer')


def sample(call, torch, hook=None, clock=time.perf_counter_ns):
    """Install/remove observation outside the interval, including on exceptions."""
    module = torch.nn.modules.module
    require(not module._global_forward_pre_hooks and not module._global_forward_hooks,
            'Unexpected global forward instrumentation')
    handle = module.register_module_forward_pre_hook(hook) if hook is not None else None
    try:
        torch.cuda.synchronize()
        tick = clock()
        returned = call()
        torch.cuda.synchronize()
        return (clock()-tick)/1e6,returned
    finally:
        if handle is not None:
            handle.remove()


def inventory(network):
    return {n:dict(shape=list(v.shape),dtype='<f4',
        sha256=hashlib.sha256(v.detach().cpu().numpy().astype('<f4',copy=False).tobytes()).hexdigest())
        for n,v in network.state_dict().items()}


def restore(model):
    import torch
    receipt = read(ROOT/model['receipt']['path'])
    config = read(ROOT/model['configuration']['path'])
    torch.manual_seed(config['seed'])
    if 'readiness' in model:
        # Preserve the already-qualified CUDA constructor, then remove every
        # scoped hook/default-device context before the common prediction call.
        with inherited_restore(model) as (network,evidence):
            pass
        require(evidence['scoped_hooks_removed'], 'Constructor hooks must be removed')
        return network,evidence
    sys.path.insert(0,str(ROOT/model['project_path']))
    spec = importlib.util.spec_from_file_location('captured_plain_trainer',ROOT/model['trainer']['path'])
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    def no_constructor_forward(owner,args):
        raise ValueError('Unexpected neural execution in a plain constructor')

    handle = torch.nn.modules.module.register_module_forward_pre_hook(no_constructor_forward)
    try:
        if hasattr(trainer,'construct_network'):
            network = trainer.construct_network(config,'cuda',statistics_path=ROOT/model['project_path']/'step-scales.json')
        elif config['schema'] == 'rheon-temporal-fit-sharing-v1':
            network = trainer.BundleNet(model['K'],config['width'],config['architecture']).to('cuda')
        else:
            network = trainer.BundleNet(model['K'],config['width']).to('cuda')
        trainer.load_weights(network,ROOT/model['weights']['path'],receipt['final_inventory'])
    finally:
        handle.remove()
    network.eval()
    return network,dict(CPU_neural_forwards=0,constructor_neural_forwards=0,
        precision_variant='Original captured constructor without neural setup queries; exact final learned tensors.',
        scoped_hooks_removed=True)


def verify_inputs(protocol,model):
    for spec in [*protocol['dependencies'],*model['bound_inputs']]:
        require(binding(ROOT/spec['path']) == spec,'Preregistered benchmark input changed: '+spec['path'])


def run(protocol_path,fit_id,phase,output):
    require(not output.exists() and output.is_relative_to(ROOT/'runs/model-portfolio-migration-v1')
            and phase in (0,1) and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda','Fresh admitted CUDA worker required')
    protocol = read(protocol_path)
    matches = [m for m in protocol['models'] if m['fit_item_id'] == fit_id]
    require(len(matches) == 1,'Exactly one original fit identity required')
    model = matches[0]
    verify_inputs(protocol,model)
    prepared = read(ROOT/protocol['cases']['path'])
    before = resources()
    require(not before['other_compute_pids'],'Another CUDA compute process is active; worker not dispatched')
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
    started = time.monotonic()
    result = dict(status='original_worker_running',pid=os.getpid(),fit_item_id=fit_id,identifier=model['identifier'],
        K=model['K'],phase=phase,protocol=binding(protocol_path),resources_before=before,rows=[],
        CPU_neural_forwards=0,forbidden_forward_attempts=0,guarded_module_calls=0,guarded_network_calls=0,
        logical_primary_calls=0,logical_calibration_off_calls=0,logical_complete_calls=0,
        primary_validation_hook_active=False,new_reference_generation=False,native_mutations=0,
        output_arrays_written=False,full_portfolio_complete=False)
    try:
        network,construction = restore(model)
        require(all(v.device.type == 'cuda' for v in [*network.parameters(),*network.buffers()]),
                'All original parameters/buffers must be CUDA')
        receipt = read(ROOT/model['receipt']['path'])
        require(inventory(network) == receipt['final_inventory'],'Original learned tensors differ')
        result['constructor'] = construction
        torch.cuda.synchronize()
        result['setup_wall_seconds'] = time.monotonic()-started
        torch.cuda.reset_peak_memory_stats()
        norm = read(ROOT/model['normalization']['path'])

        # This is the original I47 callback's input test and counter work, not
        # a stronger diagnostic that would measure a different hook overhead.
        def guard(owner,args):
            tensors = [v for v in args if isinstance(v,torch.Tensor)]
            if not tensors or any(v.device.type != 'cuda' for v in tensors):
                result['forbidden_forward_attempts'] += 1
                raise ValueError('CPU/unclassified neural forward forbidden')
            result['guarded_module_calls'] += 1
            result['guarded_network_calls'] += owner is network

        for case in (prepared['cases'] if phase == 0 else list(reversed(prepared['cases']))):
            require(time.monotonic()-started < protocol['worker_deadline_seconds'],'Admitted worker deadline exceeded')
            sizes = field_sizes(case['scene']['counts'])
            current = ROOT/case['current']['path']
            require(binding(current)['sha256'] == case['current']['sha256'],'Original current native frame changed')
            with np.load(current,allow_pickle=False) as arrays:
                native = {n:arrays[n].copy() for n in FIELDS}
            input_hashes = hashes(native)
            expected,previous,metrics = None,None,None

            def predict():
                with torch.inference_mode():
                    generated = network.predict(cpu_views(np,torch,native,sizes),case['scene'],case['start'],norm)
                    owned = [{n:r[n].to('cpu',copy=True).contiguous() for n in FIELDS} for r in generated]
                    return numpy_outputs(torch,owned,model['K'],sizes)

            def measure(hook=None):
                nonlocal expected,previous,metrics
                elapsed,returned = sample(predict,torch,hook)
                result['logical_complete_calls'] += 1
                require(hashes(native) == input_hashes and all(np.isfinite(r[n]).all() for r in returned for n in FIELDS),
                        'Input mutation or nonfinite output')
                check_owned(np,returned,native,previous)
                actual = [hashes(r) for r in returned]
                if expected is None:
                    expected,metrics = actual,[]
                    for lead,row in enumerate(returned,1):
                        reference = case['reference_leads'][lead-1]
                        path = ROOT/reference['path']
                        require(binding(path)['sha256'] == reference['sha256'],'Original target native frame changed')
                        with np.load(path,allow_pickle=False) as arrays:
                            truth = {n:arrays[n] for n in (*FIELDS,'rhs')}
                            metrics.append(dict(lead=lead,time_s=reference['time_s'],values=score(np,row,truth,case['scene'])))
                else:
                    require(actual == expected,'Repeated physical output bytes changed across hook modes')
                previous = returned
                return elapsed

            first = measure(guard)
            warmups = [measure(guard) for _ in range(protocol['warmups'])]
            primary = [measure() for _ in range(protocol['samples'])]
            result['logical_primary_calls'] += len(primary)
            calibration = []
            for pair in range(protocol['calibration_pairs']):
                order = ('on','off') if (pair+phase)%2 == 0 else ('off','on')
                values = {mode:measure(guard if mode == 'on' else None) for mode in order}
                calibration.append(dict(order=list(order),hook_on_ms=values['on'],hook_off_ms=values['off']))
                result['logical_calibration_off_calls'] += 1
            result['rows'].append(dict(identifier=case['identifier'],start=case['start'],K=model['K'],
                counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],
                output_dt_s=1/60,block_advance_s=model['K']/60,
                pressure_route=route(network.pressure_core,case['scene']['counts']) if hasattr(network,'pressure_core') else 'captured_direct_predictor',
                first_call_block_ms=first,warmup_block_ms=warmups,block_ms=primary,
                per_output_ms=[v/model['K'] for v in primary],calibration=calibration,
                output_hashes=expected,metrics=metrics,repeat_and_ownership_checks_passed=True))
        require(len(result['rows']) == 64 and result['guarded_network_calls'] == 64*(1+protocol['warmups']+protocol['calibration_pairs'])
                and result['logical_primary_calls'] == 64*protocol['samples'], 'Exact common cases/work required')
        require(inventory(network) == receipt['final_inventory'],'Prediction changed original learned tensors')
        result['original_saved_probe_metric_comparison'] = saved_probe_check(model,prepared,result)
        result.update(status='all64_original_cases_uninstrumented_primary_and_paired_hooks_measured',
            resources_after=resources(),peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            original_reference_binary_sha256=prepared['reference_native_probe_binary_sha256'],
            exposure='All16consumed fitting scenes atfourstarts;no independent holdout or own-history drift claim.')
        require(result['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes'],'Admitted CUDA allocation exceeded')
        verify_inputs(protocol,model)
        print(fit_id,model['identifier'],'all64cases primary hooks-off / paired calibration complete.',flush=True)
    except BaseException as error:
        result.update(status='original_worker_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        output.write_bytes(canonical(result)+b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    parser.add_argument('--phase',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.fit_id,args.phase,args.output.resolve())
