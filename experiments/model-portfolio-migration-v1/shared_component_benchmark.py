"""Original shared response, complete native RHS-to-pressure CUDA requests."""
import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file, selected_inventory
from cuda_forward_observer import CUDAForwardObserver
from pressure_portfolio_gpu_timing import resources


def verify_sources(protocol):
    for name,sha in protocol['source_sha256'].items():
        with (ROOT/name).open('rb') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest() == sha, 'Frozen modal source changed: '+name)


def imported(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(protocol_path,fit_id,mode,phase):
    current_publication(ROOT)
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rheon.shared-component-GPU-benchmark.v1'
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda' and phase in (0,1)
            and mode in ('unprepared','prepared'), 'Explicit shared CUDA/cache protocol required')
    verify_sources(protocol)
    contract = read(ROOT/protocol['contract']['path'])
    require(binding(ROOT/protocol['contract']['path']) == protocol['contract'], 'Modal contract changed')
    selected = protocol['selected']
    require(selected['fit_item_id'] == fit_id, 'Original selected shared fit required')
    material = read(ROOT/contract['material']['path'])
    require(binding(ROOT/contract['material']['path']) == contract['material'], 'Original material changed')
    entry = next(row for row in material['models'] if row['fit_item_id'] == fit_id)
    require(entry['task'] == 'pressure_component' and entry['serialized_tensor_elements'] == 4
            and selected['weights'] == entry['outputs']['weights.npz'], 'Exact original selected modal fit required')
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values()]:
        observed_file(ROOT,spec['path'],spec)
    inventory,_ = selected_inventory(read(ROOT/entry['original_receipt']['path']),
        read(ROOT/entry['outputs']['history.json']['path']),selected['weights']['sha256'])
    require(inventory == entry['selected_inventory'] and entry['parameter_dtypes'] == ['<f8'], 'Original selected F64 parameters required')
    folder = ROOT/protocol['output_root']/f'{fit_id}-{mode}-phase{phase}'
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh original modal worker required; no retry')
    folder.mkdir(parents=True)
    os.sched_setaffinity(0,{protocol['host_cpu']})
    result = dict(status='starting_original_shared_GPU_phase',pid=os.getpid(),created_unix=time.time(),
        fit_item_id=fit_id,mode=mode,phase=phase,protocol=binding(protocol_path),rows=[],resource_observations=[],
        inference_device='cuda',timing_units='ms/native_RHS_pressure_request',physical_simulation_steps=0,
        primary_forward_observer_enabled=False,CPU_neural_forwards=0,reference_labels_forwarded=False,
        learned_response_cache=mode == 'prepared',new_parameter_updates=0,new_reference_generation=False,full_portfolio_complete=False)
    save(folder/'result.json',result)
    observer = None
    began = time.monotonic()
    try:
        import numpy as np
        import torch
        from modal_component_data import load_fields, digest
        import score_pressure_portfolio as common_score
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU'], 'Registered real GPU required')
        require(not torch.is_autocast_enabled('cuda'), 'Autocast must remain disabled')
        observer = CUDAForwardObserver(torch);observer.enable('construction')
        geometries = list({row['key']:dict(key=row['key'],counts_zyx=row['counts_zyx']) for row in contract['rows']}.values())
        with torch.device('cuda'):
            model_source = imported('original_shared_model',ROOT/selected['model']['path'])
            basis_source = imported('original_shared_basis',ROOT/selected['basis']['path'])
            model = model_source.SharedSpectralResponse(device='cuda').eval().requires_grad_(False)
        with np.load(ROOT/selected['weights']['path'],allow_pickle=False) as archive:
            require(set(archive.files) == set(inventory) == set(model.state_dict()), 'Complete modal parameter inventory required')
            loaded = {}
            for name in archive.files:
                value = archive[name];spec = inventory[name]
                require(value.dtype.str == spec['dtype'] == '<f8' and list(value.shape) == spec['shape']
                        and digest(value) == spec['sha256'], 'Original modal tensor differs')
                loaded[name] = torch.from_numpy(value.copy()).to('cuda')
            model.load_state_dict(loaded,strict=True)
        architecture = repr(model)
        def check_model():
            require(repr(model) == architecture, 'Captured modal architecture changed')
            for name,value in model.state_dict().items():
                require(value.device.type == 'cuda' and value.dtype == torch.float64
                        and digest(value.detach().cpu().numpy()) == inventory[name]['sha256'], 'Original modal tensors changed')
        check_model()
        common_score.operator_check()
        fields = load_fields(contract,protocol)
        with torch.inference_mode():
            axes = {row['key']:basis_source.bases(tuple(row['counts_zyx']),'cuda') for row in geometries}
        basis_hashes = {key:[digest(value.cpu().numpy()) for value in group] for key,group in axes.items()}
        responses,preparation = {},{}
        if mode == 'prepared':
            observer.set_phase('response_preparation')
            with torch.inference_mode():
                for row in geometries:
                    key = row['key']
                    torch.cuda.synchronize();tick = time.perf_counter_ns()
                    response = model.response_grid(tuple(row['counts_zyx']))
                    torch.cuda.synchronize();elapsed = time.perf_counter_ns()-tick
                    require(response.device.type == 'cuda' and response.dtype == torch.float64
                            and tuple(response.shape) == tuple(row['counts_zyx']) and bool(response.isfinite().all())
                            and bool((response.flatten()[1:] > 0).all()), 'Finite positive nonDC learned CUDA response required')
                    responses[key] = response
                    preparation[key] = dict(wall_nanoseconds=elapsed,response_sha256=digest(response.cpu().numpy()),
                        shape=list(response.shape),inference_device='cuda')
        prior = None
        if phase == 1 or mode == 'prepared':
            previous = read(folder.parent/f'{fit_id}-unprepared-phase0/result.json')
            require(previous['status'] == 'original_shared_GPU_phase_qualified'
                    and not Path(f'/proc/{previous["pid"]}').exists(), 'Original phase0 must be terminal')
            prior = {row['case_index']:row for row in previous['rows']}
        result.update(GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            host_cpu=protocol['host_cpu'],threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            architecture=architecture,weights=selected['weights'],basis_hashes=basis_hashes,
            learned_response_preparation=preparation,setup_wall_seconds=time.monotonic()-began)

        def complete(rhs,h,rho,key):
            require(rhs.dtype.str == '<f8' and rhs.shape == tuple(len(axis) for axis in axes[key])
                    and np.isfinite(rhs).all() and 0 < h < math.inf and 0 < rho < math.inf, 'Finite owned native RHS/units required')
            q = torch.from_numpy(np.ascontiguousarray(rhs))[None,None].to('cuda')*(rho/h)
            response = responses.get(key)
            if response is not None:
                require(response.device.type == 'cuda' and response.dtype == torch.float64, 'Prepared CUDA F64 response required')
            value = model.decode(q,axes[key],response=response)
            require(value.device.type == 'cuda' and value.dtype == torch.float64 and value.shape == q.shape,
                    'Complete captured CUDA F64 pressure required')
            owned = value.detach().to('cpu',copy=True).numpy()[0,0]
            torch.cuda.synchronize()
            return owned

        predictions = {}
        with torch.inference_mode():
            for position,index in enumerate(range(144) if phase == 0 else reversed(range(144))):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Finite worker time bound exceeded')
                if position%36 == 0:
                    observation = resources();result['resource_observations'].append(observation)
                    require(not observation['other_compute_pids'], 'Competing compute observed; retain failure')
                case = fields[index];identity = case['identity'];rhs = case['rhs'].copy()
                observer.set_phase('first_call')
                tick = time.perf_counter_ns();prediction = complete(rhs,case['h'],case['rho'],identity['key']);first = time.perf_counter_ns()-tick
                require(np.isfinite(prediction).all() and not np.shares_memory(prediction,rhs), 'Finite independently owned pressure required')
                expected_sha = digest(prediction)
                if prior:
                    require(prior[index]['output_sha256'] == expected_sha, 'Exact cross-phase modal outputs differ')
                if phase == 0 and mode == 'unprepared':
                    predictions[f'case{index:03d}'] = prediction.copy()
                p = torch.from_numpy(prediction.copy())[None,None]
                reference = torch.from_numpy(case['reference'])[None,None]
                target_rhs = torch.from_numpy(case['rhs'])[None,None]
                h = torch.tensor([case['h']],dtype=torch.float64,device='cpu')
                rho = torch.tensor([case['rho']],dtype=torch.float64,device='cpu')
                residual = common_score.equation(p,h,rho)-target_rhs
                raw_difference = p-reference;difference = raw_difference-raw_difference.mean()
                centered = reference-reference.mean()
                rhs_norm = float(target_rhs.norm());reference_norm = float(centered.norm())
                metrics = dict(common_equation_relative_l2=float(residual.norm())/max(rhs_norm,1e-30),
                    pressure_mean_free_relative_l2=float(difference.norm())/max(reference_norm,1e-30),
                    pressure_absolute_rms=float(difference.square().mean().sqrt()),pressure_absolute_max=float(difference.abs().max()),
                    pressure_gauge_absolute_rms=float(raw_difference.square().mean().sqrt()),
                    pressure_gauge_absolute_max=float(raw_difference.abs().max()),equation_rhs_l2=rhs_norm,
                    pressure_reference_mean_free_l2=reference_norm,equation_floor_active=rhs_norm < 1e-30,
                    pressure_floor_active=reference_norm < 1e-30)
                require(all(math.isfinite(value) for value in metrics.values()), 'Finite pressure scores required')
                def sample():
                    tick = time.perf_counter_ns();value = complete(rhs,case['h'],case['rho'],identity['key']);elapsed = time.perf_counter_ns()-tick
                    require(elapsed > 0 and digest(value) == expected_sha and np.isfinite(value).all()
                            and not np.shares_memory(value,rhs), 'Repeated modal result differs')
                    return elapsed
                observer.set_phase('warmup');warmups = [sample() for _ in range(protocol['warmups_per_case'])]
                before = dict(observer.calls)
                with observer.unobserved('primary'):
                    samples = [sample() for _ in range(protocol['samples_per_case'])]
                    require(observer.handle is None and dict(observer.calls) == before, 'Primary observer unexpectedly attached')
                pairs = []
                for number in range(protocol['calibration_pairs_per_case']):
                    order = ['on','off'] if (phase+index+number)%2 == 0 else ['off','on']
                    pair = dict(pair_index=number,order=order)
                    for calibration_mode in order:
                        if calibration_mode == 'on':
                            observer.set_phase('calibration_on');pair['on_ns'] = sample()
                        else:
                            with observer.unobserved('calibration_off'):
                                pair['off_ns'] = sample()
                    pair['on_minus_off_ns'] = pair['on_ns']-pair['off_ns'];pairs.append(pair)
                require(digest(rhs) == case['rhs_sha256'], 'Caller native RHS mutated')
                result['rows'].append(dict(case_index=index,identity=identity,input_rhs_sha256=case['rhs_sha256'],
                    reference_pressure_sha256=case['reference_sha256'],output_sha256=expected_sha,metrics=metrics,
                    raw_wall_nanoseconds=samples,median_ms=statistics.median(samples)/1e6,
                    instrumented_first_call_ns=first,instrumented_warmup_ns=warmups,calibration=pairs))
                require(torch.cuda.max_memory_allocated() <= protocol['maximum_CUDA_allocated_bytes']
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['maximum_RSS_bytes'], 'Modal resource bound exceeded')
                if position%36 == 35:
                    save(folder/'result.json',result)
        observation = resources();result['resource_observations'].append(observation)
        require(not observation['other_compute_pids'], 'Competing compute observed')
        check_model()
        require(basis_hashes == {key:[digest(value.cpu().numpy()) for value in group] for key,group in axes.items()}, 'Prescribed bases mutated')
        require(all(preparation[key]['response_sha256'] == digest(value.cpu().numpy()) for key,value in responses.items()),
                'Prepared learned response mutated')
        observer.close()
        if phase == 0 and mode == 'unprepared':
            archive = folder/'pressure-predictions.npz';np.savez_compressed(archive,**predictions)
            result['prediction_archive'] = binding(archive)
        else:
            result['prediction_archive'] = previous['prediction_archive']
        verify_sources(protocol)
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT)
        require(not [name for name in coverage['lab_python'] if name not in protocol['source_sha256']], 'Unbound actual modal imports')
        result.update(status='original_shared_GPU_phase_qualified',finished_unix=time.time(),cases=144,
            learned_tensors_and_architecture_unchanged=True,primary_observer_calls=0,source_coverage=coverage,
            device_observation=observer.evidence(),primary_calls=1440,logical_calls=3456,calibration_pairs=576,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_wall_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(status='original_shared_GPU_adapter_gap',error_type=type(error).__name__,error=str(error),scientific_accuracy_failure_claim=False)
        raise
    finally:
        if observer:
            observer.close()
            result['device_observation'] = observer.evidence()
        save(folder/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    parser.add_argument('--mode',choices=('unprepared','prepared'),required=True)
    parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    args = parser.parse_args();run(args.protocol.resolve(),args.fit_id,args.mode,args.phase)
