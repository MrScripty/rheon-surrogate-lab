"""Original causal/factored F32 q heads on all2048retained one-step transitions."""
import argparse
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file, selected_inventory
from modal_component_benchmark import imported, verify_sources
from cuda_forward_observer import CUDAForwardObserver
from pressure_portfolio_gpu_timing import resources

INPUTS = ('velocity','known_forces','scalars')


def run(protocol_path,fit_id,phase):
    current_publication(ROOT);protocol = read(protocol_path);verify_sources(protocol)
    require(protocol['schema'] == 'rheon.causal-request-component-GPU-benchmark.v1' and phase in (0,1)
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Explicit original q-head CUDA protocol required')
    require(binding(ROOT/protocol['material']['path']) == protocol['material']
            and binding(ROOT/protocol['sampling_membership']['path']) == protocol['sampling_membership'], 'Original material/sampling replay changed')
    descriptor = next(row for row in protocol['fits'] if row['fit_item_id'] == fit_id)
    entry = next(row for row in read(ROOT/protocol['material']['path'])['models'] if row['fit_item_id'] == fit_id)
    require(entry['task'] == 'pressure_request_component' and entry['parameter_dtypes'] == ['<f4']
            and entry['serialized_tensor_elements'] in (3936,4385), 'Original one-step F32 q head required')
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),*entry['inputs'].values()]:
        observed_file(ROOT,spec['path'],spec)
    inventory,_ = selected_inventory(read(ROOT/entry['original_receipt']['path']),read(ROOT/entry['outputs']['history.json']['path']),entry['outputs']['weights.npz']['sha256'])
    require(inventory == entry['selected_inventory'], 'Original selected q-head inventory differs')
    folder = ROOT/protocol['output_root']/f'{fit_id}-phase{phase}'
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh original q-head phase required')
    folder.mkdir(parents=True);os.sched_setaffinity(0,{protocol['host_cpu']})
    result = dict(status='starting_original_causal_q_GPU_phase',pid=os.getpid(),phase=phase,created_unix=time.time(),
        fit_item_id=fit_id,protocol=binding(protocol_path),rows=[],resource_observations=[],inference_device='cuda',
        timing_units='ms/normalized_future_q_request',physical_simulation_steps=0,CPU_neural_forwards=0,
        reference_labels_forwarded=False,primary_forward_observer_enabled=False,new_parameter_updates=0,
        new_reference_generation=False,full_portfolio_complete=False)
    save(folder/'result.json',result);observer = None;began = time.monotonic()
    try:
        import numpy as np
        import torch
        from causal_request_component_data import load_original, owned_case
        from modal_component_data import digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False;torch.backends.cudnn.benchmark = False
        require(torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda') and torch.get_default_dtype() == torch.float32, 'Registered real GPU/F32configuration required')
        observer = CUDAForwardObserver(torch);observer.enable('construction')
        norm = read(ROOT/entry['inputs']['normalization']['path']);config = read(ROOT/entry['original_configuration']['path'])
        source = next(row for row in entry['sources'] if row['original_name'] == 'pressure_request_model.py')
        with torch.device('cuda'):
            module = imported('original_causal_q_model',ROOT/source['path'])
            model = module.CausalRequestNet(config['horizons'],norm['scales']['velocity'],norm['scales']['q']).eval().requires_grad_(False)
        with np.load(ROOT/entry['outputs']['weights.npz']['path'],allow_pickle=False) as archive:
            require(set(archive.files) == set(inventory) == set(model.state_dict()), 'Complete original q-head tensors required')
            values = {}
            for name in archive.files:
                value = archive[name];spec = inventory[name]
                require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                        and digest(value) == spec['sha256'], 'Original q-head tensor differs')
                values[name] = torch.from_numpy(value.copy()).to('cuda')
            model.load_state_dict(values,strict=True)
        architecture = repr(model)
        def check_model():
            require(repr(model) == architecture and model.horizons == 1 and model.velocity_scale == norm['scales']['velocity']
                    and model.q_scale == norm['scales']['q'], 'Original q-head architecture/scales changed')
            for name,value in model.state_dict().items():
                require(value.device.type == 'cuda' and value.dtype == torch.float32
                        and digest(value.detach().cpu().numpy()) == inventory[name]['sha256'], 'Original q-head tensors changed')
        check_model();observer.set_phase('data_preparation')
        data,manifest,loaded_norm,parents,caches = load_original(entry)
        require(loaded_norm == norm, 'Original decoder normalization differs')
        sampling = next(row for row in read(ROOT/protocol['sampling_membership']['path'])['fits'] if row['fit_item_id'] == fit_id)['counts_by_case']
        prior = None
        if phase == 1:
            previous = read(folder.parent/f'{fit_id}-phase0/result.json')
            require(previous['status'] == 'original_causal_q_GPU_phase_qualified'
                    and not Path(f'/proc/{previous["pid"]}').exists(), 'Original phase0must be terminal')
            prior = {row['case_index']:row for row in previous['rows']}
        result.update(GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            architecture=architecture,weights=entry['outputs']['weights.npz'],host_cpu=protocol['host_cpu'],
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            nn_conv3d_calls_per_forward=descriptor['nn_conv3d_calls_per_forward'],setup_wall_seconds=time.monotonic()-began)

        def complete(host,shape):
            require(set(host) == set(INPUTS) and host['velocity'].shape == host['known_forces'].shape == (1,3,*(n+1 for n in shape))
                    and host['scalars'].shape == (1,4), 'Original normalized input layouts required')
            forwarded = {}
            for name in INPUTS:
                value = host[name];require(value.dtype.str == '<f4' and np.isfinite(value).all(), 'Original finiteF32inputs required')
                forwarded[name] = torch.from_numpy(np.ascontiguousarray(value)).to('cuda')
            value = model(*(forwarded[name] for name in INPUTS))
            require(value.device.type == 'cuda' and value.dtype == torch.float32 and value.shape == (1,1,1,*shape), 'Original completeF32futureqrequired')
            owned = value.detach().to('cpu',copy=True).numpy();torch.cuda.synchronize();return owned

        def score(prediction,target):
            p = torch.from_numpy(prediction).double();t = torch.from_numpy(target).double();difference = p-t
            centered = difference-difference.mean();reference = t-t.mean();target_norm = float(t.norm());centered_norm = float(reference.norm())
            rms = float(difference.square().mean().sqrt());maximum = float(difference.abs().max())
            return dict(q_relative_l2=float(difference.norm())/max(target_norm,1e-30),
                q_mean_free_relative_l2=float(centered.norm())/max(centered_norm,1e-30),
                q_absolute_rms_normalized=rms,q_absolute_max_normalized=maximum,
                q_absolute_rms_physical=rms*norm['scales']['q'],q_absolute_max_physical=maximum*norm['scales']['q'],
                q_target_l2=target_norm,q_target_floor_active=target_norm < 1e-30,
                q_reference_centered_l2=centered_norm,q_reference_centered_floor_active=centered_norm < 1e-30)

        predictions = {}
        with torch.inference_mode():
            for position,index in enumerate(range(2048) if phase == 0 else reversed(range(2048))):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Finite q-head worker deadline exceeded')
                if position%256 == 0:
                    observation = resources();result['resource_observations'].append(observation)
                    require(not observation['other_compute_pids'], 'Other GPUcompute observed')
                identity,host,target = owned_case(data,parents,caches,index)
                identity.update(main_q_fit_visits=sampling[index],main_q_fit_seen=sampling[index] > 0,
                    sampling_support_region='0..120' if identity['current_state'] <= 120 else '121..127')
                shape = tuple(reversed(identity['counts_xyz']));input_hashes = {name:digest(value) for name,value in host.items()}
                observer.set_phase('first_call');tick = time.perf_counter_ns();prediction = complete(host,shape);first = time.perf_counter_ns()-tick
                require(np.isfinite(prediction).all() and not any(np.shares_memory(prediction,value) for value in host.values()), 'Finite independently owned q output required')
                expected = digest(prediction)
                if prior:
                    require(prior[index]['output_sha256'] == expected, 'Cross-phase q-head outputs differ')
                if phase == 0:
                    predictions[f'case{index:04d}'] = prediction.copy()
                metrics = score(prediction,target);require(all(math.isfinite(value) for value in metrics.values()), 'Finite q errors required')
                def sample():
                    tick = time.perf_counter_ns();value = complete(host,shape);elapsed = time.perf_counter_ns()-tick
                    require(elapsed > 0 and digest(value) == expected and np.isfinite(value).all()
                            and not any(np.shares_memory(value,item) for item in host.values()), 'Repeated owned q output differs')
                    return elapsed
                observer.set_phase('warmup');warmups = [sample() for _ in range(protocol['warmups_per_case'])];before = dict(observer.calls)
                with observer.unobserved('primary'):
                    samples = [sample() for _ in range(protocol['samples_per_case'])]
                    require(observer.handle is None and dict(observer.calls) == before, 'Primary observer unexpectedly attached')
                pairs = []
                for number in range(protocol['calibration_pairs_per_case']):
                    order = ['on','off'] if (phase+index+number)%2 == 0 else ['off','on'];pair = dict(pair_index=number,order=order)
                    for calibration_mode in order:
                        if calibration_mode == 'on':
                            observer.set_phase('calibration_on');pair['on_ns'] = sample()
                        else:
                            with observer.unobserved('calibration_off'):
                                pair['off_ns'] = sample()
                    pair['on_minus_off_ns'] = pair['on_ns']-pair['off_ns'];pairs.append(pair)
                require(input_hashes == {name:digest(value) for name,value in host.items()}, 'Caller q-head inputs mutated')
                result['rows'].append(dict(case_index=index,identity=identity,input_sha256=input_hashes,target_sha256=digest(target),
                    output_sha256=expected,metrics=metrics,raw_wall_nanoseconds=samples,median_ms=statistics.median(samples)/1e6,
                    instrumented_first_call_ns=first,instrumented_warmup_ns=warmups,calibration=pairs))
                require(torch.cuda.max_memory_allocated() <= protocol['maximum_CUDA_allocated_bytes']
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['maximum_RSS_bytes'], 'Q-head resource bound exceeded')
                if position%256 == 255:
                    save(folder/'result.json',result);print(fit_id,'phase',phase,'cases',position+1,flush=True)
        observation = resources();result['resource_observations'].append(observation);require(not observation['other_compute_pids'], 'Other GPUcompute observed')
        check_model();observer.close()
        if phase == 0:
            archive = folder/'q-predictions.npz';np.savez_compressed(archive,**predictions);result['prediction_archive'] = binding(archive)
        else:
            result['prediction_archive'] = previous['prediction_archive']
        verify_sources(protocol)
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT);require(all(name in protocol['source_sha256'] for name in coverage['lab_python']), 'Actual unbound q-head source')
        result.update(status='original_causal_q_GPU_phase_qualified',finished_unix=time.time(),cases=2048,
            learned_tensors_and_architecture_unchanged=True,primary_observer_calls=0,source_coverage=coverage,
            primary_calls=20480,logical_calls=49152,calibration_pairs=8192,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_wall_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(status='original_causal_q_GPU_adapter_gap',error_type=type(error).__name__,error=str(error),scientific_accuracy_failure_claim=False)
        raise
    finally:
        if observer:
            observer.close();result['device_observation'] = observer.evidence()
        save(folder/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True);parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    args = parser.parse_args();run(args.protocol.resolve(),args.fit_id,args.phase)
