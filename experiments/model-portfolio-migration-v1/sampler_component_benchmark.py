"""Original48F64parameter recurrent head on all576retained local-point groups."""
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

INPUTS = ('point','patch','kind','alpha')


def run(protocol_path,phase):
    current_publication(ROOT);protocol = read(protocol_path)
    require(protocol['schema'] == 'rheon.sampler-component-GPU-benchmark.v1' and phase in (0,1)
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Explicit original sampler CUDA protocol required')
    verify_sources(protocol)
    require(binding(ROOT/protocol['material']['path']) == protocol['material'], 'Original sampler material changed')
    entry = next(row for row in read(ROOT/protocol['material']['path'])['models'] if row['fit_item_id'] == protocol['fit_item_id'])
    require(entry['task'] == 'recurrent_transport_component' and entry['serialized_tensor_elements'] == 48
            and entry['parameter_dtypes'] == ['<f8'], 'Exact original48F64head required')
    for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),*entry['inputs'].values()]:
        observed_file(ROOT,spec['path'],spec)
    inventory,_ = selected_inventory(read(ROOT/entry['original_receipt']['path']),read(ROOT/entry['outputs']['history.json']['path']),entry['outputs']['weights.npz']['sha256'])
    require(inventory == entry['selected_inventory'], 'Original selected sampler inventory differs')
    folder = ROOT/protocol['output_root']/f'phase-{phase}'
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh original sampler phase required')
    folder.mkdir(parents=True);os.sched_setaffinity(0,{protocol['host_cpu']})
    result = dict(status='starting_original_sampler_GPU_phase',pid=os.getpid(),phase=phase,created_unix=time.time(),
        fit_item_id=entry['fit_item_id'],protocol=binding(protocol_path),rows=[],resource_observations=[],inference_device='cuda',
        timing_units='ms/local_sampler_group',physical_simulation_steps=0,CPU_neural_forwards=0,reference_labels_forwarded=False,
        primary_forward_observer_enabled=False,new_parameter_updates=0,new_reference_generation=False,full_portfolio_complete=False)
    save(folder/'result.json',result);observer = None;began = time.monotonic()
    try:
        import numpy as np
        import torch
        from sampler_component_data import load_groups
        from modal_component_data import digest
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Registered GPU/full precision required')
        observer = CUDAForwardObserver(torch);observer.enable('construction')
        normalization = read(ROOT/entry['inputs']['normalization']['path'])
        source = next(row for row in entry['sources'] if row['original_name'] == 'learned_sampler_head.py')
        with torch.device('cuda'):
            module = imported('original_sampler_head',ROOT/source['path'])
            model = module.LearnedSamplerHead(normalization['inherited_mix']).eval().requires_grad_(False)
        with np.load(ROOT/entry['outputs']['weights.npz']['path'],allow_pickle=False) as archive:
            require(set(archive.files) == set(inventory) == set(model.state_dict()), 'Complete original sampler tensors required')
            loaded = {}
            for name in archive.files:
                value = archive[name];spec = inventory[name]
                require(value.dtype.str == spec['dtype'] == '<f8' and list(value.shape) == spec['shape']
                        and digest(value) == spec['sha256'], 'Original learned sampler tensor differs')
                loaded[name] = torch.from_numpy(value.copy()).to('cuda')
            model.load_state_dict(loaded,strict=True)
        architecture = repr(model)
        def check_model():
            require(repr(model) == architecture, 'Sampler architecture changed')
            for name,value in model.state_dict().items():
                require(value.device.type == 'cuda' and value.dtype == torch.float64
                        and digest(value.detach().cpu().numpy()) == inventory[name]['sha256'], 'Original learned sampler tensors changed')
        check_model();groups = load_groups(entry)
        prior = None
        if phase == 1:
            previous = read(folder.parent/'phase-0/result.json')
            require(previous['status'] == 'original_sampler_GPU_phase_qualified' and not Path(f'/proc/{previous["pid"]}').exists(), 'Original phase0must be terminal')
            prior = {row['case_index']:row for row in previous['rows']}
        result.update(GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            architecture=architecture,weights=entry['outputs']['weights.npz'],host_cpu=protocol['host_cpu'],
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),setup_wall_seconds=time.monotonic()-began)

        def complete(host):
            require(set(host) == set(INPUTS), 'Only original current-point/patch/carrier/alpha inputs allowed')
            count = len(host['kind'])
            shapes = dict(point=(count,3),patch=(count,3,3,3,3),kind=(count,),alpha=(count,))
            forwarded = {}
            for name in INPUTS:
                value = host[name];dtype = '<f4' if name == 'patch' else '<i8' if name == 'kind' else '<f8'
                require(value.dtype.str == dtype and value.shape == shapes[name] and np.isfinite(value).all(), 'Original finite sampler input types/shapes required')
                forwarded[name] = torch.from_numpy(np.ascontiguousarray(value)).to('cuda')
            require(np.isin(host['kind'],range(4)).all() and (host['alpha'] > 0).all(), 'Physical carrier/alpha inputs required')
            middle,end = model(*(forwarded[name] for name in INPUTS))
            require(all(value.device.type == 'cuda' and value.dtype == torch.float64 and value.shape == (count,3)
                        for value in (middle,end)), 'Original complete CUDA F64 stage vectors required')
            outputs = tuple(value.detach().to('cpu',copy=True).numpy() for value in (middle,end))
            torch.cuda.synchronize();return outputs

        def score(prediction,target):
            p = torch.from_numpy(prediction);t = torch.from_numpy(target)
            difference = p-t;reference_norm = float(t.norm());delta_norm = float(difference.norm())
            point_reference = torch.linalg.vector_norm(t,dim=1);point_delta = torch.linalg.vector_norm(difference,dim=1)
            floor = point_reference < 1e-30;valid = ~floor
            return dict(relative_l2=delta_norm/max(reference_norm,1e-30),absolute_rms=float(difference.square().mean().sqrt()),
                absolute_max=float(difference.abs().max()),target_l2=reference_norm,target_floor_active=reference_norm < 1e-30,
                points=len(target),zero_target_points=int(floor.sum()),
                worst_point_relative_l2_nonzero_targets=float((point_delta[valid]/point_reference[valid]).max()) if bool(valid.any()) else None,
                zero_target_worst_absolute_l2=float(point_delta[floor].max()) if bool(floor.any()) else None)

        predictions = {}
        with torch.inference_mode():
            for position,index in enumerate(range(576) if phase == 0 else reversed(range(576))):
                require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Finite sampler worker deadline exceeded')
                if position%144 == 0:
                    observation = resources();result['resource_observations'].append(observation)
                    require(not observation['other_compute_pids'], 'Other GPUcompute observed')
                group = groups[index];arrays = group['arrays'];host = {name:arrays[name].copy() for name in INPUTS}
                input_hashes = {name:digest(value) for name,value in host.items()}
                observer.set_phase('first_call');tick = time.perf_counter_ns();outputs = complete(host);first = time.perf_counter_ns()-tick
                expected = {name:digest(value) for name,value in zip(('middle','end'),outputs,strict=True)}
                if prior:
                    require(prior[index]['output_sha256'] == expected, 'Exact cross-phase sampler outputs differ')
                for stage,value in zip(('middle','end'),outputs,strict=True):
                    require(np.isfinite(value).all() and not any(np.shares_memory(value,item) for item in host.values()), 'Finite independently owned stage vectors required')
                    if phase == 0:
                        predictions[f'case{index:03d}_{stage}'] = value.copy()
                values = {stage:score(value,arrays[stage]) for stage,value in zip(('middle','end'),outputs,strict=True)}
                require(all(value is None or math.isfinite(value) for metric in values.values() for value in metric.values()), 'Finite vector errors required')
                def sample():
                    tick = time.perf_counter_ns();values = complete(host);elapsed = time.perf_counter_ns()-tick
                    require(elapsed > 0 and all(digest(value) == expected[stage] and np.isfinite(value).all()
                            and not any(np.shares_memory(value,item) for item in host.values())
                            for stage,value in zip(('middle','end'),values,strict=True)), 'Repeated sampler outputs differ')
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
                require(input_hashes == {name:digest(value) for name,value in host.items()}, 'Caller sampler inputs mutated')
                result['rows'].append(dict(case_index=index,identity=group['identity'],input_sha256=input_hashes,
                    target_sha256={stage:digest(arrays[stage]) for stage in ('middle','end')},output_sha256=expected,metrics=values,
                    raw_wall_nanoseconds=samples,median_ms=statistics.median(samples)/1e6,
                    instrumented_first_call_ns=first,instrumented_warmup_ns=warmups,calibration=pairs))
                require(torch.cuda.max_memory_allocated() <= protocol['maximum_CUDA_allocated_bytes']
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['maximum_RSS_bytes'], 'Sampler resource bound exceeded')
                if position%144 == 143:
                    save(folder/'result.json',result);print('Sampler phase',phase,'groups',position+1,flush=True)
        observation = resources();result['resource_observations'].append(observation)
        require(not observation['other_compute_pids'], 'Other GPUcompute observed')
        check_model();observer.close()
        if phase == 0:
            archive = folder/'stage-predictions.npz';np.savez_compressed(archive,**predictions);result['prediction_archive'] = binding(archive)
        else:
            result['prediction_archive'] = previous['prediction_archive']
        verify_sources(protocol)
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT)
        require(all(name in protocol['source_sha256'] for name in coverage['lab_python']), 'Unbound actual sampler source')
        result.update(status='original_sampler_GPU_phase_qualified',finished_unix=time.time(),cases=576,points=77998,
            learned_tensors_and_architecture_unchanged=True,primary_observer_calls=0,source_coverage=coverage,
            primary_calls=5760,logical_calls=13824,calibration_pairs=2304,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_wall_seconds=time.monotonic()-began)
    except BaseException as error:
        result.update(status='original_sampler_GPU_adapter_gap',error_type=type(error).__name__,error=str(error),scientific_accuracy_failure_claim=False)
        raise
    finally:
        if observer:
            observer.close();result['device_observation'] = observer.evidence()
        save(folder/'result.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    args = parser.parse_args();run(args.protocol.resolve(),args.phase)
