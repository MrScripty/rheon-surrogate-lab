"""One original joint generator: native CPU fields to K owned CPU outputs.

Only the neural forward runs on CUDA. Reference loading, hashes, repeatability,
ownership and scoring are outside primary wall timing. No arrays are published.
"""
import argparse
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
import time

from full_state_gpu_timing import check_owned, cpu_views, field_sizes, hashes, numpy_outputs, resources
from review_direction_pressure_fits import operator
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('pressure','vx','vy','vz','tracer')


def score(np, output, reference, scene):
    result = {}
    for name in FIELDS:
        predicted, truth = output[name].astype(np.float64), reference[name].astype(np.float64)
        if name == 'pressure':
            predicted, truth = predicted-predicted.mean(), truth-truth.mean()
        denominator = float(np.dot(truth,truth))
        difference = predicted-truth
        squared = float(np.dot(difference,difference))
        result[name+'_relative_l2'] = math.sqrt(squared/denominator) if denominator else (0. if squared == 0 else None)
        result[name+'_absolute_rms'] = math.sqrt(squared/truth.size)
    shape = tuple(reversed(scene['counts']))
    rhs = reference['rhs'].reshape(shape)
    residual = rhs-operator(output['pressure'].reshape(shape),1/scene['cells_per_metre'],1000.)
    denominator = float(np.square(rhs).sum())
    numerator = float(np.square(residual).sum())
    result['equation_relative_l2'] = math.sqrt(numerator/denominator) if denominator else (0. if numerator == 0 else None)
    velocity_squared = sum(float(np.square(output[n].astype(np.float64)-reference[n]).sum()) for n in ('vx','vy','vz'))
    velocity_norm = sum(float(np.square(reference[n].astype(np.float64)).sum()) for n in ('vx','vy','vz'))
    result['velocity_relative_l2'] = math.sqrt(velocity_squared/velocity_norm) if velocity_norm else (0. if velocity_squared == 0 else None)
    return result


def run(protocol_path, identifier, phase, output):
    require(not output.exists() and output.is_relative_to(ROOT/'runs/model-portfolio-migration-v1')
            and phase in (0,1) and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Fresh admitted CUDA result required')
    protocol = read(protocol_path)
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec, 'Admitted benchmark dependency changed')
    case_path = ROOT/protocol['cases']['path']
    require(binding(case_path) == protocol['cases'], 'Prequalified native cases changed')
    prepared = read(case_path)
    models = [m for m in prepared['models'] if m['identifier'] == identifier]
    require(len(models) == 1, 'One original joint model required')
    selected = models[0]
    before = resources()
    require(not before['other_compute_pids'], 'Another CUDA compute process is active; no benchmark dispatched')
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
    started = time.perf_counter()
    result = dict(status='running',identifier=identifier,fit_item_id=selected['fit_item_id'],phase=phase,
        protocol_sha256=binding(protocol_path)['sha256'],reference_method=prepared['reference_method'],
        reference_native_probe_binary_sha256=prepared['reference_native_probe_binary_sha256'],
        K=selected['K'],output_dt_s=1/60,exposure='consumed_train_scenes',rows=[],resources_before=before,
        CPU_neural_forwards=0,CUDA_module_calls=0,CUDA_network_calls=0,forbidden_forward_attempts=0,new_reference_generation=False,array_copies_to_disk=0,
        complete_all_model_portfolio=False)
    try:
        project = ROOT/selected['project_path']
        for spec in [selected['configuration'],selected['receipt'],selected['weights'],*selected['sources']]:
            require(binding(ROOT/spec['path']) == spec, 'Original model/source/weight identity changed')
        normalization = selected['normalization']
        require(binding(ROOT/normalization['path'])['sha256'] == normalization['sha256'], 'Normalization changed')
        norm = read(ROOT/normalization['path'])
        config = read(ROOT/selected['configuration']['path'])
        receipt = read(ROOT/selected['receipt']['path'])
        sys.path.insert(0,str(project))
        module = importlib.import_module('bundle_model_v5')
        torch.manual_seed(config['seed'])
        network = module.BundleNet(selected['K'],config['width'])
        with np.load(ROOT/selected['weights']['path'],allow_pickle=False) as weights:
            inventory = receipt['final_inventory']
            require(set(weights.files) == set(network.state_dict()) == set(inventory), 'Complete original tensor inventory required')
            values = {}
            for name in weights.files:
                value, expected = weights[name], inventory[name]
                require(value.dtype.str == '<f4' and list(value.shape) == expected['shape']
                        and hashlib.sha256(value.tobytes()).hexdigest() == expected['sha256'], 'Original tensor differs')
                values[name] = torch.from_numpy(value.copy())
            network.load_state_dict(values,strict=True)
        network.to('cuda').eval()
        require(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in network.parameters()), 'Neural parameters must be CUDAF32')

        def guard(owner, args):
            tensors = [v for v in args if isinstance(v,torch.Tensor)]
            if not tensors or any(v.device.type != 'cuda' for v in tensors):
                result['forbidden_forward_attempts'] += 1
                raise ValueError('CPU/unclassified neural forward forbidden')
            result['CUDA_module_calls'] += 1
            result['CUDA_network_calls'] += owner is network

        handle = torch.nn.modules.module.register_module_forward_pre_hook(guard)
        torch.cuda.synchronize()
        result['setup_wall_seconds'] = time.perf_counter()-started
        torch.cuda.reset_peak_memory_stats()
        cases = prepared['cases'] if phase == 0 else list(reversed(prepared['cases']))
        for case in cases:
            require(time.perf_counter()-started < protocol['worker_deadline_seconds'], 'Admitted benchmark deadline exceeded')
            sizes = field_sizes(case['scene']['counts'])
            current_path = ROOT/case['current']['path']
            require(binding(current_path)['sha256'] == case['current']['sha256'], 'Current native frame changed')
            with np.load(current_path,allow_pickle=False) as arrays:
                native = {name:arrays[name].copy() for name in FIELDS}
            input_hashes = hashes(native)
            k = selected['K']
            expected_hashes, previous = None, None
            samples, warmup_ms, first_ms, metrics = [], [], None, None
            for repetition in range(1+protocol['warmups']+protocol['samples']):
                torch.cuda.synchronize()
                tick = time.perf_counter_ns()
                with torch.inference_mode():
                    views = cpu_views(np,torch,native,sizes)
                    generated = network.predict(views,case['scene'],case['start'],norm)
                    owned = [{name:row[name].to('cpu',copy=True).contiguous() for name in FIELDS} for row in generated]
                    returned = numpy_outputs(torch,owned,k,sizes)
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns()-tick)/1e6
                if repetition == 0:
                    first_ms = elapsed
                elif repetition > protocol['warmups']:
                    samples.append(elapsed)
                else:
                    warmup_ms.append(elapsed)
                require(all(np.isfinite(row[name]).all() for row in returned for name in FIELDS), 'Nonfinite generated field')
                check_owned(np,returned,native,previous)
                require(hashes(native) == input_hashes, 'Prediction mutated native input')
                actual_hashes = [hashes(row) for row in returned]
                if expected_hashes is None:
                    expected_hashes = actual_hashes
                    metrics = []
                    for lead,row in enumerate(returned,1):
                        ref = case['reference_leads'][lead-1]
                        path = ROOT/ref['path']
                        require(binding(path)['sha256'] == ref['sha256'], 'Native target frame changed')
                        with np.load(path,allow_pickle=False) as arrays:
                            truth = {name:arrays[name] for name in (*FIELDS,'rhs')}
                            metrics.append(dict(lead=lead,time_s=ref['time_s'],values=score(np,row,truth,case['scene'])))
                else:
                    require(actual_hashes == expected_hashes, 'Repeated identical case changes output bytes')
                previous = returned
            require(len(samples) == protocol['samples'], 'All admitted timed samples required')
            result['rows'].append(dict(identifier=case['identifier'],start=case['start'],counts_xyz=case['scene']['counts'],
                cells_per_metre=case['scene']['cells_per_metre'],K=k,output_dt_s=1/60,block_advance_s=k/60,
                first_call_block_ms=first_ms,warmup_block_ms=warmup_ms,block_ms=samples,per_output_ms=[v/k for v in samples],
                output_hashes=expected_hashes,metrics=metrics,all_repeat_and_ownership_checks_passed=True))
        handle.remove()
        require(len(result['rows']) == 64 and result['CUDA_module_calls'] > 0
                and result['CUDA_network_calls'] == 64*(1+protocol['warmups']+protocol['samples']),
                'All64cases and exact original CUDA network calls required')
        result.update(status='all64_native_cases_measured',peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            resources_after=resources(),wall_seconds=time.perf_counter()-started)
        require(result['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes'], 'Admitted CUDA allocation exceeded')
        for spec in protocol['dependencies']:
            require(binding(ROOT/spec['path']) == spec, 'Benchmark source changed during measurement')
    except BaseException as error:
        result.update(status='failed_original_worker_no_automatic_replay',error_type=type(error).__name__,error=str(error),
                      wall_seconds=time.perf_counter()-started)
        raise
    finally:
        output.write_bytes(canonical(result)+b'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--identifier',required=True)
    parser.add_argument('--phase',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.identifier,args.phase,args.output.resolve())
