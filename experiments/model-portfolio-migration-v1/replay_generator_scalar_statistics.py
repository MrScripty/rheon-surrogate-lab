"""One original generator, exact accepted CUDA outputs, scalar-only retention.

Accuracy replay may share the GPU. No primary latency is collected. Every output
lead must match the accepted benchmark field hashes before new diagnostics count.
Source-scoped Python calls and guarded module calls are partial work evidence,
not GPU kernels, independent projections, or per-lead cost attribution.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require


def run(protocol_path, fit_id):
    protocol = read(protocol_path)
    require(os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'CUDA-only learned replay required')
    selected = next(m for m in protocol['models'] if m['fit_item_id'] == fit_id)
    folder = ROOT/protocol['output_root']/fit_id
    require(not folder.exists(), 'Fresh scalar replay owner; no automatic retry')
    for spec in [*protocol['dependencies'],*selected['bound_inputs']]:
        require(binding(ROOT/spec['path']) == spec, 'Preregistered replay input changed: '+spec['path'])
    for name,sha in protocol['admitted_lab_python'].items():
        require(binding(ROOT/name)['sha256'] == sha, 'Frozen Python changed: '+name)
    from component_material import current_publication
    current_publication(ROOT)
    folder.mkdir(parents=True)
    began = time.monotonic()
    result = dict(status='starting_generator_scalar_replay',fit_item_id=fit_id,pid=os.getpid(),
        protocol=binding(protocol_path),identifier=selected['identifier'],K=selected['K'],rows=[],
        inference_device='cuda',latency_qualified=False,primary_timing_samples=0,
        prediction_arrays_saved=False,new_parameter_fits=0,new_reference_generation=0,native_mutations=0)
    save(folder/'result.json',result)
    observer = None
    try:
        os.sched_setaffinity(0,{protocol['host_cpu']})
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import numpy as np
        import torch
        from cuda_forward_observer import CUDAForwardObserver
        from complete_generator_gpu_benchmark import restore, inventory
        from full_state_gpu_timing import cpu_views, field_sizes, numpy_outputs, hashes, check_owned, resources
        from joint_generator_gpu_benchmark import score
        from full_state_scalar_diagnostics import measure, FIELDS
        from python_source_coverage import collect
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        require(torch.cuda.get_device_name() == protocol['expected_GPU'], 'Expected GPU required')
        result['resources_before'] = resources()
        observer = CUDAForwardObserver(torch)
        observer.enable('construction')
        # All implicit learned constructors and any constructor unit queries
        # use CUDA. The original inherited restorer also owns its scoped guards.
        with torch.device('cuda'):
            network,construction = restore(selected)
        require(all(v.device.type == 'cuda' for v in [*network.parameters(),*network.buffers()]),
            'All neural parameters and buffers must reside on CUDA')
        receipt = read(ROOT/selected['receipt']['path'])
        require(inventory(network) == receipt['final_inventory'], 'Exact original learned state required')
        norm = read(ROOT/selected['normalization']['path'])
        cases = read(ROOT/protocol['cases']['path'])['cases']
        accepted = next(m for m in read(ROOT/protocol['accepted_accuracy']['path'])['models'] if m['fit_item_id'] == fit_id)
        prior = {(r['identifier'],r['start']):r for r in accepted['pooled_rows']}
        source_calls = Counter()
        actual_lab_calls = set()
        tracked = {str(ROOT/s['path']) for s in selected['bound_inputs'] if s['path'].endswith('.py')}
        allowed = protocol['admitted_lab_python']
        prefix = str(ROOT)+os.sep
        def profile(frame,event,arg):
            if event != 'call':
                return
            name = frame.f_code.co_filename
            if name.startswith(prefix):
                relative = name[len(prefix):]
                actual_lab_calls.add(relative)
                if name in tracked:
                    source_calls[relative+':'+frame.f_code.co_qualname] += 1
        scalar_path = folder/'statistics.jsonl.gz'
        output_count = 0
        score_checks = 0
        with scalar_path.open('xb') as destination:
            with gzip.GzipFile(filename='',mode='wb',fileobj=destination,compresslevel=6,mtime=0) as compressed:
                for index,case in enumerate(cases):
                    require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Replay worker deadline exceeded')
                    current = ROOT/case['current']['path']
                    require(binding(current)['sha256'] == case['current']['sha256'], 'Original current frame changed')
                    with np.load(current,allow_pickle=False) as stored:
                        native = {n:stored[n] for n in FIELDS}
                    before = hashes(native)
                    sizes = field_sizes(case['scene']['counts'])
                    observed_before = observer.calls.copy()
                    source_calls.clear()
                    observer.set_phase('prediction')
                    require(sys.getprofile() is None, 'Existing profiler cannot be replaced')
                    try:
                        sys.setprofile(profile)
                        with torch.inference_mode():
                            generated = network.predict(cpu_views(np,torch,native,sizes),case['scene'],case['start'],norm)
                            owned = [{n:r[n].to('cpu',copy=True).contiguous() for n in FIELDS} for r in generated]
                            returned = numpy_outputs(torch,owned,selected['K'],sizes)
                            torch.cuda.synchronize()
                    finally:
                        sys.setprofile(None)
                    require(not (actual_lab_calls-set(allowed)), 'Unbound lab Python executed during prediction')
                    require(hashes(native) == before, 'Input mutation')
                    check_owned(np,returned,native)
                    previous = prior[case['identifier'],case['start']]
                    outputs = [hashes(r) for r in returned]
                    require(outputs == previous['output_hashes'], 'Replay changed accepted physical output bytes')
                    preceding_output,preceding_truth = native,native
                    identities = []
                    for lead,row in enumerate(returned,1):
                        reference = case['reference_leads'][lead-1]
                        path = ROOT/reference['path']
                        require(binding(path)['sha256'] == reference['sha256'], 'Original reference frame changed')
                        with np.load(path,allow_pickle=False) as stored:
                            truth = {n:stored[n] for n in (*FIELDS,'rhs')}
                        actual = score(np,row,truth,case['scene'])
                        original = previous['metrics'][lead-1]
                        require(original['lead'] == lead and original['time_s'] == reference['time_s']
                            and actual == original['values'], 'Published metric/physical lead differs')
                        score_checks += len(actual)
                        h = 1/case['scene']['cells_per_metre']
                        stats = measure(row,truth,native,case['scene']['counts'],[h]*3,1000.,preceding_output,preceding_truth)
                        identity = dict(identifier=case['identifier'],start=case['start'],K=selected['K'],lead=lead,
                            output_role='endpoint' if lead == selected['K'] else 'intermediate',output_dt_s=case['output_dt_s'],
                            physical_lead_dt_s=lead*case['output_dt_s'],block_horizon_s=selected['K']*case['output_dt_s'],
                            reference_time_s=reference['time_s'],counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'])
                        compressed.write((json.dumps(dict(fit_item_id=fit_id,case_index=index,identity=identity,statistics=stats),
                            separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode())
                        identities.append(identity)
                        preceding_output,preceding_truth = row,truth
                        output_count += 1
                    result['rows'].append(dict(case_index=index,identities=identities,output_hashes=outputs,
                        source_bound_Python_calls=dict(source_calls),
                        CUDA_module_calls={name:count for (phase,name),count in (observer.calls-observed_before).items()},
                        scalar_output_count=len(returned)))
                    if index%16 == 15:
                        save(folder/'result.json',result)
        require(len(result['rows']) == 64 and output_count == 64*selected['K']
            and inventory(network) == receipt['final_inventory'], 'Complete unchanged fit/state population required')
        coverage = collect(sys.modules,ROOT)
        require(not (set(coverage['lab_python'])-set(allowed)), 'Actual imported Python source not admitted')
        for name in set(coverage['lab_python'])|actual_lab_calls:
            require(binding(ROOT/name)['sha256'] == allowed[name], 'Actually used source changed')
        observer.close()
        result.update(status='all64_generator_cases_exact_original_outputs_expanded_scalars_retained',
            constructor=construction,physical_outputs=output_count,published_metric_checks=score_checks,
            device_observation=observer.evidence(),CPU_neural_forwards=0,
            all_original_physical_output_hashes_exact=True,all_original_metric_scalars_exact=True,
            scalar_statistics=dict(**binding(scalar_path),bytes=scalar_path.stat().st_size),
            source_coverage=coverage,actual_profiled_lab_python=sorted(actual_lab_calls),
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),resources_after=resources(),
            GPU=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            work_scope='Source-bound Python method counts and observed CUDA nn.Module invocations; partial overlapping call-level work,not kernels,not independent projections,not per-lead timing.',
            wall_seconds=time.monotonic()-began,wall_time_is_accuracy_replay_cost_not_inference_latency=True)
        require(result['peak_CUDA_allocated_bytes'] <= protocol['maximum_CUDA_allocated_bytes'], 'Admitted CUDA memory bound exceeded')
    except BaseException as error:
        result.update(status='generator_scalar_replay_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if observer is not None:
            observer.close()
        save(folder/'result.json',result)
    print(fit_id,selected['identifier'],output_count,'physical outputs exact;scalars retained,no fields or timing.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fit-id',required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.fit_id)
