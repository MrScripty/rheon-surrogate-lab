"""Original CPU data rounding, independent NumPy q-error/timing reduction.

Torch is used only by the captured decoder/known-control preparation. A global
guard forbids CPU neural execution; no model, trainer or objective is imported.
"""
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from modal_component_benchmark import verify_sources
from modal_component_data import digest
from causal_request_component_data import load_original, owned_case
from cuda_forward_observer import CUDAForwardObserver


def score_q(prediction,target,q_scale):
    require(prediction.dtype.str == target.dtype.str == '<f4' and prediction.shape == target.shape
            and prediction.ndim == 6 and prediction.shape[:3] == (1,1,1)
            and np.isfinite(prediction).all() and np.isfinite(target).all() and 0 < q_scale < math.inf, 'Finite original F32q layouts/scale required')
    p,t = prediction.astype('<f8'),target.astype('<f8');delta = p-t;centered = delta-delta.mean();reference = t-t.mean()
    norm = math.sqrt(float(np.dot(t.ravel(),t.ravel())));centered_norm = math.sqrt(float(np.dot(reference.ravel(),reference.ravel())))
    error = math.sqrt(float(np.dot(delta.ravel(),delta.ravel())));rms = error/math.sqrt(delta.size);maximum = float(np.abs(delta).max())
    return dict(q_relative_l2=error/max(norm,1e-30),
        q_mean_free_relative_l2=math.sqrt(float(np.dot(centered.ravel(),centered.ravel())))/max(centered_norm,1e-30),
        q_absolute_rms_normalized=rms,q_absolute_max_normalized=maximum,q_absolute_rms_physical=rms*q_scale,
        q_absolute_max_physical=maximum*q_scale,q_target_l2=norm,q_target_floor_active=norm < 1e-30,
        q_reference_centered_l2=centered_norm,q_reference_centered_floor_active=centered_norm < 1e-30)


def summarize(rows):
    valid = [row for row in rows if not row['metrics']['q_target_floor_active']]
    centered = [row for row in rows if not row['metrics']['q_reference_centered_floor_active']]
    return dict(cases=len(rows),target_floor_cases=len(rows)-len(valid),centered_reference_floor_cases=len(rows)-len(centered),
        worst_q_relative_l2_nonzero_targets=max((row['metrics']['q_relative_l2'] for row in valid),default=None),
        worst_q_mean_free_relative_l2_nonzero_targets=max((row['metrics']['q_mean_free_relative_l2'] for row in centered),default=None),
        worst_q_absolute_rms_normalized=max(row['metrics']['q_absolute_rms_normalized'] for row in rows),
        worst_q_absolute_max_normalized=max(row['metrics']['q_absolute_max_normalized'] for row in rows),
        worst_q_absolute_rms_physical=max(row['metrics']['q_absolute_rms_physical'] for row in rows),
        worst_q_absolute_max_physical=max(row['metrics']['q_absolute_max_physical'] for row in rows),
        median_case_GPU_ms=statistics.median(row['pooled_median_ms'] for row in rows),
        maximum_case_median_GPU_ms=max(row['pooled_median_ms'] for row in rows))


def run():
    output = DOC/'reports/causal-request-component-GPU-benchmark-01.json';require(not output.exists(), 'Preserve original causal q reduction')
    protocol_path = RUN/'causal-request-component-GPU-protocol-01.json';protocol = read(protocol_path);verify_sources(protocol)
    folder = ROOT/protocol['output_root'];owner = read(folder/'status.json')
    require(owner['status'] == 'all_four_original_causal_q_GPU_workers_terminal' and not owner['pending'] and owner['active_child'] is None
            and len(owner['qualified']) == 4 and not Path(f'/proc/{owner["pid"]}').exists() and owner['protocol'] == binding(protocol_path), 'Original four-worker owner must be terminal')
    require(binding(ROOT/protocol['material']['path']) == protocol['material']
            and binding(ROOT/protocol['sampling_membership']['path']) == protocol['sampling_membership'], 'Original material/sampling replay changed')
    entries = {row['fit_item_id']:row for row in read(ROOT/protocol['material']['path'])['models']}
    sampling = {row['fit_item_id']:row['counts_by_case'] for row in read(ROOT/protocol['sampling_membership']['path'])['fits']}
    phases,pids = {},set()
    for item in owner['qualified']:
        spec = item['result'];data = read(ROOT/spec['path']);key = (item['fit_item_id'],item['phase'])
        require(binding(ROOT/spec['path']) == spec and data['status'] == 'original_causal_q_GPU_phase_qualified'
                and data['pid'] == item['pid'] and data['pid'] not in pids and not Path(f'/proc/{data["pid"]}').exists()
                and key == (data['fit_item_id'],data['phase']) and key not in phases and data['protocol'] == binding(protocol_path), 'Exact original worker binding required')
        pids.add(data['pid']);phases[key] = data
        require(data['cases'] == len(data['rows']) == 2048
                and [row['case_index'] for row in data['rows']] == (list(range(2048)) if item['phase'] == 0 else list(reversed(range(2048))))
                and data['primary_calls'] == 20480 and data['logical_calls'] == 49152 and data['calibration_pairs'] == 8192, 'Complete original case/call counts required')
        require(data['inference_device'] == 'cuda' and data['GPU'] == protocol['expected_GPU'] and data['host_cpu'] == 4
                and data['threads'] == data['interop_threads'] == 1 and data['CPU_neural_forwards'] == data['physical_simulation_steps'] == 0
                and data['timing_units'] == 'ms/normalized_future_q_request' and data['learned_tensors_and_architecture_unchanged']
                and not data['reference_labels_forwarded'] and not data['primary_forward_observer_enabled'] and data['primary_observer_calls'] == 0,
                'Original model/device/precision/request boundary differs')
        calls = data['device_observation']['calls_by_phase'];observer = data['device_observation']
        multiplier = next(row['nn_conv3d_calls_per_forward'] for row in protocol['fits'] if row['fit_item_id'] == item['fit_item_id'])
        require(observer['closed'] and not observer['observer_attached'] and not observer['forbidden_forward_attempts']
                and calls['first_call'] == {'CausalRequestNet':2048,'Conv3d':2048*multiplier}
                and calls['warmup'] == {'CausalRequestNet':10240,'Conv3d':10240*multiplier}
                and calls['calibration_on'] == {'CausalRequestNet':8192,'Conv3d':8192*multiplier}
                and not calls.get('data_preparation') and not calls.get('primary') and not calls.get('calibration_off'), 'Actual observed CUDA module-call scopes differ')
        require(len(data['resource_observations']) == 9 and all(not row['other_compute_pids'] for row in data['resource_observations'])
                and data['peak_CUDA_allocated_bytes'] <= protocol['maximum_CUDA_allocated_bytes']
                and data['peak_RSS_bytes'] <= protocol['maximum_RSS_bytes'], 'Resource qualification differs')
        require(all(name in protocol['source_sha256'] for name in data['source_coverage']['lab_python']), 'Actual unbound source')
    require(set(phases) == {(row['fit_item_id'],phase) for row in protocol['fits'] for phase in (0,1)}, 'Both original fits in both phases required')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'Independent decoder/scorer must not initialize CUDA')
    guard = CUDAForwardObserver(torch);guard.enable('original_CPU_decoder_and_independent_numpy_score')
    models = [];canonical_bindings = None
    try:
        canonical_entry = entries[protocol['fits'][0]['fit_item_id']]
        module,manifest,norm,parents,caches = load_original(canonical_entry)
        for descriptor in protocol['fits']:
            fid = descriptor['fit_item_id'];first,second = (phases[(fid,phase)] for phase in (0,1))
            require(first['weights'] == second['weights'] == entries[fid]['outputs']['weights.npz']
                    and first['architecture'] == second['architecture'] and first['prediction_archive'] == second['prediction_archive'], 'Original unchanged q head and one reused archive required')
            archive_spec = first['prediction_archive'];require(binding(ROOT/archive_spec['path']) == archive_spec, 'Saved q fields changed')
            lookup = [{row['case_index']:row for row in data['rows']} for data in (first,second)];rows,overheads,bindings = [],[],[]
            with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
                require(set(archive.files) == {f'case{i:04d}' for i in range(2048)}, 'All2048saved q fields required')
                for index in range(2048):
                    identity,host,target = owned_case(module,parents,caches,index)
                    identity.update(main_q_fit_visits=sampling[fid][index],main_q_fit_seen=sampling[fid][index] > 0,
                        sampling_support_region='0..120' if identity['current_state'] <= 120 else '121..127')
                    prediction = archive[f'case{index:04d}'];require(prediction.shape == target.shape, 'Original complete predicted q layout required')
                    metrics = score_q(prediction,target,norm['scales']['q']);input_hashes = {name:digest(value) for name,value in host.items()}
                    target_hash = digest(target);bindings.append(dict(identity=identity,input_sha256=input_hashes,target_sha256=target_hash))
                    samples,calibration = [],[]
                    for phase in (0,1):
                        row = lookup[phase][index]
                        require(row['identity'] == identity and row['input_sha256'] == input_hashes and row['target_sha256'] == target_hash
                                and row['output_sha256'] == digest(prediction), 'Exact original/current input/target/output identities differ')
                        require(set(metrics) == set(row['metrics']), 'Same q metric fields required')
                        for name,value in metrics.items():
                            expected = row['metrics'][name]
                            require(value == expected if isinstance(value,bool) else math.isclose(value,expected,rel_tol=2e-12,abs_tol=1e-300), 'Independent NumPy q score differs: '+name)
                        raw = row['raw_wall_nanoseconds']
                        require(len(raw) == 10 and all(type(value) is int and value > 0 for value in raw)
                                and row['median_ms'] == statistics.median(raw)/1e6 and row['instrumented_first_call_ns'] > 0
                                and len(row['instrumented_warmup_ns']) == 5 and all(value > 0 for value in row['instrumented_warmup_ns']), 'Raw primary/first/warmup timings differ')
                        require(len(row['calibration']) == 4, 'Four paired calibrations required')
                        for number,pair in enumerate(row['calibration']):
                            require(pair['pair_index'] == number and pair['order'] == (['on','off'] if (phase+index+number)%2 == 0 else ['off','on'])
                                    and pair['on_ns'] > 0 and pair['off_ns'] > 0 and pair['on_minus_off_ns'] == pair['on_ns']-pair['off_ns'], 'Signed alternating observer calibration differs')
                            calibration.append(pair['on_minus_off_ns'])
                        samples.extend(raw)
                    rows.append(dict(case_index=index,identity=identity,metrics=metrics,input_sha256=input_hashes,target_sha256=target_hash,
                        output_sha256=digest(prediction),raw_wall_nanoseconds=samples,pooled_median_ms=statistics.median(samples)/1e6,
                        phase_medians_ms=[lookup[p][index]['median_ms'] for p in (0,1)],paired_observer_overhead_median_ms=statistics.median(calibration)/1e6))
                    overheads.extend(calibration)
            require(canonical_bindings is None or canonical_bindings == bindings, 'All2048input/target/sampling identities must match across both heads')
            canonical_bindings = bindings
            groups = dict(overall=summarize(rows),by_cells_per_metre={},by_counts_xyz={},by_shape_and_density={},by_main_q_fit_seen={},by_sampling_support={})
            for row in rows:
                identity = row['identity'];density = str(identity['cells_per_metre']);shape = 'x'.join(map(str,identity['counts_xyz']))
                for group,key in [('by_cells_per_metre',density),('by_counts_xyz',shape),('by_shape_and_density',f'{shape}@{density}'),
                                  ('by_main_q_fit_seen','sampled' if identity['main_q_fit_seen'] else 'not_sampled'),('by_sampling_support',identity['sampling_support_region'])]:
                    groups[group].setdefault(key,[]).append(row)
            for name in groups:
                if name != 'overall':
                    groups[name] = {key:summarize(value) for key,value in groups[name].items()}
            models.append(dict(fit_item_id=fid,weights=first['weights'],groups=groups,rows=rows,prediction_archive=archive_spec,
                paired_observer_overhead_median_ms=statistics.median(overheads)/1e6,negative_paired_differences=sum(value < 0 for value in overheads),
                raw_results=[item['result'] for item in owner['qualified'] if item['fit_item_id'] == fid]))
    finally:
        guard.close()
    require(not torch.cuda.is_initialized() and not guard.calls and not guard.forbidden_attempts, 'Independent decoder/scorer must perform no neural forwards or CUDA work')
    forbidden_sources = {row['path'] for entry in entries.values() for row in entry['sources'] if row['original_name'] in ('pressure_request_model.py','pressure_request_train.py','pressure_request_objective.py')}
    from python_source_coverage import collect
    coverage = collect(sys.modules,ROOT)
    require(not forbidden_sources.intersection(coverage['lab_python']) and all(name in protocol['source_sha256'] for name in coverage['lab_python']), 'Model/objective import or unbound independent decoder source')
    save(output,dict(status='two_original_causal_q_GPU_benchmarks_independently_reduced',created_unix=time.time(),models=models,
        fits=2,cases_per_fit=2048,fresh_processes=4,primary_calls=81920,logical_calls=196608,paired_calibration_pairs=32768,
        inference_device='cuda',GPU=protocol['expected_GPU'],timing_units='ms/normalized_future_q_request',physical_simulation_steps=0,
        all_cross_fit_input_target_and_membership_bindings_exact=True,all_process_output_hashes_exact=True,
        primary_observer_removed=True,observer_overhead_subtracted=False,independent_metric_backend='NumPy;capturedTorchCPUdecoder only,guardedagainstCPUmodel execution',
        independent_decoder_no_neural_forward_evidence=guard.evidence(),independent_decoder_CUDA_initialized=False,
        independent_source_coverage=coverage,scope=protocol['population'],precision_limit=protocol['precision_limit'],
        protocol=binding(protocol_path),contract=protocol['contract'],material=protocol['material'],sampling_membership=protocol['sampling_membership'],
        owner=binding(folder/'status.json'),producer=binding(Path(__file__)),CPU_neural_forwards=0,
        new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,full_portfolio_complete=False))
    for model in models:
        summary = model['groups']['overall'];print(model['fit_item_id'],summary['median_case_GPU_ms'],'GPUms/qrequest;worstq',summary['worst_q_relative_l2_nonzero_targets'],flush=True)


if __name__ == '__main__':
    run()
