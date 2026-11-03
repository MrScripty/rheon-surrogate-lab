"""Independently score saved GPUfields andreduce two point timing phases.

NumPy is used only for immutable output/reference scoring andscalar bounds.
No Torch/model import or CPUlearned convolution is performed.
"""
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from component_material import observed_file

REPORT = DOC/'reports/point-component-GPU-benchmark-01.json'
DESTINATIONS = ('vx_face', 'vy_face', 'vz_face', 'tracer_cell')


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def independent_metrics(prediction, target, mask):
    """Score only selected physical values using scalar dot reductions."""
    require(prediction.shape == target.shape == mask.shape and np.isin(mask,[0,1]).all()
            and np.count_nonzero(mask) and np.isfinite(prediction).all() and np.isfinite(target).all(),
            'Finite matching arrays and nonempty binary mask required')
    active = mask.astype(bool)
    reference = target.astype('<f8')[active]
    difference = prediction.astype('<f8')[active]-reference
    error_norm = math.sqrt(float(np.dot(difference,difference)))
    target_norm = math.sqrt(float(np.dot(reference,reference)))
    return dict(masked_relative_l2=error_norm/max(target_norm,1e-30),
        masked_absolute_rms=error_norm/math.sqrt(len(difference)),masked_absolute_max=float(np.max(np.abs(difference))),
        target_masked_l2=target_norm,target_denominator_floor_active=target_norm < 1e-30,masked_values_count=len(difference))


def independent_destination_metrics(prediction, target, mask):
    require(prediction.shape[0] == target.shape[0] == mask.shape[0] == 12, 'Four consecutive XYZ groups required')
    return {name:independent_metrics(prediction[3*k:3*k+3],target[3*k:3*k+3],mask[3*k:3*k+3])
            for k,name in enumerate(DESTINATIONS)}


def same_metrics(actual, expected):
    require(set(actual) == set(expected), 'Metric fields differ')
    for key,value in actual.items():
        require(value == expected[key] if isinstance(value,(bool,int))
                else math.isclose(value,expected[key],rel_tol=2e-12,abs_tol=1e-300),
                'Independent masked point score differs: '+key)


def metric_summary(metrics):
    valid = [value for value in metrics if not value['target_denominator_floor_active']]
    zero = [value for value in metrics if value['target_denominator_floor_active']]
    return dict(cases=len(metrics),nonzero_target_cases=len(valid),floor_active_cases=len(zero),
        worst_masked_relative_l2_nonzero_targets=max((value['masked_relative_l2'] for value in valid),default=None),
        worst_masked_absolute_rms=max(value['masked_absolute_rms'] for value in metrics),
        worst_masked_absolute_max=max(value['masked_absolute_max'] for value in metrics),
        zero_target_worst_absolute_max=max((value['masked_absolute_max'] for value in zero),default=None))


def run():
    require(not REPORT.exists(),'Preserve original point reduction')
    protocol_path = RUN/'point-component-GPU-protocol-01.json'
    protocol = read(protocol_path)
    folder = ROOT/protocol['output_root']
    owner = read(folder/'status.json')
    require(owner['status'] == 'both_original_point_GPU_phases_terminal' and not owner['pending']
            and owner['active_child'] is None and not Path(f'/proc/{owner["pid"]}').exists()
            and len(owner['qualified']) == 2,'Both original point workers must be terminal')
    require(owner['protocol'] == binding(protocol_path),'Original point owner protocol changed')
    for name,sha in protocol['source_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha,'Frozen point source changed')
    require(binding(ROOT/protocol['material']['path']) == protocol['material'],'Original component material changed')
    entry = next(row for row in read(ROOT/protocol['material']['path'])['models'] if row['fit_item_id'] == protocol['fit_item_id'])
    observed_file(ROOT,entry['inputs']['teacher']['path'],entry['inputs']['teacher'])
    manifest = read(ROOT/entry['inputs']['teacher_manifest']['path'])
    with np.load(ROOT/entry['inputs']['teacher']['path'],allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in archive.files}
    require(set(arrays) == {'velocity','target','mask'} and all(digest(value) == manifest['inventory'][name]['sha256'] for name,value in arrays.items()),
            'Original teacher arrays changed')
    mask = arrays['mask'][0]
    require(np.isin(mask,[0,1]).all() and np.count_nonzero(mask),'Exact nonempty binary destination mask required')
    require(protocol['destination_channel_order'] == list(DESTINATIONS), 'Original destination order required')
    phases,pids,raw_results = {},set(),[]
    for accepted in owner['qualified']:
        spec = accepted['result']
        require(binding(ROOT/spec['path']) == spec,'Original point result changed')
        data = read(ROOT/spec['path'])
        phase = accepted['phase']
        require(data['status'] == 'original_point_GPU_phase_qualified' and data['phase'] == phase and data['cases'] == 80
                and data['pid'] == accepted['pid'] and data['pid'] not in pids and not Path(f'/proc/{data["pid"]}').exists()
                and data['protocol'] == binding(protocol_path) and data['fit_item_id'] == protocol['fit_item_id']
                and data['learned_tensors_and_architecture_unchanged'] and not data['unbound_lab_modules']
                and data['primary_observer_calls'] == 0 and not data['primary_forward_observer_enabled'],
                'Two fresh originalCUDAphases with exactsource/tensor/output policy required')
        pids.add(data['pid'])
        require(data['inference_device'] == 'cuda' and data['GPU'] == protocol['expected_GPU']
                and data['threads'] == data['interop_threads'] == 1 and data['host_cpu'] == 4
                and data['physical_simulation_steps'] == 0 and data['timing_units'] == 'ms/local_point_request'
                and data['destination_channel_order'] == list(DESTINATIONS),
                'Original component device/unit boundary changed')
        observer = data['device_observation']
        require(observer['closed'] and not observer['observer_attached'] and not observer['forbidden_forward_attempts'],
                'CUDAobserver lifecycle differs')
        counts = observer['calls_by_phase']
        require(counts['first_call'] == {'Conv3d':80,'PointSamplingNet':80}
                and counts['warmup'] == {'Conv3d':400,'PointSamplingNet':400}
                and counts['calibration_on'] == {'Conv3d':320,'PointSamplingNet':320}
                and not counts.get('primary') and not counts.get('calibration_off'),'Actual guarded/module-call scopes differ')
        require(len(data['resource_observations']) == 5 and all(not item['other_compute_pids'] for item in data['resource_observations']),
                'GPUcompute observations differ')
        require([row['pattern_index'] for row in data['rows']] == (list(range(80)) if phase == 0 else list(reversed(range(80))))
                and data['primary_calls'] == 800 and data['logical_calls'] == 1920 and data['calibration_pairs'] == 320,
                'Original pattern order orcall counts differ')
        phases[phase] = data
        raw_results.append(spec)
    require(set(phases) == {0,1} and phases[0]['prediction_archive'] == phases[1]['prediction_archive'],
            'One retained archive must be reused across exactphases')
    archive_spec = phases[0]['prediction_archive']
    require(binding(ROOT/archive_spec['path']) == archive_spec,'Saved original GPUoutput archive changed')
    index = {phase:{row['pattern_index']:row for row in data['rows']} for phase,data in phases.items()}
    rows,all_overheads = [],[]
    with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
        require(set(archive.files) == {f'pattern{i:03d}' for i in range(80)},'All80retained predictions required')
        for number in range(80):
            prediction = archive[f'pattern{number:03d}']
            require(prediction.dtype.str == '<f4' and prediction.shape == (12,13,13,13) and np.isfinite(prediction).all(),
                    'Original finiteF32negative point velocity field required')
            # Restrict to physical destination values before reduction, unlike
            # theworker's full-vector zero-mask reduction. No model arithmetic.
            metrics = independent_metrics(prediction,arrays['target'][number],mask)
            destinations = independent_destination_metrics(prediction,arrays['target'][number],mask)
            samples,overheads = [],[]
            for phase in (0,1):
                row = index[phase][number]
                require(row['input_sha256'] == digest(arrays['velocity'][number])
                        and row['target_sha256'] == digest(arrays['target'][number]) and row['mask_sha256'] == digest(mask)
                        and row['output_sha256'] == digest(prediction),'Originalinput/target/mask/crossphaseoutput hashes differ')
                same_metrics(metrics,row['metrics'])
                require(set(row['destination_metrics']) == set(DESTINATIONS), 'All four destination scores required')
                for name in DESTINATIONS:
                    same_metrics(destinations[name],row['destination_metrics'][name])
                raw = row['raw_wall_nanoseconds']
                require(len(raw) == 10 and all(type(v) is int and v > 0 for v in raw)
                        and row['median_ms'] == statistics.median(raw)/1e6,'Raw primary timings differ')
                require(row['instrumented_first_call_ns'] > 0 and len(row['instrumented_warmup_ns']) == 5
                        and all(v > 0 for v in row['instrumented_warmup_ns']),'Separate first/warmup records required')
                for pair_number,pair in enumerate(row['calibration']):
                    order = ['on','off'] if (phase+number+pair_number)%2 == 0 else ['off','on']
                    require(pair['pair_index'] == pair_number and pair['order'] == order and pair['on_ns'] > 0 and pair['off_ns'] > 0
                            and pair['on_minus_off_ns'] == pair['on_ns']-pair['off_ns'],'Signed alternating calibration differs')
                    overheads.append(pair['on_minus_off_ns'])
                require(len(row['calibration']) == 4,'Exactly4pairedcalibrations required')
                samples.extend(raw)
            rows.append(dict(pattern_index=number,population='fitting' if number < 64 else 'consumed_development',metrics=metrics,
                destination_metrics=destinations,
                input_sha256=index[0][number]['input_sha256'],output_sha256=digest(prediction),raw_wall_nanoseconds=samples,
                pooled_median_ms=statistics.median(samples)/1e6,minimum_ms=min(samples)/1e6,maximum_ms=max(samples)/1e6,
                phase_medians_ms=[index[p][number]['median_ms'] for p in (0,1)],
                paired_observer_overhead_median_ms=statistics.median(overheads)/1e6))
            all_overheads.extend(overheads)
    groups = {}
    for name,selected in [('all',rows),('fitting',rows[:64]),('consumed_development',rows[64:])]:
        groups[name] = dict(metric_summary([row['metrics'] for row in selected]),
            destinations={kind:metric_summary([row['destination_metrics'][kind] for row in selected]) for kind in DESTINATIONS},
            median_case_GPU_ms=statistics.median(row['pooled_median_ms'] for row in selected),
            maximum_case_median_GPU_ms=max(row['pooled_median_ms'] for row in selected))
    # Read original CPUprobe only. The arithmetic below isdifference/scalarbound,
    # never a CPUlearned convolution orCPUmodel execution.
    observed_file(ROOT,protocol['original_probe']['path'],protocol['original_probe'])
    probe = np.load(ROOT/protocol['original_probe']['path'],allow_pickle=False)
    with np.load(ROOT/entry['outputs']['weights.npz']['path'],allow_pickle=False) as weights:
        weight_l1 = max(float(np.abs(weights[name].astype('<f8')).reshape(weights[name].shape[0],-1).sum(axis=1).max()) for name in weights.files)
    unit = float(np.finfo(np.float32).eps)/2;gamma = 81*unit/(1-81*unit)
    with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
        for number in (0,1):
            maximum_difference = float(np.abs(archive[f'pattern{number:03d}'].astype('<f8')-probe[number].astype('<f8')).max())
            bound = 2*gamma*float(np.abs(arrays['velocity'][number]).max())*weight_l1+float(np.finfo(np.float32).tiny)
            require(maximum_difference <= bound,'Original publishedCPUprobe exceeds declaredF32roundoff bound')
            for phase in (0,1):
                actual = index[phase][number]['original_CPU_probe_comparison']
                require(actual == dict(saved_CPU_probe_maximum_difference=maximum_difference,predeclared_absolute_roundoff_bound=bound,CPU_probe_regenerated=False),
                        'OriginalCPUprobe comparison/scalar bound differs')
    require('torch' not in sys.modules,'Independent score reducer imported a model runtime')
    save(REPORT,dict(status='original_point_80pattern_GPU_benchmark_independently_reduced',created_unix=time.time(),
        fit_item_id=protocol['fit_item_id'],cases=80,groups=groups,rows=rows,fresh_processes=2,primary_calls=1600,logical_calls=3840,
        calibration_pairs=640,paired_observer_overhead_median_ms=statistics.median(all_overheads)/1e6,
        paired_negative_overhead_differences=sum(v < 0 for v in all_overheads),overhead_subtracted=False,
        prediction_archive=archive_spec,raw_results=raw_results,protocol=binding(protocol_path),owner=binding(folder/'status.json'),
        material=protocol['material'],original_model_source=next(spec for spec in entry['sources'] if spec['original_name']=='point_sampling_model.py'),
        original_weights=entry['outputs']['weights.npz'],inference_device='cuda',GPU=protocol['expected_GPU'],
        timing_units='ms/local_point_request',published_dtype='float32',reference_dtype='float64',physical_simulation_steps=0,
        native_counts_xyz=[12,12,12],padded_tensor_spatial_shape=[13,13,13],cells_per_metre=None,
        destination_channel_order=list(DESTINATIONS), independent_metric_groups=400,
        timing_denominator='One12channel local sampling request;not divided bydestination count orphysical timesteps',
        device_scope='Constructor/first/warmup/on-calibration observedCUDA;primary/off observer removed,unchangedCUDAcode/inputs/weights andexactrepeat/crossphaseoutputs. NoCPUneuralforward.',
        population='64fitting/16previously-consumeddevelopment synthetic native-qualified patterns;not independentholdout,physical trajectory drift orpressure error.',
        original_CPU_probe_read_without_forward=True,all_original_tensors_and_architecture_unchanged=True,
        snapshot_resources_not_continuous_exclusivity=True,CPU_model_forwards=0,new_parameter_updates=0,new_reference_generation=False,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Original point972publishedF32coefficients/80patterns/1600primarycalls independently scored;not a full simulation step.',flush=True)


if __name__ == '__main__':
    run()
