"""NumPy-only independent scoring of both sampler stages and complete GPU timing."""
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from modal_component_benchmark import verify_sources
from modal_component_data import digest
from sampler_component_data import load_groups


def score_stage(prediction,target):
    require(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 3
            and np.isfinite(prediction).all() and np.isfinite(target).all(), 'Finite physical XYZvectors required')
    delta = prediction.astype('<f8')-target
    target_norm = math.sqrt(float(np.dot(target.ravel(),target.ravel())))
    delta_norm = math.sqrt(float(np.dot(delta.ravel(),delta.ravel())))
    point_reference = np.sqrt(np.einsum('ij,ij->i',target,target))
    point_delta = np.sqrt(np.einsum('ij,ij->i',delta,delta));floor = point_reference < 1e-30;valid = ~floor
    return dict(relative_l2=delta_norm/max(target_norm,1e-30),absolute_rms=delta_norm/math.sqrt(delta.size),
        absolute_max=float(np.abs(delta).max()),target_l2=target_norm,target_floor_active=target_norm < 1e-30,
        points=len(target),zero_target_points=int(np.count_nonzero(floor)),
        worst_point_relative_l2_nonzero_targets=float(np.max(point_delta[valid]/point_reference[valid])) if valid.any() else None,
        zero_target_worst_absolute_l2=float(np.max(point_delta[floor])) if floor.any() else None)


def summarize(rows):
    stages = {}
    for stage in ('middle','end'):
        values = [row['metrics'][stage] for row in rows]
        valid = [value for value in values if not value['target_floor_active']]
        point_relative = [value['worst_point_relative_l2_nonzero_targets'] for value in values if value['worst_point_relative_l2_nonzero_targets'] is not None]
        zero_error = [value['zero_target_worst_absolute_l2'] for value in values if value['zero_target_worst_absolute_l2'] is not None]
        stages[stage] = dict(groups=len(values),target_floor_active_groups=len(values)-len(valid),
            worst_relative_l2_nonzero_target_groups=max((value['relative_l2'] for value in valid),default=None),
            worst_absolute_rms=max(value['absolute_rms'] for value in values),worst_absolute_component=max(value['absolute_max'] for value in values),
            points=sum(value['points'] for value in values),zero_target_points=sum(value['zero_target_points'] for value in values),
            worst_point_relative_l2_nonzero_targets=max(point_relative,default=None),zero_target_worst_absolute_l2=max(zero_error,default=None))
    return dict(groups=len(rows),points=sum(row['identity']['points'] for row in rows),stages=stages,
        median_case_GPU_ms=statistics.median(row['pooled_median_ms'] for row in rows),
        maximum_case_median_GPU_ms=max(row['pooled_median_ms'] for row in rows))


def run():
    output = DOC/'reports/sampler-component-GPU-benchmark-01.json';require(not output.exists(), 'Preserve original sampler reduction')
    protocol_path = RUN/'sampler-component-GPU-protocol-01.json';protocol = read(protocol_path);verify_sources(protocol)
    folder = ROOT/protocol['output_root'];owner = read(folder/'status.json')
    require(owner['status'] == 'both_original_sampler_GPU_phases_terminal' and not owner['pending']
            and owner['active_child'] is None and len(owner['qualified']) == 2
            and not Path(f'/proc/{owner["pid"]}').exists() and owner['protocol'] == binding(protocol_path), 'Original owner must be terminal')
    require(binding(ROOT/protocol['material']['path']) == protocol['material'], 'Original material changed')
    entry = next(row for row in read(ROOT/protocol['material']['path'])['models'] if row['fit_item_id'] == protocol['fit_item_id'])
    groups = load_groups(entry);phases = {};pids = set()
    for item in owner['qualified']:
        spec = item['result'];data = read(ROOT/spec['path']);phase = item['phase']
        require(binding(ROOT/spec['path']) == spec and data['status'] == 'original_sampler_GPU_phase_qualified'
                and data['pid'] == item['pid'] and data['pid'] not in pids and not Path(f'/proc/{data["pid"]}').exists()
                and phase == data['phase'] and phase not in phases and data['protocol'] == binding(protocol_path)
                and data['fit_item_id'] == protocol['fit_item_id'], 'Exact original process/result binding required')
        pids.add(data['pid']);phases[phase] = data
        require(data['cases'] == len(data['rows']) == 576 and data['points'] == 77998
                and [row['case_index'] for row in data['rows']] == (list(range(576)) if phase == 0 else list(reversed(range(576))))
                and data['primary_calls'] == 5760 and data['logical_calls'] == 13824 and data['calibration_pairs'] == 2304,
                'Complete original group order/point/call counts required')
        require(data['inference_device'] == 'cuda' and data['GPU'] == protocol['expected_GPU'] and data['host_cpu'] == 4
                and data['threads'] == data['interop_threads'] == 1 and data['CPU_neural_forwards'] == 0
                and data['physical_simulation_steps'] == 0 and data['timing_units'] == 'ms/local_sampler_group'
                and data['learned_tensors_and_architecture_unchanged'] and not data['reference_labels_forwarded']
                and not data['primary_forward_observer_enabled'] and data['primary_observer_calls'] == 0, 'Original sampler execution boundary differs')
        observer = data['device_observation'];counts = observer['calls_by_phase']
        require(observer['closed'] and not observer['observer_attached'] and not observer['forbidden_forward_attempts']
                and counts['first_call'] == {'LearnedSamplerHead':576} and counts['warmup'] == {'LearnedSamplerHead':2880}
                and counts['calibration_on'] == {'LearnedSamplerHead':2304} and not counts.get('primary') and not counts.get('calibration_off'),
                'Actual sampler CUDA observer scopes differ')
        require(len(data['resource_observations']) == 5 and all(not value['other_compute_pids'] for value in data['resource_observations'])
                and data['peak_CUDA_allocated_bytes'] <= protocol['maximum_CUDA_allocated_bytes']
                and data['peak_RSS_bytes'] <= protocol['maximum_RSS_bytes'], 'Resource qualification differs')
        require(all(name in protocol['source_sha256'] for name in data['source_coverage']['lab_python']), 'Actual unbound sampler source')
    require(set(phases) == {0,1} and phases[0]['architecture'] == phases[1]['architecture']
            and phases[0]['weights'] == phases[1]['weights'] == entry['outputs']['weights.npz']
            and phases[0]['prediction_archive'] == phases[1]['prediction_archive'], 'Original architecture/tensors/one reused archive required')
    archive_spec = phases[0]['prediction_archive'];require(binding(ROOT/archive_spec['path']) == archive_spec, 'Saved stage fields changed')
    lookup = {phase:{row['case_index']:row for row in data['rows']} for phase,data in phases.items()}
    rows,overheads = [],[]
    with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
        require(set(archive.files) == {f'case{i:03d}_{stage}' for i in range(576) for stage in ('middle','end')}, 'All1152stage vector arrays required')
        for index,group in enumerate(groups):
            arrays = group['arrays'];metrics = {};output_hashes = {}
            for stage in ('middle','end'):
                prediction = archive[f'case{index:03d}_{stage}']
                require(prediction.dtype.str == '<f8' and prediction.shape == (group['identity']['points'],3), 'Original F64 stage vector layout required')
                metrics[stage] = score_stage(prediction,arrays[stage]);output_hashes[stage] = digest(prediction)
            samples,calibration = [],[]
            for phase in (0,1):
                row = lookup[phase][index]
                require(row['identity'] == group['identity'] and row['input_sha256'] == {name:digest(arrays[name]) for name in ('point','patch','kind','alpha')}
                        and row['target_sha256'] == {stage:digest(arrays[stage]) for stage in ('middle','end')}
                        and row['output_sha256'] == output_hashes, 'Exact original local input/target/output identities required')
                for stage,values in metrics.items():
                    require(set(values) == set(row['metrics'][stage]), 'Same physical vector score fields required')
                    for name,value in values.items():
                        expected = row['metrics'][stage][name]
                        require(value == expected if value is None or isinstance(value,(bool,int))
                                else math.isclose(value,expected,rel_tol=2e-12,abs_tol=1e-300), 'Independent sampler stage score differs: '+name)
                raw = row['raw_wall_nanoseconds']
                require(len(raw) == 10 and all(type(value) is int and value > 0 for value in raw)
                        and row['median_ms'] == statistics.median(raw)/1e6 and row['instrumented_first_call_ns'] > 0
                        and len(row['instrumented_warmup_ns']) == 5 and all(value > 0 for value in row['instrumented_warmup_ns']), 'Raw primary/first/warmup timings differ')
                require(len(row['calibration']) == 4, 'Four paired calibrations required')
                for number,pair in enumerate(row['calibration']):
                    require(pair['pair_index'] == number and pair['order'] == (['on','off'] if (phase+index+number)%2 == 0 else ['off','on'])
                            and pair['on_ns'] > 0 and pair['off_ns'] > 0 and pair['on_minus_off_ns'] == pair['on_ns']-pair['off_ns'], 'Signed alternating calibration differs')
                    calibration.append(pair['on_minus_off_ns'])
                samples.extend(raw)
            rows.append(dict(case_index=index,identity=group['identity'],metrics=metrics,output_sha256=output_hashes,
                input_sha256=lookup[0][index]['input_sha256'],target_sha256=lookup[0][index]['target_sha256'],raw_wall_nanoseconds=samples,
                pooled_median_ms=statistics.median(samples)/1e6,phase_medians_ms=[lookup[p][index]['median_ms'] for p in (0,1)],
                paired_observer_overhead_median_ms=statistics.median(calibration)/1e6));overheads.extend(calibration)
    summaries = dict(overall=summarize(rows),by_carrier={},by_cells_per_metre={},by_counts_xyz={},by_shape_and_density={},by_carrier_and_density={})
    for row in rows:
        identity = row['identity'];carrier = str(identity['kind']);density = str(identity['cells_per_metre']);shape = 'x'.join(map(str,identity['counts_xyz']))
        for group,key in [('by_carrier',carrier),('by_cells_per_metre',density),('by_counts_xyz',shape),
                          ('by_shape_and_density',f'{shape}@{density}'),('by_carrier_and_density',f'{carrier}@{density}')]:
            summaries[group].setdefault(key,[]).append(row)
    for name in summaries:
        if name != 'overall':
            summaries[name] = {key:summarize(value) for key,value in summaries[name].items()}
    require('torch' not in sys.modules, 'Independent vector scoring imported a model runtime')
    save(output,dict(status='original_sampler_77998point_GPU_benchmark_independently_reduced',created_unix=time.time(),
        fit_item_id=protocol['fit_item_id'],groups=summaries,rows=rows,cases=576,points=77998,fresh_processes=2,
        primary_calls=11520,logical_calls=27648,paired_calibration_pairs=4608,primary_observer_removed=True,
        paired_observer_overhead_median_ms=statistics.median(overheads)/1e6,negative_paired_differences=sum(value < 0 for value in overheads),
        observer_overhead_subtracted=False,prediction_archive=archive_spec,original_weights=entry['outputs']['weights.npz'],
        raw_results=[item['result'] for item in owner['qualified']],protocol=binding(protocol_path),owner=binding(folder/'status.json'),
        material=protocol['material'],producer=binding(Path(__file__)),inference_device='cuda',GPU=protocol['expected_GPU'],
        timing_units='ms/local_sampler_group',physical_simulation_steps=0,output_units=protocol['output_units'],
        scope=protocol['population'],execution_scope='Original eagerTorch48parameterlocalhead,notcompiled full-state generator. All130-136points andbothstageoutputs ineach timedrequest.',
        CPU_neural_forwards=0,new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,full_portfolio_complete=False))
    print('Original sampler576groups/77998points/twostages independently scored;medianGPUms/localgroup',summaries['overall']['median_case_GPU_ms'],flush=True)


if __name__ == '__main__':
    run()
