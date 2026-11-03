"""NumPy-only independent accuracy/timing reduction of eight modal GPU workers."""
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from modal_component_data import load_fields, digest
from reduce_original_train_pressure_accuracy import metrics as independent_metrics, equation

METRICS = ('common_equation_relative_l2','pressure_mean_free_relative_l2','pressure_absolute_rms',
           'pressure_absolute_max','pressure_gauge_absolute_rms','pressure_gauge_absolute_max')


def summarize(rows):
    return dict(cases=len(rows),worst={name:max(row['metrics'][name] for row in rows) for name in METRICS},
        median_case_GPU_ms=statistics.median(row['pooled_median_ms'] for row in rows),
        maximum_case_median_GPU_ms=max(row['pooled_median_ms'] for row in rows),
        maximum_native_reference_equation_relative_l2=max(row['native_reference_equation_relative_l2'] for row in rows),
        pressure_floor_active_cases=sum(row['metrics']['pressure_floor_active'] for row in rows),
        equation_floor_active_cases=sum(row['metrics']['equation_floor_active'] for row in rows))


def run():
    output = DOC/'reports/modal-component-GPU-benchmark-01.json'
    require(not output.exists(), 'Preserve original modal reduction')
    protocol_path = RUN/'modal-component-GPU-protocol-01.json';protocol = read(protocol_path)
    folder = ROOT/protocol['output_root'];owner = read(folder/'status.json')
    require(owner['status'] == 'all_eight_original_modal_GPU_workers_terminal' and not owner['pending']
            and owner['active_child'] is None and len(owner['qualified']) == 8
            and not Path(f'/proc/{owner["pid"]}').exists() and owner['protocol'] == binding(protocol_path),
            'Original eight-worker owner must be qualified and terminal')
    for name,sha in protocol['source_sha256'].items():
        with (ROOT/name).open('rb') as stream:
            require(hashlib.file_digest(stream,'sha256').hexdigest() == sha, 'Frozen modal source/input changed: '+name)
    require(binding(ROOT/protocol['contract']['path']) == protocol['contract'], 'Modal admission changed')
    contract = read(ROOT/protocol['contract']['path']);fields = load_fields(contract,protocol)
    phases,processes = {},set()
    for item in owner['qualified']:
        path = ROOT/item['result']['path'];data = read(path)
        require(binding(path) == item['result'] and data['status'] == 'original_modal_GPU_phase_qualified'
                and data['fit_item_id'] == item['fit_item_id'] and data['phase'] == item['phase']
                and data['pid'] == item['pid'] and data['pid'] not in processes and not Path(f'/proc/{data["pid"]}').exists()
                and data['protocol'] == binding(protocol_path), 'Original independent process binding differs')
        processes.add(data['pid']);key = (item['fit_item_id'],item['phase'])
        require(key not in phases and data['cases'] == 144 and len(data['rows']) == 144
                and [row['case_index'] for row in data['rows']] == (list(range(144)) if item['phase'] == 0 else list(reversed(range(144))))
                and data['primary_calls'] == 1440 and data['logical_calls'] == 3456 and data['calibration_pairs'] == 576,
                'Exact phase membership/order/call counts required')
        require(data['inference_device'] == 'cuda' and data['GPU'] == protocol['expected_GPU']
                and data['host_cpu'] == 4 and data['threads'] == data['interop_threads'] == 1
                and data['timing_units'] == 'ms/native_RHS_pressure_request' and data['physical_simulation_steps'] == 0
                and data['CPU_neural_forwards'] == 0 and not data['learned_gain_cache'] and not data['reference_labels_forwarded']
                and data['learned_tensors_and_architecture_unchanged'] and data['primary_observer_calls'] == 0
                and not data['primary_forward_observer_enabled'], 'Original model/device/timing boundary differs')
        observer = data['device_observation'];counts = observer['calls_by_phase']
        require(observer['closed'] and not observer['observer_attached'] and not observer['forbidden_forward_attempts']
                and counts['first_call'] == {'LearnedModalPressure':144}
                and counts['warmup'] == {'LearnedModalPressure':720}
                and counts['calibration_on'] == {'LearnedModalPressure':576}
                and not counts.get('primary') and not counts.get('calibration_off'), 'Actual CUDA observer scopes differ')
        require(len(data['resource_observations']) == 5 and all(not row['other_compute_pids'] for row in data['resource_observations'])
                and data['peak_CUDA_allocated_bytes'] <= protocol['maximum_CUDA_allocated_bytes']
                and data['peak_RSS_bytes'] <= protocol['maximum_RSS_bytes'], 'Resource qualification differs')
        require(all(name in protocol['source_sha256'] for name in data['source_coverage']['lab_python']), 'Unbound imported lab source')
        phases[key] = data
    require(set(phases) == {(fid,p) for fid in protocol['fit_ids'] for p in (0,1)}, 'All four original fit/phase pairs required')
    models = []
    for selected in contract['fits']:
        fid = selected['fit_item_id'];first,second = (phases[(fid,p)] for p in (0,1))
        require(first['prediction_archive'] == second['prediction_archive'] and first['weights'] == second['weights'] == selected['weights']
                and first['architecture'] == second['architecture'] and first['basis_hashes'] == second['basis_hashes'],
                'Original weights/architecture/prescribed bases and one reused archive required')
        archive_spec = first['prediction_archive']
        require(binding(ROOT/archive_spec['path']) == archive_spec, 'Retained GPU fields changed')
        lookup = [{row['case_index']:row for row in phase['rows']} for phase in (first,second)]
        rows,overheads = [],[]
        with np.load(ROOT/archive_spec['path'],allow_pickle=False) as archive:
            require(set(archive.files) == {f'case{i:03d}' for i in range(144)}, 'Complete original field archive required')
            for index,case in enumerate(fields):
                pressure = archive[f'case{index:03d}'];reference,rhs = case['reference'],case['rhs']
                require(pressure.dtype.str == '<f8' and pressure.shape == rhs.shape and np.isfinite(pressure).all()
                        and pressure[0,0,0] == 0., 'Finite original physical F64/cell-zero pressure required')
                metrics = independent_metrics(pressure[None,None],reference[None,None],rhs[None,None],case['h'],case['rho'])
                raw_difference = pressure-reference
                rhs_norm = float(np.linalg.norm(rhs));reference_norm = float(np.linalg.norm(reference-reference.mean()))
                metrics.update(pressure_gauge_absolute_rms=float(np.sqrt(np.mean(raw_difference**2))),
                    pressure_gauge_absolute_max=float(np.abs(raw_difference).max()),equation_rhs_l2=rhs_norm,
                    pressure_reference_mean_free_l2=reference_norm,equation_floor_active=rhs_norm < 1e-30,pressure_floor_active=reference_norm < 1e-30)
                samples,calibration = [],[]
                for phase in (0,1):
                    row = lookup[phase][index]
                    require(row['identity'] == case['identity'] and row['input_rhs_sha256'] == case['rhs_sha256']
                            and row['reference_pressure_sha256'] == case['reference_sha256'] and row['output_sha256'] == digest(pressure),
                            'Original physical input/reference/output identity differs')
                    require(set(metrics) == set(row['metrics']), 'Same physical metric fields required')
                    for name,value in metrics.items():
                        expected = row['metrics'][name]
                        require(value == expected if isinstance(value,bool) else math.isclose(value,expected,rel_tol=2e-12,abs_tol=1e-300),
                                'Independent native pressure score differs: '+name)
                    raw = row['raw_wall_nanoseconds']
                    require(len(raw) == 10 and all(type(v) is int and v > 0 for v in raw)
                            and row['median_ms'] == statistics.median(raw)/1e6 and row['instrumented_first_call_ns'] > 0
                            and len(row['instrumented_warmup_ns']) == 5 and all(v > 0 for v in row['instrumented_warmup_ns']),
                            'Raw primary and separate first/warmup timing differs')
                    require(len(row['calibration']) == 4, 'Four paired calibrations required')
                    for number,pair in enumerate(row['calibration']):
                        require(pair['pair_index'] == number and pair['order'] == (['on','off'] if (phase+index+number)%2 == 0 else ['off','on'])
                                and pair['on_ns'] > 0 and pair['off_ns'] > 0 and pair['on_minus_off_ns'] == pair['on_ns']-pair['off_ns'],
                                'Signed alternating observer calibration differs')
                        calibration.append(pair['on_minus_off_ns'])
                    samples.extend(raw)
                native_residual = equation(reference[None,None],case['h'],case['rho'])-rhs[None,None]
                rows.append(dict(case_index=index,identity=case['identity'],metrics=metrics,
                    native_reference_equation_relative_l2=float(np.linalg.norm(native_residual))/max(rhs_norm,1e-30),
                    input_rhs_sha256=case['rhs_sha256'],reference_pressure_sha256=case['reference_sha256'],output_sha256=digest(pressure),
                    raw_wall_nanoseconds=samples,pooled_median_ms=statistics.median(samples)/1e6,
                    phase_medians_ms=[lookup[p][index]['median_ms'] for p in (0,1)],
                    paired_observer_overhead_median_ms=statistics.median(calibration)/1e6))
                overheads.extend(calibration)
        groups = dict(overall=summarize(rows),by_cells_per_metre={},by_counts_xyz={},by_shape_and_density={})
        for row in rows:
            identity = row['identity'];density = str(identity['cells_per_metre']);shape = 'x'.join(map(str,identity['counts_zyx'][::-1]))
            for group,key in [('by_cells_per_metre',density),('by_counts_xyz',shape),('by_shape_and_density',f'{shape}@{density}')]:
                groups[group].setdefault(key,[]).append(row)
        for name in ('by_cells_per_metre','by_counts_xyz','by_shape_and_density'):
            groups[name] = {key:summarize(value) for key,value in groups[name].items()}
        models.append(dict(fit_item_id=fid,weights=selected['weights'],groups=groups,rows=rows,prediction_archive=archive_spec,
            paired_observer_overhead_median_ms=statistics.median(overheads)/1e6,paired_negative_overhead_differences=sum(v < 0 for v in overheads),
            original_fit_extra_query_rows=selected['original_extra_query_rows'],raw_results=[item['result'] for item in owner['qualified'] if item['fit_item_id'] == fid]))
    require('torch' not in sys.modules, 'Independent reducer imported a learned runtime')
    save(output,dict(status='four_original_modal_GPU_benchmarks_independently_reduced',created_unix=time.time(),models=models,
        fits=4,cases_per_fit=144,fresh_processes=8,primary_calls=11520,logical_calls=27648,paired_calibration_pairs=4608,
        inference_device='cuda',GPU=protocol['expected_GPU'],timing_units='ms/native_RHS_pressure_request',physical_simulation_steps=0,
        primary_observer_removed=True,observer_overhead_subtracted=False,learned_gain_cache=False,
        scope=contract['population'],density_and_grid_groupings_separate=True,
        latency_not_same_as_150_CNN_request_boundary=True,independent_holdout=False,CPU_neural_forwards=0,
        new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,
        protocol=binding(protocol_path),contract=protocol['contract'],owner=binding(folder/'status.json'),
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    for model in models:
        summary = model['groups']['overall']
        print(model['fit_item_id'],summary['median_case_GPU_ms'],'GPUms/nativeRHSrequest;worstP',summary['worst']['pressure_mean_free_relative_l2'],flush=True)


if __name__ == '__main__':
    run()
