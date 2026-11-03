"""Reduce the existing150fit pressure captures to clean per-case statistics.

Only NumPy and bound reference/prediction fields are used. Generated diagnostic
records contain scalars; prediction fields are never copied or saved. Existing
captures remain in place pending the separate expanded disposal/consumer audit.
"""
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from pressure_scalar_diagnostics import measure


FOCUS_METRICS = {
    'pressure_error_rms': ('pressure_mean_free_error','rms'),
    'pressure_error_mae': ('pressure_mean_free_error','mae'),
    'pressure_error_absolute_max': ('pressure_mean_free_error','maximum_absolute'),
    'pressure_error_absolute_p95': ('pressure_mean_free_error','absolute_p95'),
    'pressure_error_absolute_p99': ('pressure_mean_free_error','absolute_p99'),
    'pressure_error_relative_l2': ('pressure_mean_free_error','relative_l2'),
    'equation_residual_rms': ('equation_residual','rms'),
    'equation_residual_mae': ('equation_residual','mae'),
    'equation_residual_absolute_max': ('equation_residual','maximum_absolute'),
    'equation_residual_relative_l2': ('equation_residual','relative_l2'),
    'prediction_pressure_mean_free_rms': ('prediction_pressure_mean_free','rms'),
    'reference_pressure_mean_free_rms': ('reference_pressure_mean_free','rms'),
    'input_base_pressure_mean_free_rms': ('input_base_mean_free','rms'),
    'prediction_minus_supplied_base_rms': ('prediction_minus_supplied_base','rms'),
    'reference_minus_supplied_base_rms': ('reference_minus_supplied_base','rms'),
    'supplied_base_pressure_error_rms': ('supplied_base_pressure_error','rms'),
    'supplied_base_equation_relative_l2': ('supplied_base_equation_residual','relative_l2'),
}


def run():
    import numpy as np
    from build_plot_statistics import timing_summary
    admission_path = RUN/'pressure-statistics-admission-01.json'
    admission = read(admission_path)
    output = RUN/'pressure-field-statistics-01.jsonl.gz'
    report_path = DOC/'reports/pressure-field-statistics-01.json'
    require(not output.exists() and not report_path.exists(), 'Preserve original extraction evidence')
    os.sched_setaffinity(0,{4 if 4 in os.sched_getaffinity(0) else min(os.sched_getaffinity(0))})
    began = time.monotonic()
    sources = {}

    def bound_read(spec):
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Scalar evidence changed')
        sources[spec['path']] = dict(path=spec['path'],sha256=spec['sha256'])
        return read(ROOT/spec['path'])

    contract = bound_read(admission['frozen_contract'])
    manifest = bound_read(admission['packet_manifest'])
    arrays = {}
    for shard in admission['packet_shards']:
        path = ROOT/shard['path']
        require(path.stat().st_size == shard['bytes'] and binding(path)['sha256'] == shard['sha256'], 'Original packet shard changed')
        with np.load(path,allow_pickle=False) as archive:
            for name in archive.files:
                value = archive[name]
                spec = manifest['arrays'][name]
                require(name not in arrays and value.dtype.str == spec['dtype'] and list(value.shape) == spec['shape']
                    and value.nbytes == spec['bytes'] and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256']
                    and np.isfinite(value).all(), 'Original packet field changed')
                arrays[name] = value
    require(set(arrays) == set(manifest['arrays']), 'Complete original packet inventory required')
    cases = []
    contexts = []
    for parent_index,parent in enumerate(manifest['parents']):
        for slot,frame in enumerate(parent['frames']):
            index = len(cases)
            frozen = contract['cases'][index]
            original = frozen['original_case']
            sample = {name:arrays[f'p{parent_index:04d}_{name}'][slot] for name in ('base','native_input_residual','reference','rhs')}
            for name,value in sample.items():
                require(value.dtype.str == '<f8' and list(value.shape) == parent['counts_xyz'][::-1]
                    and hashlib.sha256(value.tobytes()).hexdigest() == frame['sha256'][name], 'Original request frame changed')
            sample.update(cell_size_m=parent['cell_size_m'],rho=parent['density_kg_m3'],dt=parent['entry']['dt_s'])
            for name,value in sample.items():
                if name in original['input_sha256']:
                    raw = value.tobytes() if isinstance(value,np.ndarray) else np.asarray([value],dtype='<f8').tobytes()
                    require(hashlib.sha256(raw).hexdigest() == original['input_sha256'][name], 'Frozen pressure input hash changed')
            require(hashlib.sha256(sample['reference'].tobytes()).hexdigest() == original['reference_pressure_sha256']
                and hashlib.sha256(sample['rhs'].tobytes()).hexdigest() == original['reference_rhs_sha256']
                and parent['entry']['trajectory_id'] == original['trajectory_id'] and frame['index'] == original['native_frame'],
                'Frozen case/reference identity differs')
            cases.append(sample)
            contexts.append(dict(case_index=index,identity=original,
                dimensions=frozen['dimensions'],source_population=frozen['source_population'],
                scene=frozen['scene'],physical_timestamp_s=frozen['physical_timestamp_s'],
                force_metadata_policy='Original native force configuration fields, including frequency_hz, retained without relabeling as pulse/output cadence'))
    require(len(cases) == 128, 'Complete128case bank required')
    models = []
    scored_archives = {}
    comparison_count = 0
    maximum_difference = defaultdict(float)
    pending = output.with_suffix('.pending')
    with pending.open('xb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,mtime=0,compresslevel=6) as compressed:
            for item in admission['fits']:
                raw = bound_read(item['accuracy_source'])
                capture = item['prediction_archive']
                path = ROOT/capture['path']
                require(binding(path)['sha256'] == capture['sha256'], 'Prediction archive changed')
                cached = scored_archives.get(capture['path'])
                if cached is None:
                    cached = []
                    with np.load(path,allow_pickle=False) as archive:
                        require(set(archive.files) == {f'case{i:03d}_pressure' for i in range(128)}, 'Only complete pressure field captures admitted')
                        for index,(sample,prior) in enumerate(zip(cases,raw['rows'],strict=True)):
                            original = contexts[index]['identity']
                            require({k:prior[k] for k in original} == original, 'Different pressure input/target population')
                            value = archive[f'case{index:03d}_pressure']
                            require(value.dtype.str == '<f8' and tuple(value.shape) == (1,1,*sample['base'].shape)
                                and hashlib.sha256(value.tobytes()).hexdigest() == prior['output_sha256'], 'Accepted predicted field differs')
                            cached.append((prior['output_sha256'],measure(value,sample,original['counts_xyz'])))
                    scored_archives[capture['path']] = cached
                population = defaultdict(list)
                improved = 0
                floor_counts = defaultdict(int)
                for index,((digest,values),prior) in enumerate(zip(cached,raw['rows'],strict=True)):
                    require(digest == prior['output_sha256'] and {k:prior[k] for k in contexts[index]['identity']} == contexts[index]['identity'],
                        'Reused archive belongs to a different model request/output')
                    measured = dict(pressure_absolute_rms=values['pressure_mean_free_error']['rms'],
                        pressure_absolute_max=values['pressure_mean_free_error']['maximum_absolute'],
                        pressure_mean_free_relative_l2=values['pressure_mean_free_error']['relative_l2'],
                        equation_relative_l2=values['equation_residual']['relative_l2'])
                    for name,actual in measured.items():
                        key = 'common_equation_relative_l2' if name == 'equation_relative_l2' and name not in prior['metrics'] else name
                        expected = prior['metrics'][key]
                        difference = abs(actual-expected)
                        require(difference <= 1e-14+1e-11*abs(expected), 'Independent scalar rescoring differs: '+item['identifier']+'/'+name)
                        maximum_difference[name] = max(maximum_difference[name],difference)
                        comparison_count += 1
                    for name,(field,metric) in FOCUS_METRICS.items():
                        population[name].append(values[field][metric])
                    improved += int(values['pressure_better_than_supplied_base'])
                    for name in ('pressure_mean_free_error','equation_residual','supplied_base_pressure_error','supplied_base_equation_residual'):
                        floor_counts[name] += int(values[name]['denominator_floor_active'])
                    record = dict(fit_item_id=item['fit_item_id'],case_index=index,statistics=values)
                    compressed.write((json.dumps(record,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n').encode())
                summaries = {name:timing_summary(v) for name,v in population.items()}
                for name,summary in summaries.items():
                    summary['minimum_case'] = population[name].index(summary['min'])
                    summary['maximum_case'] = population[name].index(summary['max'])
                models.append(dict(**item,observed_cases=128,statistics=summaries,
                    pressure_RMS_better_than_supplied_base_cases=improved,denominator_floor_counts=dict(floor_counts),
                    velocity_and_tracer_accuracy='not_applicable_pressure_only_request',
                    drift='not_applicable_single_pressure_request;no physical advancement claim',
                    missing=['Own-history effects when embedded in a complete generator',
                        'Exact derived feature/iteration/pass contract still requires per-source review'],
                    prediction_capture_deleted=False))
                if len(models)%25 == 0:
                    print(len(models),'pressure fit populations reduced to scalars',flush=True)
    counters = defaultdict(int)
    with gzip.open(pending,'rt') as stream:
        for line in stream:
            record = json.loads(line)
            counters[record['fit_item_id']] += 1
    require(len(counters) == 150 and set(counters.values()) == {128} and comparison_count == 76800
        and 'torch' not in sys.modules, 'Complete scalar population and NumPy-only extraction required')
    pending.replace(output)
    save(report_path,dict(status='150pressure_fit_fields_reduced_to_verified_scalars_captures_retained',created_unix=time.time(),
        producer=binding(Path(__file__)),helper=binding(Path(__file__).with_name('pressure_scalar_diagnostics.py')),
        admission=binding(admission_path),source_reports=list(sources.values()),packet_shards=admission['packet_shards'],
        models=models,case_metadata=contexts,
        scalar_case_payload=dict(**binding(output),bytes=output.stat().st_size,format='gzip UTF8 JSON Lines;scalar diagnostic dictionaries only'),
        fits=150,unique_prediction_archives_scored=len(scored_archives),evaluated_case_outputs=19200,
        accepted_metric_comparisons=comparison_count,maximum_absolute_rescoring_difference=dict(maximum_difference),
        definitions=dict(pressure_error='Mean-free prediction-minus-reference; L2 denominator max(reference mean-free L2,1e-30)',
            equation='Original sealed finite-volume stencil, independently evaluated in NumPy with supplied h/rho and fixed RHS; original accepted metric retained separately',
            baseline='Supplied request base pressure, not a temporal persistence assumption; same gauge/operator/target',
            change='Mean-free prediction-minus-base and reference-minus-base; describes correction magnitude, not accuracy by itself',
            gradient='XYZ cell-neighbour pressure-gradient errors; not velocity prediction accuracy',
            percentiles='Linear interpolation on spatial absolute errors within each case; aggregate percentiles over case scalars are distinct',
            summaries='Case-weighted min/max/mean/median/population_std/p90/p95/p99/count; signed/state min/max have no best/worst interpretation',
            geometry_and_frequency='Join case_index to case_metadata, retaining density/grid and original force fields separately; no inferred physical timestamps'),
        declared_output='One pressure field per request; no velocity/tracer/physical simulation outputs',
        scope=contract['scope_id'],exposure=contract['exposure'],
        CPU_affinity=list(os.sched_getaffinity(0)),fixed_scoring_wall_seconds=time.monotonic()-began,
        new_model_forwards=0,CPU_neural_forwards=0,Torch_imported=False,new_reference_generation=0,
        field_copies_saved=0,prediction_files_deleted=0,deletion_paused=True,native_mutations=0,full_portfolio_complete=False))
    print('Verified150fits/19200outputs/76800metric checks;',len(scored_archives),'unique archives;no fields saved or deleted.',flush=True)


if __name__ == '__main__':
    run()
