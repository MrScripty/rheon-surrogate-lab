"""Export case/output scalar evidence without reading any prediction fields.

The export is a compact plotting view, not a second benchmark authority. Bound
source JSON retains original protocols, timings, failures and normalization.
No model, reference array, prediction array or private credential is opened.
"""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from retain_portfolio_scalar_statistics import identity, error_values
from portfolio_statistics import physical_metrics


ROLE_CONTRACTS = {
    'pressure_request': (
        'Captured model-specific pressure request; exact input_names retained per model.',
        'One pressure proposal; no complete physical simulation advance.',
        'pressure128'),
    'complete_state': (
        'Current native pressure, staggered XYZ velocity, tracer, geometry and known controls.',
        'K complete physical outputs, each with pressure, XYZ velocity and tracer.',
        'generator64'),
    'guided_component_complete_composition': (
        'Current native pressure/velocity/tracer and prescribed controls/events; captured guided composition.',
        'All K complete physical outputs; every lead is evaluated separately.',
        'guided2880'),
    'full_state_readout_component': (
        'Supplied causal frozen-backbone stage fields; three/four-stage readout context.',
        'One next physical pressure/velocity/tracer output; backbone preparation is a separate dependency.',
        'readout58'),
    'full_state_correction_component': (
        '42 normalized channels from supplied fine-history internal group context and frozen coupled response.',
        'Four carrier deltas and coupled pressure/velocity/tracer for a partial internal group; not a full step.',
        'correction783'),
    'modal_pressure_component': (
        'Native RHS, fluid mass density and cell spacing; captured full modal basis.',
        'One pressure solution; basis/gain preparation and request timing have explicit boundaries.',
        'native_pressure144'),
    'shared_spectral_pressure_component': (
        'Native RHS, fluid mass density and cell spacing; declared prepared/unprepared shared response.',
        'One pressure solution; distinct input/timing boundary from the pressure CNN request lane.',
        'native_pressure144'),
    'causal_future_q_component': (
        'Normalized current velocity, known forcing and physical h/dt/rho/mu scalars.',
        'Normalized future pressure RHS q; no pressure inverse or full state advancement.',
        'future_q2048'),
    'recurrent_transport_component': (
        'Original local carrier, velocity/position/context requests; midpoint boundary-clamped targets.',
        'Local sampler middle/end predictions, not a full simulation step.',
        'sampler576'),
    'point_sampling_component': (
        'Native-qualified synthetic local point requests; fitting/development pattern membership retained.',
        'Local VX/VY/VZ-face and tracer-cell sampling outputs.',
        'point80'),
    'viscosity_component': (
        'Native-qualified synthetic local viscosity requests; fitting/development pattern membership retained.',
        'Local viscosity response on free faces; not a full simulation step.',
        'viscosity80'),
}


def scalar_tree(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and scalar_tree(v) for k, v in value.items())
    return value is None or isinstance(value, (str, bool)) or (
        type(value) in (int, float) and math.isfinite(value))


def timing_summary(values):
    values = list(values)
    require(values and all(math.isfinite(v) and v >= 0 for v in values), 'Finite measured timings required')
    ordered = sorted(values)
    def percentile(q):
        position = q * (len(ordered) - 1)
        lo = math.floor(position)
        return ordered[lo] + (ordered[math.ceil(position)] - ordered[lo]) * (position - lo)
    return dict(samples=len(values), min=ordered[0], max=ordered[-1],
                mean=statistics.fmean(values), median=statistics.median(values),
                population_std=statistics.pstdev(values),
                p90=percentile(.9), p95=percentile(.95), p99=percentile(.99))


def geometry(context):
    result = identity(context)
    for name in ('input_dt_s', 'fluid_density_kg_m3', 'physical_internal_group_dt_s',
                 'block_advance_s', 'pressure_route'):
        if name in context:
            result[name] = context[name]
    counts = result.get('counts_xyz')
    if counts is not None:
        require(len(counts) == 3 and all(type(v) is int and v > 0 for v in counts), 'Grid dimensions required')
        result.update(voxel_count=math.prod(counts), aspect_ratio_xyz=[v/min(counts) for v in counts])
        if result.get('cells_per_metre'):
            result['physical_extents_m'] = [v/result['cells_per_metre'] for v in counts]
    return result


def run():
    output = RUN/'portfolio-case-statistics-01.jsonl.gz'
    report_path = DOC/'reports/plot-statistics-01.json'
    require(not output.exists() and not report_path.exists(), 'Preserve original plotting evidence')
    coverage_path = DOC/'reports/common-fit-benchmark-coverage-13.json'
    scalar_path = DOC/'reports/portfolio-scalar-statistics-01.json'
    coverage = read(coverage_path)
    scalar = read(scalar_path)
    expected = {m['fit_item_id']: m for m in scalar['models']}
    sources = {}
    cache = {}

    def load(spec):
        if isinstance(spec, Path):
            spec = binding(spec)
        path = spec['path']
        require(path.endswith('.json') and not Path(path).is_absolute(), 'Only bound repository scalar JSON is admitted')
        actual = binding(ROOT/path)
        require(actual['sha256'] == spec['sha256'], 'Source identity changed: '+path)
        sources[path] = actual
        if path not in cache:
            cache[path] = read(ROOT/path)
        return cache[path]

    named = load(DOC/'reports/pressure-common-score-01.json')
    named_timing = load(DOC/'reports/pressure-portfolio-GPU-timing-01.json')
    pressure_dimensions = {(r['dimensions']['trajectory_id'],r['dimensions']['native_frame']):r['dimensions']
        for r in named_timing['models']['C107']['rows']}
    original_timing = load(DOC/'reports/original-pressure-GPU-timing-batch-01.json')
    original_workers = defaultdict(list)
    for spec in original_timing['raw_results']:
        worker = load(spec)
        original_workers[worker['fit_item_id']].append(worker)
    manifest_models = []
    totals = Counter()
    pending_path = output.with_suffix('.pending')

    with pending_path.open('xb') as raw_output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_output, compresslevel=6, mtime=0) as compressed:
            def emit(value):
                compressed.write((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())

            def process(fit, benchmark, group, model, report, source_spec, rows, *, anchor=False):
                fid = fit['fit_item_id']
                inputs, outputs, population = ROLE_CONTRACTS[group]
                counts = Counter()
                metric_names = set()
                work = {r['case_index']: r['actual_work'] for r in model.get('actual_work_by_block', [])}
                model_metadata = dict(subject_id=fid, subject_kind='assembly_execution' if anchor else 'fit_default_state',
                    name=fit['name'], task=fit['task'], comparison_group=group, case_population=population,
                    inputs=inputs, outputs=outputs, accuracy_source=source_spec,
                    qualified_timing=bool(benchmark.get('benchmark_authority')),
                    timing_units=benchmark.get('timing_units'),
                    input_names=model.get('input_names'),
                    precision={k:model[k] for k in ('pressure_dtype','network_dtype','published_dtype','reference_dtype','tf32','autocast') if k in model},
                    measured_scope=report.get('scope', report.get('exposure')),
                    timing_scope=report.get('timing_scope', benchmark.get('timing_units')),
                    declaration_status='Role-level reviewed description; exact implementation and conditional inputs remain bound to source/contract',
                    full_contract_complete=False)
                for case_index, context, observations, timing in rows:
                    dims = geometry(context)
                    if 'input_dt_s' in context:
                        dims['input_dt_s'] = context['input_dt_s']
                    call = dict(subject_id=fid, case_population=population,
                        case_index=case_index, dimensions=dims, source_path=source_spec['path'], outputs=[])
                    if case_index in work:
                        call['observed_work'] = work[case_index]
                        call['work_scope'] = report['work_observer_scope']
                        counts['calls_with_partial_work_observations'] += 1
                    if 'actual_internal_advances' in context:
                        call['actual_internal_advances'] = context['actual_internal_advances']
                        counts['calls_with_actual_internal_advances'] += 1
                    for observed in observations:
                        metrics = observed['metrics']
                        require(scalar_tree(metrics), 'Only scalar metric dictionaries allowed; no raw fields')
                        view = dict(statistics=metrics)
                        if 'relative' in metrics:
                            view['plot_metrics'] = physical_metrics({'metrics': metrics})
                        else:
                            view['plot_metrics'] = error_values(metrics)
                        for name in ('destination_metrics', 'archived_zero_correction_baseline', 'unchanged_guide_metrics'):
                            if name in observed:
                                require(scalar_tree(observed[name]), 'Scalar baseline/channel metrics required')
                                view[name] = observed[name]
                        lead = observed.get('lead', observed.get('identity', {}).get('lead'))
                        if lead is not None:
                            view['lead'] = lead
                            if dims.get('K') is not None:
                                view['output_role'] = 'endpoint' if lead == dims['K'] else 'intermediate'
                                require(1 <= lead <= dims['K'], 'Physical lead lies outside K')
                            if dims.get('output_dt_s') is not None:
                                view['physical_lead_dt_s'] = lead*dims['output_dt_s']
                        if 'time_s' in observed:
                            view['reference_target_time_s'] = observed['time_s']
                        for name in ('physical_lead_dt_s', 'target_fine_step'):
                            if name in observed.get('identity', {}):
                                view[name] = observed['identity'][name]
                        call['outputs'].append(view)
                        metric_names.update(view['plot_metrics'])
                        counts['evaluated_outputs_or_requests'] += 1
                        counts['outputs_with_physical_lead'] += int(lead is not None)
                    if 'K' in dims and group in ('complete_state', 'guided_component_complete_composition'):
                        require([o.get('lead') for o in call['outputs']] == list(range(1, dims['K']+1)),
                            'Every physical output lead must be preserved in order')
                    if timing:
                        call['GPU_timing_ms'] = {k:timing_summary(v) for k,v in timing.items()}
                        counts['qualified_timing_samples'] += len(next(iter(timing.values())))
                        counts['timed_calls'] += 1
                    counts['calls'] += 1
                    for name in ('counts_xyz','cells_per_metre','input_dt_s','output_dt_s','K'):
                        counts['calls_with_'+name] += int(dims.get(name) is not None)
                    emit(call)
                if not anchor:
                    require(counts['evaluated_outputs_or_requests'] == expected[fid]['accuracy']['observations'],
                        'Plot rows do not cover original fit population: '+fid)
                    if expected[fid]['qualified_timing']:
                        require(counts['qualified_timing_samples'] == expected[fid]['timing']['observations'],
                            'Plot timing observations do not cover qualified source: '+fid)
                model_metadata.update(counts=dict(counts), measured_plot_metrics=sorted(metric_names),
                    additional_frequency_statistics=binding(DOC/'reports/recent-component-scalar-statistics-01.json')
                        if group in ('guided_component_complete_composition','full_state_readout_component','full_state_correction_component') else None,
                    deletion_disposition='paused_pending_expanded_metric_and_consumer_audit',
                    unavailable_or_incomplete=['Full-population state-change/baseline/physical diagnostics where not measured',
                        'Per-stage cost attribution for shared computations', 'Own-history drift and independent holdout',
                        'Normalized exact per-model pass schedule and complete input/output contract audit'])
                manifest_models.append(model_metadata)
                for key,value in counts.items():
                    totals[('anchor_' if anchor else 'fit_')+key] += value

            for fit in coverage['rows']:
                benchmark = fit['default_benchmark'] or fit.get('accuracy_benchmark')
                if benchmark is None:
                    continue
                group = benchmark['group']
                source = benchmark.get('accuracy_authority', benchmark.get('independent_reduction', benchmark.get('benchmark_authority')))
                report = load(source)
                rows = []
                if group == 'pressure_request':
                    ident = benchmark['identifier']
                    if ident in named['models']:
                        source = named['models'][ident]['result']
                        model = load(source)
                        timed = named_timing['models'][ident]['rows']
                        require(len(model['rows']) == len(timed), 'Pressure case populations differ')
                        for index,(row,t) in enumerate(zip(model['rows'], timed)):
                            require(row['trajectory_id'] == t['dimensions']['trajectory_id']
                                and row['native_frame'] == t['dimensions']['native_frame']
                                and row['output_sha256'] == t['output_sha256'], 'Pressure accuracy/timing join differs')
                            context = dict(row, **{k:v for k,v in t['dimensions'].items() if k not in row})
                            rows.append((index,context,[row],{'call':[v/1e6 for v in t['raw_wall_nanoseconds']]}))
                    else:
                        entry = next(r for r in report['rows'] if r['fit_item_id'] == fit['fit_item_id'])
                        source = entry['result']
                        model = load(source)
                        times = defaultdict(list)
                        hashes = {}
                        for worker in original_workers[fit['fit_item_id']]:
                            for row in worker['rows']:
                                key = (row['trajectory_id'],row['native_frame'])
                                times[key].extend(v/1e6 for v in row['raw_wall_nanoseconds'])
                                require(key not in hashes or hashes[key] == row['output_sha256'], 'Original phase outputs differ')
                                hashes[key] = row['output_sha256']
                        for index,row in enumerate(model['rows']):
                            key = (row['trajectory_id'],row['native_frame'])
                            require(hashes[key] == row['output_sha256'], 'Original timing/accuracy outputs differ')
                            context = dict(row, **{k:v for k,v in pressure_dimensions[key].items() if k not in row})
                            rows.append((index,context,[row],{'call':times[key]}))
                    process(fit,benchmark,group,model,report,source,rows)
                    continue
                if group == 'complete_state':
                    model = next(m for m in report['models'] if m['fit_item_id'] == fit['fit_item_id'])
                    for index,row in enumerate(model['pooled_rows']):
                        observations = [dict(metrics=o['values'],lead=o['lead'],time_s=o['time_s']) for o in row['metrics']]
                        rows.append((index,row,observations,{'block':row['block_ms'],'per_physical_output':row['per_output_ms']}))
                else:
                    if 'models' in report:
                        model = next(m for m in report['models'] if m['fit_item_id'] == fit['fit_item_id'])
                    elif 'modes' in report:
                        model = next(m for m in report['modes'] if m['mode'] == benchmark['cache_policy'])
                    else:
                        model = report
                    if group == 'guided_component_complete_composition':
                        blocks = defaultdict(list)
                        for row in model['rows']:
                            blocks[row['case_index']].append(row)
                        for index,values in blocks.items():
                            context = dict(values[0]['identity'])
                            context.pop('lead',None)
                            context.pop('physical_lead_dt_s',None)
                            context.pop('target_fine_step',None)
                            rows.append((index,{'identity':context},sorted(values,key=lambda r:r['lead']),None))
                    else:
                        for index,row in enumerate(model['rows']):
                            context = dict(row)
                            if 'identity' not in row:
                                context.update(cells_per_metre=report.get('cells_per_metre'), counts_xyz=report.get('native_counts_xyz'))
                            observed = row
                            if group == 'full_state_readout_component':
                                context['identity'] = dict(row['identity'],K=1,output_dt_s=1/60,internal_stages=model['stages'])
                                observed = dict(row,lead=1)
                            timing = {'call':[v/1e6 for v in row['raw_wall_nanoseconds']]} if 'raw_wall_nanoseconds' in row else None
                            rows.append((index,context,[observed],timing))
                process(fit,benchmark,group,model,report,source,rows)

            anchor_source = binding(DOC/'reports/same-case-F64-GPU-benchmark-02.json')
            anchor_report = load(anchor_source)
            for model in anchor_report['models']:
                fit = dict(fit_item_id=model['assembly_execution_id'],name=model['assembly_execution_id'],task='full_state_assembly_control')
                rows = []
                for index,row in enumerate(model['pooled_rows']):
                    observations = [dict(metrics=o['values'],lead=o['lead'],time_s=o['time_s']) for o in row['metrics']]
                    rows.append((index,row,observations,{'block':row['block_ms'],'per_physical_output':row['per_output_ms']}))
                process(fit,dict(benchmark_authority=anchor_source,timing_units=anchor_report['timing_scope']),
                    'complete_state',model,anchor_report,anchor_source,rows,anchor=True)

    require(len(manifest_models) == 207 and len(expected) == 203, '203 fits plus four real assembly controls required')
    require(totals['fit_evaluated_outputs_or_requests'] == 47584 and totals['fit_qualified_timing_samples'] == 539840,
        'Whole-portfolio accepted error/timing coverage must be unchanged')
    output_counts = Counter()
    subjects = set()
    with gzip.open(pending_path,'rt') as stream:
        for line in stream:
            case = json.loads(line)
            subjects.add(case['subject_id'])
            output_counts['calls'] += 1
            output_counts['outputs'] += len(case['outputs'])
    require(len(subjects) == 207 and output_counts['outputs'] == totals['fit_evaluated_outputs_or_requests']+totals['anchor_evaluated_outputs_or_requests'],
        'Compressed scalar export must re-read with exact subjects and output membership')
    pending_path.replace(output)
    save(report_path,dict(status='plotting_scalar_view_verified_expanded_retention_audit_incomplete_deletion_paused',
        created_unix=time.time(),producer=binding(Path(__file__)),contract=binding(DOC/'statistics-retention-contract.md'),
        portfolio_coverage=binding(coverage_path),aggregate_statistics=binding(scalar_path),
        scalar_case_payload=dict(**binding(output),bytes=output.stat().st_size,format='gzip UTF8 JSON Lines; case records with every output scalar'),
        source_reports=list(sources.values()),models=manifest_models,counts=dict(totals),
        pending_accuracy_fit_ids=coverage['pending_accuracy_fit_ids'],
        frequency_statistics=binding(DOC/'reports/recent-component-scalar-statistics-01.json'),
        temporal_policy='Score every physical lead independently; do not use endpoint error as intermediate error or allocate joint timing to leads.',
        plot_policy='Match task, input context, population, reference, metric definition and timing boundary. Keep physical K, lead, horizon and internal passes separate.',
        source_context_policy='Exact formulas/floors, failures, setup/resource/precision and method details remain in bound scalar source JSON. The view is not a substitute for these sources.',
        unavailable_policy='Null/absent is unavailable, never zero. Existing sources retain supplementary diagnostics not normalized into this view.',
        deletion_paused=True,prediction_fields_opened=0,prediction_files_deleted=0,new_model_forwards=0,
        CPU_neural_forwards=0,model_dataset_or_project_copies=0,native_mutations=0,full_portfolio_complete=False))
    print(json.dumps(dict(models=len(manifest_models),counts=dict(totals),compressed_bytes=output.stat().st_size,
        deletion_paused=True),sort_keys=True),flush=True)


if __name__ == '__main__':
    run()
