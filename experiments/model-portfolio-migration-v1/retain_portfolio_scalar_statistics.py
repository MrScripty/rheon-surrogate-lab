"""Summarize all203qualified fit populations from existing scalar evidence only."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from component_material import observed_file
from portfolio_statistics import distribution, physical_metrics


ERROR_NAMES={'common_equation_relative_l2','equation_relative_l2','pressure_mean_free_relative_l2',
    'pressure_absolute_rms','pressure_absolute_max','pressure_gauge_absolute_rms','pressure_gauge_absolute_max'}


def error_values(value,prefix=''):
    result={}
    for name,item in value.items():
        key=prefix+name
        if isinstance(item,dict):result.update(error_values(item,key+'.'))
        elif type(item) in (float,int) and ('relative_l2' in name or 'absolute_rms' in name or 'absolute_max' in name):
            require(math.isfinite(item) and item>=0, 'Finite nonnegative observed error required')
            result[key]=item
    return result


def identity(row,fallback=None):
    ident=dict(row.get('identity',{}))
    if not ident:
        ident={name:row[name] for name in ('identifier','trajectory_id','native_frame','start','counts_xyz',
            'cells_per_metre','cell_size_m','input_dt_s','output_dt_s','K','population','pattern_index') if name in row}
    if 'identifier' not in ident:ident['identifier']=ident.get('trajectory_id',fallback or 'calibration-pattern-family')
    if 'counts_zyx' in ident:ident['counts_xyz']=list(reversed(ident.pop('counts_zyx')))
    if 'cells_per_metre' not in ident and 'cell_size_m' in ident:ident['cells_per_metre']=1/ident['cell_size_m']
    if 'output_dt_s' in ident:ident['output_steps_per_s']=1/ident['output_dt_s']
    # Source paths and large scene definitions remain at their original owner.
    return {k:v for k,v in ident.items() if k not in ('source','source_sha256','scene','scalar_h_dt_rho_mu_F32')}


def summarize(rows):
    require(rows, 'Actual observed scalar rows required')
    result={}
    for name in sorted(set().union(*(r['values'] for r in rows))):
        subset=[r for r in rows if name in r['values']]
        values=distribution(r['values'][name] for r in subset)
        values.pop('p95_definition')
        values.update(best_case=min(subset,key=lambda r:r['values'][name])['identity'],
            worst_case=max(subset,key=lambda r:r['values'][name])['identity'])
        result[name]=values
    return dict(observations=len(rows),distinct_named_identifiers=len({r['identity']['identifier'] for r in rows}),
        metrics=result)


def grouped(rows):
    selectors=dict(by_cells_per_metre=lambda i:i.get('cells_per_metre'),
        by_counts_xyz=lambda i:'x'.join(map(str,i['counts_xyz'])) if 'counts_xyz' in i else None,
        by_spacing_and_grid=lambda i:str(i['cells_per_metre'])+'cpm/'+'x'.join(map(str,i['counts_xyz']))
            if 'counts_xyz' in i and 'cells_per_metre' in i else None,
        by_output_steps_per_s=lambda i:i.get('output_steps_per_s'),by_K=lambda i:i.get('K'),
        by_lead=lambda i:i.get('lead'),by_input_dt_s=lambda i:i.get('input_dt_s',i.get('nominal_dt_s')),
        by_population=lambda i:i.get('population'),by_main_fit_seen=lambda i:i.get('main_q_fit_seen'),
        by_sampling_support=lambda i:i.get('sampling_support_region'))
    result={};coverage={}
    for name,select in selectors.items():
        groups=defaultdict(list)
        for row in rows:
            key=select(row['identity'])
            if key is not None:groups[str(key)].append(row)
        if groups:result[name]={key:summarize(v) for key,v in sorted(groups.items())}
        coverage[name]=dict(recorded_observations=sum(map(len,groups.values())),unavailable_observations=len(rows)-sum(map(len,groups.values())))
    return result,coverage


def run():
    output=DOC/'reports/portfolio-scalar-statistics-01.json'
    require(not output.exists(), 'Preserve first whole-portfolio scalar adoption')
    coverage_path=DOC/'reports/common-fit-benchmark-coverage-13.json';coverage=read(coverage_path)
    cache={};sources={};predictions={};models=[];extra_modes=[]
    def load(spec):
        if isinstance(spec,Path):spec=binding(spec)
        name=spec['path']
        if name not in cache:
            observed_file(ROOT,name,spec);cache[name]=read(ROOT/name);sources[name]=binding(ROOT/name)
        require(sources[name]['sha256']==spec['sha256'], 'Conflicting source authority')
        return cache[name]
    def archive(spec,source):
        if not isinstance(spec,dict) or not isinstance(spec.get('path'),str):return
        path=ROOT/spec['path']
        if not path.is_relative_to(RUN) or '/sources/' in str(path) or path.suffix not in ('.npz','.npy'):return
        item=predictions.setdefault(spec['path'],dict(identity=spec,source_reports=[]))
        require(item['identity']['sha256']==spec['sha256'], 'Conflicting prediction identity')
        item['source_reports'].append(source)
    timing_named=load(DOC/'reports/pressure-portfolio-GPU-timing-01.json')
    timing_original=load(DOC/'reports/original-pressure-GPU-timing-batch-01.json')
    original_workers=defaultdict(list)
    for spec in timing_original['raw_results']:
        value=load(spec);original_workers[value['fit_item_id']].append(value)
    common=load(DOC/'reports/pressure-common-score-01.json')
    for fit in coverage['rows']:
        fid=fit['fit_item_id'];default=fit['default_benchmark'];accuracy=fit.get('accuracy_benchmark')
        if default is None:
            if accuracy is None:continue
            prior=load(accuracy['independent_reduction'])
            model=next(m for m in prior['models'] if m['fit_item_id']==fid)
            observations=[dict(identity=identity(r),values=physical_metrics(r)) for r in model['rows']]
            summary,dimensions=grouped(observations)
            archive(model.get('prediction_archive',model.get('worst_field_capture')),accuracy['independent_reduction'])
            models.append(dict(fit_item_id=fid,name=fit['name'],task=fit['task'],comparison_scope=accuracy['scope'],
                accuracy_source=accuracy['independent_reduction'],accuracy=summarize(observations),groups=summary,
                dimension_coverage=dimensions,qualified_timing=False,timing=None,
                additional_frequency_statistics=coverage['scalar_statistics'],
                forcing_frequency_from_output_cadence_inferred=False))
            continue
        group=default['group'];score_rows=[];timing_rows=[];case_times=[];expected=default['cases']
        source=default.get('accuracy_authority',default['benchmark_authority']);report=load(source)
        gpu=None;prediction=None
        if group=='pressure_request':
            ident=default['identifier']
            if ident in common['models']:
                raw=load(common['models'][ident]['result']);t=timing_named['models'][ident]
                require(raw['weights_sha256']==t['weights_sha256'], 'Accuracy/timing weights must join')
                require(len(raw['rows'])==len(t['rows'])==expected, 'All named pressure cases required')
                for a,b in zip(raw['rows'],t['rows']):
                    require(a['output_sha256']==b['output_sha256'], 'Same pressure outputs must join timing and accuracy')
                    require(a['trajectory_id']==b['dimensions']['trajectory_id']
                            and a['native_frame']==b['dimensions']['native_frame'], 'Same named pressure case identities required')
                    vals={k:v for k,v in b['accuracy'].items() if k in ERROR_NAMES}
                    score_rows.append(dict(identity=identity(a),values=vals))
                    for index,ns in enumerate(b['raw_wall_nanoseconds']):timing_rows.append(dict(identity=dict(identity(a),sample=index),values={'GPU_call_ms':ns/1e6}))
                    case_times.append(dict(identity=identity(a),values={'case_median_GPU_call_ms':b['median_ms']}))
                prediction=raw['output_archive'];gpu=raw.get('gpu_model')
            else:
                entry=next(r for r in report['rows'] if r['fit_item_id']==fid);raw=load(entry['result'])
                table={(r['trajectory_id'],r['native_frame']):r for r in raw['rows']}
                require(len(raw['rows'])==len(table)==expected, 'Every distinct original pressure case required')
                for a in raw['rows']:score_rows.append(dict(identity=identity(a),values={k:v for k,v in a['metrics'].items() if k in ERROR_NAMES}))
                workers=original_workers[fid];require(len(workers)==2, 'Two original timing workers required')
                pooled=defaultdict(list)
                for worker in workers:
                    for row in worker['rows']:
                        key=(row['trajectory_id'],row['native_frame'])
                        require(row['output_sha256']==table[key]['output_sha256'], 'Original accuracy/timing hashes differ')
                        pooled[key].extend(row['raw_wall_nanoseconds'])
                require(len(pooled)==expected, 'Every original pressure timing case must join')
                for index,values in pooled.items():
                    ident=identity(table[index])
                    for sample,ns in enumerate(values):timing_rows.append(dict(identity=dict(ident,sample=sample),values={'GPU_call_ms':ns/1e6}))
                    case_times.append(dict(identity=ident,values={'case_median_GPU_call_ms':statistics.median(values)/1e6}))
                prediction=entry['prediction_archive'];gpu=raw['GPU']
        elif group=='complete_state':
            model=next(m for m in report['models'] if m['fit_item_id']==fid);gpu=model['GPU']
            require(len(model['pooled_rows'])==expected, 'Full generator case count differs')
            for row in model['pooled_rows']:
                ident=identity(row)
                for lead in row['metrics']:score_rows.append(dict(identity=dict(ident,lead=lead['lead']),values=error_values(lead['values'])))
                for sample,(block,step) in enumerate(zip(row['block_ms'],row['per_output_ms'])):
                    timing_rows.append(dict(identity=dict(ident,sample=sample),values={'GPU_block_ms':block,'GPU_per_physical_output_ms':step}))
                case_times.append(dict(identity=ident,values={'case_median_GPU_per_physical_output_ms':statistics.median(row['per_output_ms'])}))
        else:
            if 'models' in report:model=next(m for m in report['models'] if m['fit_item_id']==fid)
            elif 'modes' in report:
                model=next(m for m in report['modes'] if m['mode']==default['cache_policy'])
                for other in report['modes']:
                    if other is not model:extra_modes.append(dict(fit_item_id=fid,mode=other['mode'],source=source,
                        accuracy=summarize([dict(identity=identity(r),values=error_values(r['metrics'])) for r in other['rows']]),
                        GPU_call_ms=distribution(ns/1e6 for row in other['rows'] for ns in row['raw_wall_nanoseconds'])))
            else:model=report
            require(len(model['rows'])==expected, 'Original component request count differs')
            gpu=report.get('GPU');prediction=model.get('prediction_archive')
            for row in model['rows']:
                ident=identity(row)
                if 'identity' not in row:
                    ident.update(cells_per_metre=report.get('cells_per_metre'),counts_xyz=report.get('native_counts_xyz'))
                values=error_values(row['metrics'])
                if 'destination_metrics' in row:values.update(error_values(row['destination_metrics'],'destination.'))
                score_rows.append(dict(identity=ident,values=values))
                for sample,ns in enumerate(row['raw_wall_nanoseconds']):timing_rows.append(dict(identity=dict(ident,sample=sample),values={'GPU_call_ms':ns/1e6}))
                case_times.append(dict(identity=ident,values={'case_median_GPU_call_ms':row['pooled_median_ms']}))
        require(score_rows and timing_rows and case_times, 'Actual error and qualified timing observations required')
        archive(prediction,source)
        strata,dimension_coverage=grouped(score_rows)
        timing_strata,_=grouped(timing_rows)
        models.append(dict(fit_item_id=fid,name=fit['name'],task=fit['task'],comparison_group=group,
            accuracy_source=source,timing_source=default['benchmark_authority'],GPU=gpu,
            timing_units=default['timing_units'],qualified_timing=True,
            accuracy=summarize(score_rows),groups=strata,dimension_coverage=dimension_coverage,
            timing=summarize(timing_rows),timing_groups=timing_strata,case_median_timing=summarize(case_times),
            forcing_frequency_from_output_cadence_inferred=False,
            unavailable_metrics_policy='Only measured scalar metrics are retained;missing forcing metadata,spatial diagnostics,drift and unsupported fields are not invented. Original raw scalar reports remain authoritative for flags/setup/memory details.'))
    require(len(models)==203 and len({m['fit_item_id'] for m in models})==203
            and sum(m['qualified_timing'] for m in models)==195, 'Exact203accuracy/195timed fit coverage required')
    save(output,dict(status='all203qualified_fit_scalar_populations_summarized',created_unix=time.time(),
        producer=binding(Path(__file__)),source_coverage=binding(coverage_path),sources=list(sources.values()),models=models,
        additional_execution_modes=extra_modes,prediction_candidates=list(predictions.values()),
        measured_accuracy_fits=203,qualified_timing_fits=195,pending_fit_ids=coverage['pending_accuracy_fit_ids'],
        distribution_fields=['best','worst','mean','median','p95','samples','best_case','worst_case'],
        p95_definition='Linear interpolation at0.95*(n-1) in sorted observations',
        statistical_scope='Per-model observed request/output error distributions. Timings pool only qualified primary samples;case-median distributions are separately labeled. Unique identifiers are not a claim of independent physical scenes. Do not pool incompatible component/full-state boundaries.',
        flags_and_resource_scope='Original bound scalar JSON retains denominator-floor flags,setup/cold costs,memory peaks,hardware and failures. Missing whole-population spatial diagnostics remain unavailable.',
        prediction_arrays_saved=False,new_model_forwards=0,CPU_neural_forwards=0,model_dataset_or_project_copies=0,
        native_mutations=0,full_portfolio_complete=False))
    print('All203fit error distributions and195qualified GPU timing distributions retained;no inference or predictions saved.',flush=True)


if __name__=='__main__':run()
