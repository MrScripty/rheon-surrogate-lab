"""Check every150fit native scalar against original errors and raw timings."""
import base64
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require

OUTPUT = RUN/'all150-pressure-GPU-measurements-01'
REPORT = DOC/'reports/all150-pressure-GPU-measurements-qualification-01.json'


def same(a,b):
    return struct.pack('!d',a) == struct.pack('!d',b)


def run():
    require(not REPORT.exists(),'Preserve original independent all150packet qualification')
    prep = read(OUTPUT/'preparation.json')
    require(prep['status'] == 'all150_pressure_GPU_measurements_prepared_not_imported'
            and prep['subjects'] == 150 and prep['observations'] == 96000,'Exact150pressure population required')
    for spec in [prep['source_manifest'],prep['original_fit_source_coverage'],prep['native_owner'],prep['producer'],*prep['evidence']]:
        require(binding(ROOT/spec['path']) == spec,'Bound scientific/preparation/native source changed')
    old = read(DOC/'reports/pressure-portfolio-GPU-timing-01.json')
    new = read(DOC/'reports/original-pressure-GPU-timing-batch-01.json')
    coverage = read(ROOT/prep['original_fit_source_coverage']['path'])
    fits = {row['fit_item_id']:row for row in coverage['rows'] if row['default_benchmark'] and row['default_benchmark']['group'] == 'pressure_request'}
    new_by_fit = {row['fit_item_id']:row for row in new['rows']}
    old_by_fit = {row['fit_item_id']:row['default_benchmark']['identifier'] for row in fits.values()
                  if row['fit_item_id'] not in new_by_fit}
    require(len(fits) == 150 and len(old_by_fit) == 127 and len(new_by_fit) == 23,'Exact150fit identity partition required')
    raw_new = defaultdict(dict)
    raw_new_bindings = defaultdict(dict)
    for spec in new['raw_results']:
        require(binding(ROOT/spec['path']) == spec,'New original raw timing result changed')
        data = read(ROOT/spec['path'])
        raw_new[data['fit_item_id']][data['phase']] = {row['case_index']:row for row in data['rows']}
        raw_new_bindings[data['fit_item_id']][data['phase']] = spec
    prior_packet_path = ROOT/prep['evidence'][2]['path']
    prior_packet = read(prior_packet_path)
    prior_context = json.loads(base64.b64decode(next(s['data_base64'] for s in prior_packet['sources'] if s['id'] == 'context')))
    sys.path.insert(0,str(RUN/'sources/Tuldok-composition-source-artifacts'))
    from retained_measurements import MeasurementTable,MeasurementComparison
    from workbench import WorkbenchError
    tables,seen,values,raw_checks = [],set(),defaultdict(list),0
    scalar_count,context_definition = 0,None
    original_capture_bindings = set()
    for part in prep['packets']:
        path = OUTPUT/part['path']
        require(binding(path)['sha256'] == part['sha256'] and path.stat().st_size == part['bytes'],'Native packet bytes changed')
        packet = read(path)
        table = MeasurementTable(path.read_bytes())
        embedded = {s['id']:json.loads(base64.b64decode(s['data_base64'])) for s in packet['sources']}
        context,evidence = embedded['context'],embedded['results']['models']
        require(context['metrics'] == prior_context['metrics'] and context['cases'] == prior_context['cases']
                and packet['cases'] == prior_packet['cases'] and packet['metrics'] == prior_packet['metrics']
                and packet['scope']['id'] == 'old-pressure-all150-complete-GPU-call-128-v1',
                'Exact original cases and metric meanings required; new roster scope must be explicit')
        if context_definition is None:
            context_definition = context
        else:
            require(context == context_definition,'Two unified partitions must have exactly equal common context')
        require(not any('raw_wall_nanoseconds' in json.dumps(evidence[name]) for name in evidence),
                'Raw process bodies must be retained by binding, not recopied into chart evidence')
        require({s['id'] for s in packet['subjects']} == evidence.keys(),'Subject source pointers differ')
        expected_by_subject = {}
        for subject in packet['subjects']:
            name,dims = subject['id'],subject['dimensions']
            fid = dims['fit_item_id']
            require(fid in fits and fid not in seen and prep['fit_to_subject'][fid] == name
                    and dims['inference_device'] == 'cuda' and dims['host_logical_cpu'] == 4 and dims['host_threads'] == 1
                    and dims['gpu_model'] == old['settings']['gpu_model'] and not dims['tf32'] and not dims['autocast']
                    and dims['torch_version'] == old['settings']['torch_version'] and dims['cuda_version'] == old['settings']['cuda_version'],
                    'Native subject must join one original fit and exact device/runtime')
            seen.add(fid)
            body = evidence[name]
            require(subject['reference'] == dict(source_id='results',pointer='/models/'+name)
                    and body['fit_item_id'] == fid and body['weights_sha256'] == dims['weights_sha256'],
                    'Subject original fit/checkpoint/source binding differs')
            if fid in new_by_fit:
                original = new_by_fit[fid]
                accuracy = read(ROOT/original['accuracy']['path'])
                require(binding(ROOT/original['accuracy']['path']) == original['accuracy']
                        and dims['benchmark_campaign'] == 'additional23' and dims['paired_calibration_available']
                        and not dims['host_metadata_overlap_in_one_pass'] and body['accuracy'] == original['accuracy']
                        and body['paired_observer_overhead_median_ms'] == original['paired_observer_overhead_median_ms']
                        and dims['weights_sha256'] == accuracy['weights_sha256'],'New pressure fit provenance differs')
                original_rows = accuracy['rows']
                samples = [raw_new[fid][0][i]['raw_wall_nanoseconds']+raw_new[fid][1][i]['raw_wall_nanoseconds'] for i in range(128)]
                require(body['original_raw_processes'] == [raw_new_bindings[fid][0],raw_new_bindings[fid][1]],
                        'New fit must retain its own two original process identities')
                require(body['benchmark_reduction'] == binding(DOC/'reports/original-pressure-GPU-timing-batch-01.json'),
                        'New fit timing authority differs')
            else:
                original = old['models'][old_by_fit[fid]]
                require(name == old_by_fit[fid] and dims['benchmark_campaign'] == 'original127' and not dims['paired_calibration_available']
                        and dims['host_metadata_overlap_in_one_pass'] == original['host_metadata_overlap_in_one_pass']
                        and dims['weights_sha256'] == original['weights_sha256'],'Historical pressure fit provenance differs')
                original_rows = original['rows']
                samples = [row['raw_wall_nanoseconds'] for row in original_rows]
                require(body['original_raw_processes'] == original['raw_blocks'],
                        'Historical fit must retain its own two original process identities')
                require(body['benchmark_reduction'] == binding(DOC/'reports/pressure-portfolio-GPU-timing-01.json'),
                        'Original127fit timing authority differs')
            for spec in body['original_raw_processes']:
                if 'path' in spec:
                    file_spec = {key:spec[key] for key in ('path','sha256')}
                else:
                    file_spec = dict(path=spec['result'],sha256=spec['sha256'])
                require(binding(ROOT/file_spec['path']) == file_spec,'Retained original process bytes changed')
                original_capture_bindings.add((file_spec['path'],file_spec['sha256']))
            require(len(body['rows']) == 128,'Every original128request required')
            expected = []
            for index,(row,source_row,raw_samples) in enumerate(zip(body['rows'],original_rows,samples,strict=True)):
                require(len(raw_samples) == 20 and all(type(v) is int and v > 0 for v in raw_samples)
                        and row['case_id'] == f'case-{index:03d}' and row['output_sha256'] == source_row['output_sha256'],
                        'Original primary samples,case or field hash differs')
                case = context['cases'][index]
                require(source_row['input_sha256'] == case['original_case']['input_sha256'],'Exact shared pressure inputs required')
                accuracy_values = source_row['metrics'] if fid in new_by_fit else source_row['accuracy']
                ordered = sorted(raw_samples)
                rank = .95*19
                low,high = math.floor(rank),math.ceil(rank)
                calculated = dict(equation_relative_l2=accuracy_values['common_equation_relative_l2'],
                    pressure_mean_free_relative_l2=accuracy_values['pressure_mean_free_relative_l2'],
                    pressure_call_ms=statistics.median(raw_samples)/1e6,
                    pressure_call_p95_ms=(ordered[low]+(ordered[high]-ordered[low])*(rank-low))/1e6,
                    pressure_call_maximum_ms=ordered[-1]/1e6)
                require(set(calculated) == set(row['values']) and all(same(row['values'][key],value) for key,value in calculated.items()),
                        'Chart evidence scalar differs from independent original error/timing calculation')
                expected.append(calculated)
                raw_checks += 20
            expected_by_subject[name] = expected
        ids = set()
        for observation in packet['observations']:
            name,case_id,metric = [observation[key] for key in ('subject_id','case_id','metric_id')]
            index = int(case_id.removeprefix('case-'))
            key = (name,index,metric)
            require(key not in ids,'Repeated scientific observation')
            ids.add(key)
            value = expected_by_subject[name][index][metric]
            require(same(observation['value'],value) and observation['state'] == 'measured' and observation['reason'] is None
                    and observation['dimensions'] == {} and observation['reference'] ==
                    dict(source_id='results',pointer=f'/models/{name}/rows/{index}/values/{metric}'),
                    'Native scientific scalar,state or provenance pointer differs')
            values[(name,metric)].append((packet['cases'][index]['dimensions'],value))
            scalar_count += 1
        require(len(ids) == len(packet['subjects'])*128*5,'Every subject/case/metric required exactly once')
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    require(seen == fits.keys() and scalar_count == 96000 and raw_checks == 384000
            and len(original_capture_bindings) == 300,'Complete150fit/128case/20samples/two-process original coverage required')
    comparison = MeasurementComparison(tables)
    case_dimensions = [case['dimensions'] for case in prior_packet['cases']]
    filters = [{}]+[dict(cells_per_metre=d) for d in sorted({row['cells_per_metre'] for row in case_dimensions})]
    filters += [dict(cells_per_metre=d,grid_x=x,grid_y=y,grid_z=z) for d,x,y,z in sorted({
        (row['cells_per_metre'],row['grid_x'],row['grid_y'],row['grid_z']) for row in case_dimensions})]
    queries = []
    for selection in filters:
        for metric in ('equation_relative_l2','pressure_mean_free_relative_l2','pressure_call_ms','pressure_call_p95_ms','pressure_call_maximum_ms'):
            result = comparison.read(metric,filters=selection,limit=1)['query']
            require(len(result['summary']) == 150,'All150subjects must remain selectable together')
            for summary in result['summary']:
                selected = [value for dims,value in values[(summary['subject_id'],metric)] if all(dims[k] == v for k,v in selection.items())]
                require(len(selected) == summary['states']['measured'] and same(max(selected),summary['maximum']['value']),
                        'Native density/grid denominator or maximum differs')
            queries.append(dict(metric=metric,filters=selection,subjects=150,selected_cases_per_subject=len(selected)))
    rejected = False
    try:
        MeasurementComparison([tables[0],(dict(origin_catalog_id='a'*32,packet_sha256=binding(prior_packet_path)['sha256']),
                                          MeasurementTable(prior_packet_path.read_bytes()))])
    except WorkbenchError as error:
        require(error.code == 'incompatible' and error.status == 409,'Historical scope should reject for its differing exact definition')
        rejected = True
    require(rejected,'Historical127roster scope must not silently join the new150scope')
    require('torch' not in sys.modules,'Scalar packet qualification must not import the neural runtime')
    save(REPORT,dict(status='all150_pressure_GPU_measurements_independently_qualified_not_imported',created_unix=time.time(),
        fits=150,original_named_fits=127,additional_original_training_fits=23,cases=128,metrics=5,exact_scientific_observations=scalar_count,
        original_primary_samples_checked=raw_checks,original_fresh_process_bindings_checked=len(original_capture_bindings),
        native_filter_queries=len(queries),queries=queries,incompatible_historical_scope_rejection=True,
        all_original_cases_metric_definitions_and_scalar_bits_exact=True,raw_process_bodies_copied=0,
        model_dataset_or_prediction_archive_copies=0,new_model_forwards=0,native_mutations=0,native_UI_qualified=False,
        preparation=binding(OUTPUT/'preparation.json'),source_manifest=prep['source_manifest'],native_owner=prep['native_owner'],
        packets=[binding(OUTPUT/p['path']) for p in prep['packets']],producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('All150pressure fits/96000native scalars independently qualified; original384000samples/300processes checked.',flush=True)


if __name__ == '__main__':
    run()
