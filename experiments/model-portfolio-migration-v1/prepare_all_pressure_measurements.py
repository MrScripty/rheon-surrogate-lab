"""One150fit native comparison, derived without copying raw captures or weights.

The historical127fit tables remain unchanged. These new two partitions share a
new exact scope; original error definitions,cases and primary timing math stay
unchanged. All process bodies remain at their original bound paths.
"""
import base64
import copy
import math
from pathlib import Path
import statistics
import sys
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from prepare_portfolio_measurements import source

OUTPUT = RUN/'all150-pressure-GPU-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
EXTRA_IDS = dict(zip([
    '047049129e1e2833','0c1daefb000fee1c','135202a799d78633','136881e20046e15a','312d4f0dbc44117c',
    '5b3b1aef2efcee62','6d1558cced039d83','733a339f563172ab','803a07c660a70a93','80e3e9ce8eb8ab94',
    '8992e19d389ea016','9fa632f2b8082525','a5328ae3e2b2180f','a928b38bd18e1c6a','acb450db5fec60b9',
    'becf4ad5b5526ae4','befe8d913f41d1c8','c675a37816966f2b','c69f29dfba8954b0','c8638134ae71a368',
    'd1c66c9301a5b890','fb8ce15096322ec9','fc820329dacc4d2c'],[
    'coarse-control','teacher-coarse','multigrid-smoke','band-lowband-smoke','early-pressure-smoke',
    'conditioning-resident','sequence-pipeline','residual-smoke','direction-smoke','v2-small16',
    'c53-u','conditioning-prepared','stability-smoke','band-control','conditioning-baseline',
    'conditioning-batch4','pressure-recovery','workspace-c675a378','conditioning-amplitude','multigrid-cohort16',
    'C69-Adam-replay','prospective-fit','workspace-fc820329'],strict=True))
METRICS = ('equation_relative_l2','pressure_mean_free_relative_l2','pressure_call_ms','pressure_call_p95_ms','pressure_call_maximum_ms')


def time_values(samples):
    require(len(samples) == 20 and all(type(v) is int and v > 0 for v in samples),'Exactly20positive original primary samples required')
    values = sorted(samples)
    rank = .95*(len(values)-1)
    lo,hi = math.floor(rank),math.ceil(rank)
    return dict(pressure_call_ms=statistics.median(samples)/1e6,
        pressure_call_p95_ms=(values[lo]+(values[hi]-values[lo])*(rank-lo))/1e6,
        pressure_call_maximum_ms=max(samples)/1e6)


def run():
    require(not OUTPUT.exists(),'Fresh unified pressure chart preparation required')
    old_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    new_path = DOC/'reports/original-pressure-GPU-timing-batch-01.json'
    coverage_path = DOC/'reports/common-fit-benchmark-coverage-03.json'
    old,new,coverage = [read(path) for path in (old_path,new_path,coverage_path)]
    require(old['named_models'] == 127 and new['status'] == 'original_pressure_GPU_timing_batch_independently_reduced'
            and new['fits'] == 23 and coverage['pressure_request_fits'] == 150,'Complete150pressure default-fit results required')
    previous_prep = read(RUN/'native-pressure-portfolio-GPU-timing-01/preparation.json')
    original_packet_path = RUN/'native-pressure-portfolio-GPU-timing-01'/previous_prep['packets'][0]['path']
    require(binding(original_packet_path)['sha256'] == previous_prep['packets'][0]['sha256'],'Historical native template changed')
    template = read(original_packet_path)
    original_context = __import__('json').loads(base64.b64decode(next(s['data_base64'] for s in template['sources'] if s['id'] == 'context')))
    old_coverage = read(DOC/'reports/common-fit-benchmark-coverage-01.json')
    old_fit = {row['default_benchmark']['identifier']:row['fit_item_id'] for row in old_coverage['rows']
               if row['default_benchmark'] and row['default_benchmark']['group'] == 'pressure_request'}
    new_raw = {}
    for spec in new['raw_results']:
        require(binding(ROOT/spec['path']) == spec,'New original raw process binding changed')
        raw = read(ROOT/spec['path'])
        new_raw[(raw['fit_item_id'],raw['phase'])] = (spec,{row['case_index']:row for row in raw['rows']})
    subjects,evidence = [],{}
    settings = old['settings']
    base_dimensions = dict(inference_device='cuda',gpu_model=settings['gpu_model'],network_dtype=settings['network_dtype'],
        pressure_dtype=settings['pressure_dtype'],tf32=False,autocast=False,torch_version=settings['torch_version'],
        cuda_version=settings['cuda_version'],host_logical_cpu=4,host_threads=1)
    for name,model in old['models'].items():
        fid = old_fit[name]
        body = dict(fit_item_id=fid,name=name,weights_sha256=model['weights_sha256'],
            benchmark_reduction=binding(old_path),original_raw_processes=model['raw_blocks'],rows=[])
        for row in model['rows']:
            values = time_values(row['raw_wall_nanoseconds'])
            require(values['pressure_call_ms'] == row['median_ms'] and values['pressure_call_p95_ms'] == row['p95_ms']
                    and values['pressure_call_maximum_ms'] == row['maximum_ms'],'Historical timing statistics differ')
            values.update(equation_relative_l2=row['accuracy']['common_equation_relative_l2'],
                          pressure_mean_free_relative_l2=row['accuracy']['pressure_mean_free_relative_l2'])
            body['rows'].append(dict(case_id=row['case_id'],output_sha256=row['output_sha256'],values=values))
        evidence[name] = body
        subjects.append(dict(id=name,label=name+' · GPU pressure request',reference=dict(source_id='results',pointer='/models/'+name),
            dimensions=dict(base_dimensions,weights_sha256=model['weights_sha256'],execution_variant=model['execution_variant'],
                host_metadata_overlap_in_one_pass=model['host_metadata_overlap_in_one_pass'],fit_item_id=fid,
                benchmark_campaign='original127',paired_calibration_available=False)))
    for model in new['rows']:
        fid = model['fit_item_id']
        name = EXTRA_IDS[fid.removeprefix('fit-')]
        require(name not in evidence,'Readable chart identity must be unique')
        accuracy = read(ROOT/model['accuracy']['path'])
        raw0,rows0 = new_raw[(fid,0)]
        raw1,rows1 = new_raw[(fid,1)]
        for spec in (raw0,raw1):
            original = read(ROOT/spec['path'])
            require(original['GPU'] == settings['gpu_model'] and original['torch_version'] == settings['torch_version']
                    and original['CUDA_version'] == settings['cuda_version'],'Actual GPU/runtime differs between campaigns')
        body = dict(fit_item_id=fid,name=model['name'],weights_sha256=accuracy['weights_sha256'],benchmark_reduction=binding(new_path),
            accuracy=model['accuracy'],original_raw_processes=[raw0,raw1],paired_observer_overhead_median_ms=model['paired_observer_overhead_median_ms'],rows=[])
        for index,case in enumerate(accuracy['rows']):
            require(rows0[index]['output_sha256'] == rows1[index]['output_sha256'] == case['output_sha256'],
                    'Cross-phase original pressure field differs')
            values = time_values(rows0[index]['raw_wall_nanoseconds']+rows1[index]['raw_wall_nanoseconds'])
            require(values['pressure_call_ms'] == model['case_summaries'][index]['pooled_median_ms'],'New timing reduction differs')
            values.update(equation_relative_l2=case['metrics']['common_equation_relative_l2'],
                          pressure_mean_free_relative_l2=case['metrics']['pressure_mean_free_relative_l2'])
            body['rows'].append(dict(case_id=f'case-{index:03d}',output_sha256=case['output_sha256'],values=values))
        evidence[name] = body
        subjects.append(dict(id=name,label=name+' · GPU pressure request',reference=dict(source_id='results',pointer='/models/'+name),
            dimensions=dict(base_dimensions,weights_sha256=accuracy['weights_sha256'],execution_variant='original_train_default_CUDA_pressure',
                host_metadata_overlap_in_one_pass=False,fit_item_id=fid,benchmark_campaign='additional23',paired_calibration_available=True)))
    require(len(subjects) == len(evidence) == len({s['dimensions']['fit_item_id'] for s in subjects}) == 150,
            'Exactly150distinct pressure fits required')
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,MeasurementComparison,encode,digest
    from projects import manifest as validate_manifest
    scope = copy.deepcopy(original_context['scope'])
    scope.update(id='old-pressure-all150-complete-GPU-call-128-v1',label='150 pressure fits · complete GPU requests · 128 consumed cases',
        timing_statistic='20primary complete-call wall samples/case,10in each fresh process phase; hooks removed.',
        hardware=dict(settings=settings,GPU_identity=old['GPU_identity']),
        resources='GPU compute PID snapshots only; not continuous exclusivity or fixed thermals. Original127host-metadata overlap is retained per fit. Additional23ran with live UI idle, no heavy native publication/profiling.',
        comparison='Two campaigns share exact inputs/references,metric definitions and full nativeCPU/CUDA/ownedCPU boundary. Not simultaneous paired optimization trials. LegacyC03/C04retain CPU feature/reconstruction with CUDA neural execution.')
    context = dict(scope=scope,metrics=copy.deepcopy(original_context['metrics']),cases=copy.deepcopy(original_context['cases']),
        reductions=[binding(old_path),binding(new_path)],coverage=binding(coverage_path),
        historical_packets_preserved=previous_prep['packets'],
        raw_capture_retention='All original process bodies remain once at their bound original paths; chart packets retain scalar derivatives and original bindings only. No pressure-field archive,weight,dataset or project copy.',
        interpretation='GPU milliseconds per pressure request; no pressure-pass/K division and no full simulation advancement. Plot maximums are maxima over per-case statistics. Fitting/consumed population,not independent holdout or long drift.',
        paired_observer_calibration='Available separately in original additional23captures and reduction. Primary values are neither instrumented nor overhead-subtracted.',
        full_portfolio_complete=False)
    dimensions = copy.deepcopy(template['dimensions'])+[
        dict(id='fit_item_id',label='Original fit identity',type='string',units=None,scope='subject'),
        dict(id='benchmark_campaign',label='Benchmark campaign',type='string',units=None,scope='subject'),
        dict(id='paired_calibration_available',label='Separate observer calibration retained',type='boolean',units=None,scope='subject')]
    metrics = [dict(id=item['id'],label=item['label'],units=item['units'],definition=dict(source_id='context',pointer=f'/metrics/{i}',sha256=digest(item)))
               for i,item in enumerate(context['metrics'])]
    require([metric['id'] for metric in metrics] == list(METRICS),'Exact five original pressure metric meanings required')
    OUTPUT.mkdir()
    packets,tables = [],[]
    for start in (0,75):
        selected = subjects[start:start+75]
        selected_evidence = dict(models={s['id']:evidence[s['id']] for s in selected})
        observations = []
        for subject in selected:
            name = subject['id']
            for index,row in enumerate(evidence[name]['rows']):
                require(row['case_id'] == template['cases'][index]['id'],'Exact original case order required')
                for metric in METRICS:
                    observations.append(dict(id=f'o-{len(observations)}',subject_id=name,case_id=row['case_id'],metric_id=metric,
                        value=row['values'][metric],state='measured',reason=None,dimensions={},
                        reference=dict(source_id='results',pointer=f'/models/{name}/rows/{index}/values/{metric}')))
        packet = dict(format=template['format'],schema_version=1,title=scope['label']+f' · part {len(packets)+1}',
            scope={k:scope[k] for k in ('id','label','population')}|dict(definition=dict(source_id='context',pointer='/scope',sha256=digest(scope))),
            dimensions=dimensions,subjects=selected,cases=copy.deepcopy(template['cases']),metrics=metrics,observations=observations,
            sources=[source('context','Exact common pressure scope,cases and metric definitions',encode(context)),
                     source('results','Scalar derivatives and original raw process bindings; no raw process copy',encode(selected_evidence))])
        raw = encode(packet)
        table = MeasurementTable(raw)
        path = OUTPUT/f'part-{len(packets)+1:02}.json'
        path.write_bytes(raw)
        packets.append(dict(path=path.name,sha256=binding(path)['sha256'],bytes=len(raw),subjects=len(selected),cases=128,observations=len(observations)))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    comparison = MeasurementComparison(tables)
    query = comparison.read('pressure_call_ms',filters=dict(cells_per_metre=8),limit=1)['query']
    require(len(query['summary']) == 150 and all(row['states']['measured'] == 64 for row in query['summary']),
            'Native150subject comparison/density denominator differs')
    declaration = dict(schema_version=1,id=digest(context)[:32],name=scope['label'],
        data=[dict(id=f'all150-pressure-{i}',label=scope['label']+f' · part {i}',path=str((OUTPUT/p['path']).relative_to(ROOT)),sha256=p['sha256'])
              for i,p in enumerate(packets,1)],recipes=[],evidence=[],requirements=[])
    validate_manifest(declaration)
    require('torch' not in sys.modules,'Scalar chart preparation must not import the neural runtime')
    (OUTPUT/'source-project.json').write_bytes(encode(declaration))
    save(OUTPUT/'preparation.json',dict(status='all150_pressure_GPU_measurements_prepared_not_imported',created_unix=time.time(),
        packets=packets,source_manifest=binding(OUTPUT/'source-project.json'),subjects=150,cases=128,metrics=5,observations=96000,
        fit_to_subject={s['dimensions']['fit_item_id']:s['id'] for s in subjects},original_fit_source_coverage=binding(coverage_path),
        evidence=[binding(old_path),binding(new_path),binding(original_packet_path)],native_owner=binding(NATIVE/'retained_measurements.py'),
        raw_process_bodies_copied=0,prediction_archives_copied=0,model_or_dataset_copies=0,native_mutations=0,new_model_forwards=0,
        independent_scalar_and_filter_qualification_pending=True,native_UI_qualified=False,producer=binding(Path(__file__))))
    print('Prepared150pressure subjects/96000scalars in two compatible native tables; no raw process or model/data copy.',flush=True)


if __name__ == '__main__':
    run()
