"""Package the independently reduced pressure portfolio in two native tables.

Two bounded partitions retain exact accuracy, all20timing samples/native case,
case/domain/density metadata and original result bindings. No model or live
project is opened. Run only after the original timing campaign is terminal.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from prepare_portfolio_measurements import source

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1/reports'
REPORT = DOC/'pressure-portfolio-GPU-timing-01.json'
OUTPUT = RUN/'native-pressure-portfolio-GPU-timing-01'
SOURCE = RUN/'sources/Tuldok-composition-source-artifacts'
DEFINITIONS = (
    ('equation_relative_l2','Common pressure equation residual (relative L2)','1','accuracy/common_equation_relative_l2'),
    ('pressure_mean_free_relative_l2','Mean-free pressure field error (relative L2)','1','accuracy/pressure_mean_free_relative_l2'),
    ('pressure_call_ms','Complete GPU pressure-call median','ms/request','median_ms'),
    ('pressure_call_p95_ms','Complete GPU pressure-call p95','ms/request','p95_ms'),
    ('pressure_call_maximum_ms','Complete GPU pressure-call maximum','ms/request','maximum_ms'))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def run(complete_existing_packets=False):
    if complete_existing_packets:
        require(OUTPUT.is_dir() and not OUTPUT.is_symlink()
                and {p.name for p in OUTPUT.iterdir()} == {'part-01.json','part-02.json'},
                'Only the two known constructor-complete, unpublished packets may be reconciled')
    else:
        require(not OUTPUT.exists(), 'Fresh task-owned packet preparation required')
    report = read(REPORT)
    require(report['status'] == 'all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified'
            and report['timed_pressure_calls'] == 325120 and len(report['models']) == 127
            and report['all_native_input_and_output_hashes_exact'] and report['every_common_accuracy_row_joined'],
            'Complete independently reduced original benchmark required')
    for spec in report['evidence']:
        require(sha(ROOT/spec['path']) == spec['sha256'], 'Independent timing evidence changed')
    sys.path.insert(0,str(SOURCE))
    from retained_measurements import MeasurementTable, MeasurementComparison, digest, encode
    from projects import manifest as validate_manifest
    template_path = RUN/'native-measurements-packet-05/measurements.json'
    require(sha(template_path) == 'c78bbc342cad9f3d9af5116f2c272e85f1d01344f7af0f33f4dce000de828400', 'Qualified native cases changed')
    template = read(template_path)
    original_context = json.loads(base64.b64decode(next(s['data_base64'] for s in template['sources'] if s['id'] == 'context')))
    original_metric = {m['id']:m for m in original_context['metrics']}
    scope = copy.deepcopy(original_context['scope'])
    scope.update(id='old-pressure-portfolio-complete-GPU-call-128-v1',
        label='127 pressure models · complete GPU calls · 128 consumed requests',
        task='Pressure proposals from native CPU fields to independent CPU outputs; no simulation advancement.',
        timing_scope=report['timing_scope'],timing_statistic='20 warm complete-call samples per case in two distinct process passes',
        hardware=dict(settings=report['settings'],GPU_identity=report['GPU_identity']),
        resources='Sampled GPU competitors only; not proof of continuous host/GPU exclusivity or fixed thermal state.')
    definitions = []
    for identity,label,units,pointer in DEFINITIONS:
        if pointer.startswith('accuracy/'):
            value = copy.deepcopy(original_metric[identity])
            value.update(label=label,source_kind='Exact retained common score joined by original native input/output hashes')
        else:
            value = dict(id=identity,label=label,units=units,
                description=report['timing_scope'],source_kind='Derived from two original ten-sample wall-nanosecond arrays',
                formula={'median_ms':'median(samples)/1e6','p95_ms':'linear percentile at rank .95*(N-1), divided by1e6',
                         'maximum_ms':'max(samples)/1e6'}[pointer])
        definitions.append(value)
    context = dict(scope=scope,metrics=definitions,cases=original_context['cases'],
        independent_reduction=dict(path=str(REPORT.relative_to(ROOT)),sha256=sha(REPORT)),
        excluded_contended_block=report['excluded_contended_block'],
        known_host_activity_disclosure=report['known_host_activity_disclosure'],
        missing_named_checkpoint_dispositions=report['missing_named_checkpoint_dispositions'],
        full205_fit_524_state_portfolio_complete=False,
        interpretation='Two error metrics and three wall-time statistics. Minimum timings and additional absolute errors remain in the full reduction. No K division, full-step speed or independent-scene claim.')
    dimensions = copy.deepcopy(template['dimensions']) + [
        dict(id='execution_variant',label='Pressure execution variant',type='string',units=None,scope='subject'),
        dict(id='host_logical_cpu',label='Pinned host logical CPU',type='integer',units=None,scope='subject'),
        dict(id='host_threads',label='Host threads',type='integer',units=None,scope='subject'),
        dict(id='host_metadata_overlap_in_one_pass',label='Possible host metadata overlap in one pass',type='boolean',units=None,scope='subject')]
    settings = report['settings']
    subject_dimensions = dict(inference_device='cuda',gpu_model=settings['gpu_model'],network_dtype=settings['network_dtype'],
        pressure_dtype=settings['pressure_dtype'],tf32=False,autocast=settings['autocast'],torch_version=settings['torch_version'],
        cuda_version=settings['cuda_version'],host_logical_cpu=4,host_threads=1)
    names = list(report['models'])
    require(names == [f'C{i:02}' for i in range(3,131) if i != 68], 'Expected all127available named checkpoints')
    packets, tables = [], []
    if not complete_existing_packets:
        OUTPUT.mkdir()
    for start in (0,64):
        selected = names[start:start+64]
        evidence = dict(original_reduction=dict(path=str(REPORT.relative_to(ROOT)),sha256=sha(REPORT)),models={})
        subjects = []
        for name in selected:
            original = report['models'][name]
            evidence['models'][name] = dict(weights_sha256=original['weights_sha256'],execution_variant=original['execution_variant'],
                raw_blocks=original['raw_blocks'],rows=[{k:r[k] for k in
                    ('case_id','output_sha256','raw_wall_nanoseconds','accuracy','minimum_ms','median_ms','p95_ms','maximum_ms')}
                    for r in original['rows']])
            subjects.append(dict(id=name,label=name+' · GPU pressure call',
                reference=dict(source_id='reduction',pointer='/models/'+name),
                dimensions=dict(subject_dimensions,weights_sha256=original['weights_sha256'],execution_variant=original['execution_variant'],
                    host_metadata_overlap_in_one_pass=original['host_metadata_overlap_in_one_pass'])))
        metrics = [dict(id=d['id'],label=d['label'],units=d['units'],definition=dict(
            source_id='context',pointer='/metrics/'+str(i),sha256=digest(d))) for i,d in enumerate(definitions)]
        observations = []
        for name in selected:
            for index, row in enumerate(report['models'][name]['rows']):
                require(row['case_id'] == template['cases'][index]['id']
                        and row['dimensions'] == template['cases'][index]['dimensions'], 'Exact original case dimensions differ')
                for metric,_,_,pointer in DEFINITIONS:
                    value = row
                    for part in pointer.split('/'):
                        value = value[part]
                    observations.append(dict(id='o-'+str(len(observations)),subject_id=name,case_id=row['case_id'],
                        metric_id=metric,value=value,state='measured',reason=None,dimensions={},
                        reference=dict(source_id='reduction',pointer='/models/'+name+'/rows/'+str(index)+'/'+pointer)))
        packet = dict(format=template['format'],schema_version=1,title=scope['label']+' · part '+str(len(packets)+1),
            scope={k:scope[k] for k in ('id','label','population')}|dict(definition=dict(source_id='context',pointer='/scope',sha256=digest(scope))),
            dimensions=dimensions,subjects=subjects,cases=copy.deepcopy(template['cases']),metrics=metrics,observations=observations,
            sources=[source('context','Exact case, metric, device and complete pressure-call scope',encode(context)),
                     source('reduction','Exact common errors, all20wall samples per case and original process bindings',encode(evidence))])
        raw = encode(packet)
        table = MeasurementTable(raw)
        require(table.metadata()['observation_count'] == len(selected)*128*5, 'Native population differs')
        path = OUTPUT/f'part-{len(packets)+1:02}.json'
        if complete_existing_packets:
            require(path.read_bytes() == raw, 'Known unpublished packet differs from exact regenerated values')
        else:
            path.write_bytes(raw)
        packets.append(dict(path=path.name,sha256=sha(path),bytes=len(raw),subjects=len(selected),cases=128,measured=len(observations)))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    comparison = MeasurementComparison(tables)
    # A bounded real cross-packet query verifies both partitions join natively.
    result = comparison.read('pressure_call_ms',filters=dict(cells_per_metre=8),limit=1)['query']
    require(len(result['summary']) == 127 and all(r['states']['measured'] == 64 for r in result['summary']),
            'Native complete comparison/density denominator differs')
    declaration = dict(schema_version=1,id=hashlib.sha256(encode(context)).hexdigest()[:32],name=scope['label'],
        data=[dict(id='GPU-pressure-timing-'+str(i),label=scope['label']+' · part '+str(i),
            path=str((OUTPUT/p['path']).relative_to(ROOT)),sha256=p['sha256']) for i,p in enumerate(packets,1)],
        recipes=[],evidence=[],requirements=[])
    validate_manifest(declaration)
    (OUTPUT/'source-project.json').write_bytes(encode(declaration))
    preparation = dict(status='complete_pressure_portfolio_GPU_timing_packets_prepared_not_imported',created_unix=time.time(),
        independent_report_sha256=sha(REPORT),source_manifest_sha256=sha(OUTPUT/'source-project.json'),packets=packets,
        subjects=127,cases=128,metrics=5,observations=sum(p['measured'] for p in packets),original_timed_calls=325120,
        retained_all20timing_samples_per_case=True,full_simulation_step_timing=False,new_model_forwards=0,
        native_project_mutated=False,native_UI_qualified=False,independent_packet_qualification_pending=True,
        existing_unpublished_packets_verified_and_reused=complete_existing_packets,
        producer_sha256=sha(Path(__file__)))
    (OUTPUT/'preparation.json').write_bytes(encode(preparation))
    print(preparation['status'],sum(p['bytes'] for p in packets),'bytes',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete-existing-packets',action='store_true',
        help='Reconcile the known local comparison-method failure after verifying both unpublished packet byte strings exactly.')
    run(parser.parse_args().complete_existing_packets)
