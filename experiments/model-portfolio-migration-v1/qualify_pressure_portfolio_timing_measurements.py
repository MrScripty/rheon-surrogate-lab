"""Verify every native pressure scalar, original common score and raw timing.

No neural runtime, project mutation or SQL service is used. Native query
denominators/maxima are checked separately for both densities and both packets.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1/reports'
OUTPUT = RUN/'native-pressure-portfolio-GPU-timing-01'
REPORT = DOC/'pressure-portfolio-GPU-timing-01.json'
QUALIFIED = DOC/'pressure-portfolio-GPU-timing-native-preparation-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def same(a,b):
    return struct.pack('!d',a) == struct.pack('!d',b)


def run():
    require(not QUALIFIED.exists(), 'Preserve original independent packet qualification')
    prep = read(OUTPUT/'preparation.json')
    reduced = read(REPORT)
    require(prep['independent_report_sha256'] == sha(REPORT)
            and prep['source_manifest_sha256'] == sha(OUTPUT/'source-project.json')
            and reduced['status'] == 'all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified',
            'Independent reduction or native source manifest differs')
    source = RUN/'sources/Tuldok-composition-source-artifacts'
    sys.path.insert(0,str(source))
    from retained_measurements import MeasurementTable, MeasurementComparison
    common = read(DOC/'pressure-common-score-01.json')
    require(sha(ROOT/common['details']['path']) == common['details']['sha256'], 'Original common score changed')
    scored = read(ROOT/common['details']['path'])['records']
    wanted_metrics = {'equation_relative_l2':'accuracy/common_equation_relative_l2',
        'pressure_mean_free_relative_l2':'accuracy/pressure_mean_free_relative_l2',
        'pressure_call_ms':'median_ms','pressure_call_p95_ms':'p95_ms','pressure_call_maximum_ms':'maximum_ms'}
    tables, queries, seen, observations = [], [], set(), 0
    for part in prep['packets']:
        path = OUTPUT/part['path']
        require(sha(path) == part['sha256'] and path.stat().st_size == part['bytes'], 'Native packet differs')
        raw = path.read_bytes()
        packet = json.loads(raw)
        table = MeasurementTable(raw)
        sources = {s['id']:s for s in packet['sources']}
        context = json.loads(base64.b64decode(sources['context']['data_base64']))
        evidence = json.loads(base64.b64decode(sources['reduction']['data_base64']))
        require(evidence['original_reduction']['sha256'] == sha(REPORT)
                and context['independent_reduction']['sha256'] == sha(REPORT)
                and set(wanted_metrics) == {m['id'] for m in packet['metrics']}
                and packet['scope']['id'] == 'old-pressure-portfolio-complete-GPU-call-128-v1', 'Wrong native scientific scope')
        require(len(packet['cases']) == 128 and len(packet['subjects']) == part['subjects'], 'Wrong native case/model coverage')
        subjects = {s['id']:s for s in packet['subjects']}
        case_indices = {c['id']:i for i,c in enumerate(packet['cases'])}
        require(not seen.intersection(subjects), 'A model was duplicated across native packets')
        seen.update(subjects)
        for name, subject in subjects.items():
            expected = reduced['models'][name]
            require(evidence['models'][name]['weights_sha256'] == subject['dimensions']['weights_sha256']
                    == expected['weights_sha256'] == common['models'][name]['weights_sha256']
                    and evidence['models'][name]['raw_blocks'] == expected['raw_blocks']
                    and subject['dimensions']['inference_device'] == 'cuda'
                    and subject['dimensions']['host_metadata_overlap_in_one_pass'] == expected['host_metadata_overlap_in_one_pass'],
                    'Checkpoint/process/device/known-host-condition attribution differs')
            for index, row in enumerate(evidence['models'][name]['rows']):
                original = expected['rows'][index]
                samples = row['raw_wall_nanoseconds']
                require(len(samples) == 20 and all(type(v) is int and v > 0 for v in samples)
                        and samples == original['raw_wall_nanoseconds']
                        and row['output_sha256'] == scored[name]['rows'][index]['output_sha256']
                        and row['accuracy'] == scored[name]['rows'][index]['metrics'], 'Original samples/common accuracy were altered')
                values = sorted(samples)
                rank = .95*(len(values)-1)
                low, high = math.floor(rank), math.ceil(rank)
                calculated = dict(minimum_ms=min(samples)/1e6,median_ms=statistics.median(samples)/1e6,
                    maximum_ms=max(samples)/1e6,p95_ms=(values[low]+(values[high]-values[low])*(rank-low))/1e6)
                require(all(same(row[k],v) and same(original[k],v) for k,v in calculated.items()),
                        'Native raw samples do not reproduce exact retained statistics')
                require(row['case_id'] == packet['cases'][index]['id']
                        and original['dimensions'] == packet['cases'][index]['dimensions'], 'Native density/domain/cadence differs')
        values = {}
        for observation in packet['observations']:
            name, case_id, metric = (observation[k] for k in ('subject_id','case_id','metric_id'))
            index = case_indices[case_id]
            suffix = wanted_metrics[metric]
            expected = reduced['models'][name]['rows'][index]
            for key in suffix.split('/'):
                expected = expected[key]
            require(same(observation['value'],expected) and observation['state'] == 'measured'
                    and observation['reason'] is None and observation['dimensions'] == {}
                    and observation['reference'] == dict(source_id='reduction',pointer='/models/'+name+'/rows/'+str(index)+'/'+suffix),
                    'Native scalar/provenance pointer differs')
            values.setdefault((name,metric),[]).append((packet['cases'][index]['dimensions'],expected))
            observations += 1
        for density in (8,16):
            for metric in wanted_metrics:
                result = table.query(metric,filters=dict(cells_per_metre=density),limit=1)
                for summary in result['summary']:
                    selected = [v for dims,v in values[(summary['subject_id'],metric)] if dims['cells_per_metre'] == density]
                    require(len(selected) == summary['states']['measured'] == 64
                            and same(summary['maximum']['value'],max(selected)), 'Native filtered denominator/maximum differs')
                queries.append(dict(metric=metric,density=density,subjects=len(result['summary'])))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    require(seen == set(reduced['models']) and len(seen) == 127 and observations == 81280,
            'Expected every127model/128case/five-metric observation exactly once')
    comparison = MeasurementComparison(tables)
    complete = comparison.read('pressure_call_ms',limit=1)['query']
    require(len(complete['summary']) == 127 and all(s['states']['measured'] == 128 for s in complete['summary']),
            'Cross-packet native comparison omitted measured cases')
    require('torch' not in sys.modules, 'Independent packet qualification imported a neural runtime')
    value = dict(status='all81280_pressure_accuracy_and_GPU_timing_native_scalars_verified',created_unix=time.time(),
        packets=2,subjects=127,cases=128,observations=observations,all20raw_samples_per_case_exact=True,
        all_common_accuracy_metrics_and_domain_density_dimensions_exact=True,
        native_density_query_checks=queries,native_cross_packet_denominators_exact=True,
        known_host_overlap_filter_preserved=True,new_neural_forwards=0,native_project_mutated=False,
        actual_UI_qualified=False,full_portfolio_complete=False,
        evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
                  (REPORT,OUTPUT/'preparation.json',OUTPUT/'source-project.json')],producer_sha256=sha(Path(__file__)))
    QUALIFIED.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'],flush=True)


if __name__ == '__main__':
    run()
