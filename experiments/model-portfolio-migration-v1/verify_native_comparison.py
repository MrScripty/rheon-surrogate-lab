"""Compare a native aggregate reply against the original frozen pressure captures.

Uses JSON and binary64 equality only. It imports neither the new comparison
implementation nor a model runtime; maxima come from the original capture rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/model-portfolio-migration-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(request_path, response_path, report):
    require(not report.exists(), 'Fresh comparison verification report required')
    request = json.loads(request_path.read_bytes())
    response = json.loads(response_path.read_bytes())
    require(request['operation']=='project_files.measurement_compare' and response['status']==200, 'Expected successful native comparison')
    arguments = request['arguments']; bound = response['result']
    require(bound['binding']==arguments['binding'] and bound['checkpoint']==arguments['checkpoint'], 'Native read selection changed')
    value = bound['value']; comparison = value['value']; query = comparison['query']
    require(query is not None and query['metric']['id']==arguments['metric_id'], 'Expected reviewed metric query')
    require(query['filters']==arguments.get('filters',{}), 'Filter request changed')
    bundle_path = BASE/'native-pressure-portfolio-bundle-01'
    bundle = json.loads((bundle_path/'bundle.json').read_bytes())
    raw = (ROOT/bundle['source_score']['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==bundle['source_score']['sha256'], 'Original score inventory changed')
    scores = json.loads(raw)
    originals = {}
    for identity, model in scores['models'].items():
        raw = (ROOT/model['result']['path']).read_bytes()
        if 'sha256' in model['result']:
            require(hashlib.sha256(raw).hexdigest()==model['result']['sha256'], 'Original capture identity changed')
        originals[identity] = json.loads(raw)
    unavailable = {row['identifier'] for row in scores['unavailable']}
    selected_sources = [(row['origin_catalog_id'],row['packet_sha256']) for row in arguments['sources']]
    actual_sources = [(row['origin_catalog_id'],row['id']) for row in value['sources']]
    require(selected_sources==actual_sources and len(set(actual_sources))==len(actual_sources)==9, 'Exact nine-packet source selection changed')
    require({key[1] for key in actual_sources}=={part['sha256'] for part in bundle['packets']}, 'Incomplete or foreign pressure packet')
    by_subject = {}
    for part in bundle['packets']:
        raw = (bundle_path/part['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==part['sha256'], 'Original packet bytes changed')
        packet = json.loads(raw)
        origin = next(origin for origin,digest in actual_sources if digest==part['sha256'])
        locator = dict(origin_catalog_id=origin,packet_sha256=part['sha256'])
        for subject in packet['subjects']:
            require(subject['id'] not in by_subject, 'Repeated original subject')
            by_subject[subject['id']] = (subject,packet,locator)
    require(set(by_subject)==set(originals)|unavailable and len(by_subject)==129, 'Complete129subject inventory required')
    require(len(comparison['subjects'])==129, 'Native metadata omitted a subject')
    for subject in comparison['subjects']:
        original,packet,locator = by_subject[subject['id']]
        require({key:value for key,value in subject.items() if key!='source'}==original and subject['source']==locator, 'Native subject facts/provenance changed')
    selected = arguments.get('subject_ids',[subject['id'] for subject in comparison['subjects']])
    require(query['subject_ids']==selected and [row['subject_id'] for row in query['summary']]==selected, 'Subject selection or summary order changed')
    metric = arguments['metric_id']; measured = missing = 0
    for summary in query['summary']:
        identity = summary['subject_id']; subject,packet,locator = by_subject[identity]
        require(summary['source']==locator, 'Maximum belongs to another original packet')
        indices = [index for index,case in enumerate(packet['cases'])
            if all({**subject['dimensions'],**case['dimensions']}.get(key)==wanted for key,wanted in query['filters'].items())]
        expected_measured = identity in originals and metric!='pressure_call_ms'
        count = len(indices)
        states = dict(measured=count if expected_measured else 0,failed=0,unsupported=0,unavailable=0 if expected_measured else count)
        require(summary['count']==count and summary['states']==states, 'Per-model complete denominator or missingness changed')
        if not expected_measured or not indices:
            require(summary['maximum'] is None, 'Unavailable/empty observations acquired a value')
            missing += count
            continue
        values = {packet['cases'][index]['id']:originals[identity]['rows'][index]['metrics'][metric] for index in indices}
        maximum = summary['maximum']
        require(maximum is not None and maximum['case_id'] in values, 'Maximum case is outside the original selected population')
        require(struct.pack('!d',maximum['value'])==struct.pack('!d',max(values.values())), 'Native maximum differs from original binary64 capture')
        require(struct.pack('!d',maximum['value'])==struct.pack('!d',values[maximum['case_id']]), 'Maximum scalar belongs to another original case')
        original_observation = next(row for row in packet['observations']
            if row['subject_id']==identity and row['metric_id']==metric and row['case_id']==maximum['case_id'])
        require(maximum['id']==original_observation['id'], 'Maximum observation reference changed')
        measured += count
    require(query['total']==measured+missing, 'Global observation population changed')
    offset,limit = arguments.get('offset',0),arguments.get('limit',50)
    require((query['offset'],query['limit'])==(offset,limit) and len(query['rows'])==min(limit,max(0,query['total']-offset)), 'Global page continuation changed')
    expected_rows = []
    for origin,digest in selected_sources:
        packet = next(packet for subject,packet,locator in by_subject.values() if locator['packet_sha256']==digest)
        cases = {case['id']:case for case in packet['cases']}
        subjects = {subject['id']:subject for subject in packet['subjects']}
        for observation in packet['observations']:
            if observation['subject_id'] not in selected or observation['metric_id']!=metric:
                continue
            dimensions = {**subjects[observation['subject_id']]['dimensions'],
                **cases[observation['case_id']]['dimensions'],**observation['dimensions']}
            if any(dimensions.get(key)!=wanted for key,wanted in query['filters'].items()):
                continue
            expected_rows.append(dict(observation,dimensions=dimensions,
                source=dict(origin_catalog_id=origin,packet_sha256=digest)))
    require(query['rows']==expected_rows[offset:offset+limit], 'Global row page differs from original packets or source order')
    result = dict(schema='rheon.native-comparison-original-verification.v1',status='passed_all_native_maxima_and_denominators_against_original_captures',
        created_unix=time.time(),metric=metric,filters=query['filters'],subjects=len(selected),source_packets=9,
        observations=query['total'],measured=measured,explicit_unavailable=missing,
        project_id=bound['project_id'],checkpoint=bound['checkpoint'],original_binary64_equality=True,
        native_request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        native_response_sha256=hashlib.sha256(response_path.read_bytes()).hexdigest(),exact_original_global_page=True,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_neural_forwards=0)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    verify(args.request.resolve(),args.response.resolve(),args.report.resolve())
