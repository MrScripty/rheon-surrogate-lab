"""Codec-authored multi-packet browser fixtures; no laboratory accuracy claim."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok-measurement-comparison'


def prepare(output):
    if output.exists():
        raise ValueError('Fresh comparison DOM fixture required')
    sys.path[:0] = [str(SOURCE), str(SOURCE/'tests')]
    from retained_measurements import MeasurementTable, MeasurementComparison, encode
    from test_retained_measurements import fixture
    documents = []
    inputs = []
    inspections = {}
    for ordinal, (original_id, count) in enumerate([('a',41),('b',2)]):
        packet = fixture()
        original = next(subject for subject in packet['subjects'] if subject['id']==original_id)
        observations = [row for row in packet['observations'] if row['subject_id']==original_id]
        packet['subjects'] = []; packet['observations'] = []
        packet['title'] = 'Controlled original partition ' + original_id
        for index in range(count):
            identifier = f'm{index+(41 if ordinal else 0):03d}'
            subject = copy.deepcopy(original)
            subject.update(id=identifier, label='<img src=x onerror=bad>' if identifier=='m000' else identifier)
            packet['subjects'].append(subject)
            for observation in observations:
                row = copy.deepcopy(observation)
                row.update(subject_id=identifier,id=identifier+'-'+row['case_id']+'-'+row['metric_id'])
                if identifier in ('m000','m041') and row['case_id']=='case2' and row['metric_id']=='error':
                    row['id'] = 'shared-worst'
                packet['observations'].append(row)
        raw = encode(packet); table = MeasurementTable(raw); origin = original_id*32
        locator = dict(origin_catalog_id=origin,packet_sha256=table.sha256)
        source = dict(format='project_measurement_source_v1',origin_catalog_id=origin,id=table.sha256,
            title=packet['title'],blob=dict(sha256=table.sha256,bytes=len(raw)),
            source=dict(origin_catalog_id='c'*32,project_id='d'*32,manifest_sha256='f'*64,
                reference_id='fixture',path=original_id+'.json',measured_sha256=table.sha256),
            actor_id='b'*32,imported_at='2026-10-23T00:00:00+00:00')
        documents.append(source); inputs.append((locator,table))
        inspections[origin+'.'+table.sha256] = {row['id']:dict(source=source,value=table.inspect(row['id'])) for row in packet['observations']}
    items = [dict(source,availability=dict(state='unverified',expected_sha256=source['id'],
        expected_bytes=source['blob']['bytes'],observed_sha256=None,observed_bytes=source['blob']['bytes'])) for source in documents]
    comparisons = []
    for chosen in ([0],[0,1]):
        selected = [inputs[index] for index in chosen]; sources = [documents[index] for index in chosen]
        locators = [item[0] for item in selected]; comparison = MeasurementComparison(selected)
        comparisons.append(dict(request=dict(sources=locators),value=dict(sources=sources,value=comparison.read())))
        subjects = [subject['id'] for subject in comparison.read()['subjects']]
        groups = [subjects,subjects[:40],subjects[40:],[subjects[-1]]]
        unique = {tuple(group):group for group in groups if group}
        for metric in ('error','latency'):
            for group in unique.values():
                for filters in ({},{'cpm':16},{'grid_x':16},{'cpm':16,'grid_x':16},{'cpm':None},{'device':'CPU'}):
                    total = comparison.read(metric,subject_ids=group,filters=filters)['query']['total']
                    for offset in range(0,max(1,total),50):
                        request = dict(sources=locators,metric_id=metric,subject_ids=group,filters=filters,offset=offset,limit=50)
                        value = comparison.read(metric,subject_ids=group,filters=filters,offset=offset,limit=50)
                        comparisons.append(dict(request=request,value=dict(sources=sources,value=value)))
    result = dict(source_items=items,comparisons=comparisons,inspections=inspections,
        scope='Controlled Chromium/representation evidence only, not laboratory or native-service acceptance')
    with output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(fixture=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),scientific_validity_claim=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
