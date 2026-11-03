"""Codec-authored scalar boundary fixtures for actual browser interaction.

This is synthetic representation/lifecycle evidence, not laboratory accuracy or
native service acceptance. Original JSON scalar bindings remain fully validated.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok-measurements'


def prepare(output):
    sys.path[:0]=[str(SOURCE),str(SOURCE/'tests')]
    from retained_measurements import MeasurementTable,encode
    from test_retained_measurements import fixture
    result={}
    for name,count in [('base',2),('large',41)]:
        packet=fixture()
        if count>2:
            originals=copy.deepcopy(packet['subjects']);rows=copy.deepcopy(packet['observations'])
            packet['subjects']=[];packet['observations']=[]
            for index in range(count):
                original=originals[index%2];subject=copy.deepcopy(original)
                subject.update(id=f's{index:03d}',label=f'Subject {index:03d}')
                packet['subjects'].append(subject)
                for observation in rows:
                    if observation['subject_id']!=original['id']:continue
                    row=copy.deepcopy(observation)
                    row['subject_id']=subject['id'];row['id']=subject['id']+'-'+row['case_id']+'-'+row['metric_id']
                    packet['observations'].append(row)
        else:packet['subjects'][0]['label']='<img src=x onerror=bad>'
        raw=encode(packet);table=MeasurementTable(raw)
        source=dict(format='project_measurement_source_v1',origin_catalog_id='a'*32,id=table.sha256,title=packet['title'],
            blob=dict(sha256=table.sha256,bytes=len(raw)),source=dict(origin_catalog_id='a'*32,project_id='c'*32,
                manifest_sha256='f'*64,reference_id='fixture',path='fixture.json',measured_sha256=table.sha256),
            actor_id='b'*32,imported_at='2026-10-23T00:00:00+00:00')
        availability=dict(state='unverified',expected_sha256=table.sha256,expected_bytes=len(raw),observed_sha256=None,observed_bytes=len(raw))
        subjects=[s['id'] for s in packet['subjects']]
        groups=[subjects[:40],subjects[40:]] if count>2 else [subjects,[subjects[0]],[subjects[1]]]
        queries=[]
        for metric in packet['metrics']:
            for selected in groups:
                for filters in ({},{'cpm':16},{'grid_x':16},{'cpm':16,'grid_x':16},{'cpm':None},{'device':'CPU'}):
                    first=table.query(metric['id'],subject_ids=selected,filters=filters,limit=50)
                    for offset in range(0,max(1,first['total']),50):
                        request=dict(metric_id=metric['id'],subject_ids=selected,filters=filters,offset=offset,limit=50)
                        queries.append(dict(request=request,value=table.query(**request)))
        result[name]=dict(source=source,source_item=dict(source,availability=availability),metadata=table.metadata(),
            subjects=packet['subjects'],queries=queries,inspections={row['id']:table.inspect(row['id']) for row in packet['observations']})
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(fixture=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),scientific_validity_claim=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
