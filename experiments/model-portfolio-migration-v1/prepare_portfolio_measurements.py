"""Retain every named pressure result in bounded native scalar packets.

The qualified two-subject packet supplies the exact case/scope/metric definitions.
Original captures and unavailable-checkpoint observations remain source-bound.
This reads retained bytes only: no neural model, new scoring or physics execution.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok'
BASE=ROOT/'runs/model-portfolio-migration-v1'
DOCS=ROOT/'docs/plans/model-portfolio-migration-v1'
TEMPLATE=BASE/'native-measurements-packet-05/measurements.json'
TEMPLATE_SHA='c78bbc342cad9f3d9af5116f2c272e85f1d01344f7af0f33f4dce000de828400'


def file_sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def source(identity,label,raw):
    return dict(id=identity,label=label,kind='json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data_base64=base64.b64encode(raw).decode('ascii'))


def prepare(output):
    if output.exists() or not output.is_relative_to(BASE):raise ValueError('Fresh task-owned portfolio output required')
    sys.path.insert(0,str(SOURCE))
    from retained_measurements import MeasurementTable,encode,digest
    from projects import manifest as source_manifest
    raw=TEMPLATE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TEMPLATE_SHA:raise ValueError('Qualified source packet changed')
    MeasurementTable(raw)
    template=json.loads(raw)
    original_sources={entry['id']:entry for entry in template['sources']}
    context=json.loads(base64.b64decode(original_sources['context']['data_base64']))
    score_path=DOCS/'reports/pressure-common-score-01.json';score_raw=score_path.read_bytes();score=json.loads(score_raw)
    if hashlib.sha256(score_raw).hexdigest()!=original_sources['common-score']['sha256']:
        raise ValueError('Qualified common-score evidence changed')
    if (len(score['models'])!=127 or score['unique_requests_per_checkpoint']!=128
            or not score['independent_pressure_scoring_exact'] or score['latency_claim']
            or score['maximum_captured_versus_common_equation_metric_difference']!=0):
        raise ValueError('Complete accepted unchanged common-pressure observation required')
    work_path=BASE/'discovery/model-worklist-05.json';work_raw=work_path.read_bytes();work=json.loads(work_raw)
    aliases={entry['label']:entry for entry in work['named_entries'] if entry['item_id'].startswith('cohorts/')}
    unavailable={row['identifier']:(index,row) for index,row in enumerate(score['unavailable'])}
    identifiers=sorted(set(score['models'])|set(unavailable),key=lambda name:int(name[1:]))
    if len(identifiers)!=129 or set(identifiers)!=set(aliases) or set(unavailable)!={'C01','C02'}:
        raise ValueError('Named pressure coverage differs; never silently omit an alias')
    captures={};capture_bytes={};receipts={}
    case_keys=tuple(context['cases'][0]['original_case'])
    denominator_keys=('equation_rhs_l2','pressure_reference_mean_free_l2','equation_floor_active','pressure_floor_active')
    metric_ids=[metric['id'] for metric in template['metrics']]
    for name,observation in score['models'].items():
        receipt=observation['result'];path=ROOT/receipt['path'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=receipt['sha256']:raise ValueError('Changed pressure capture: '+name)
        value=json.loads(raw)
        if (value['identifier']!=name or value['cases']!=128 or len(value['rows'])!=128
                or value['status']!='completed_captured_pressure_GPU_accuracy' or value['latency_claim']
                or value['CPU_model_forward'] or value['neural_module_forward_calls']['cpu']
                or not value['architecture_unchanged'] or not value['learned_parameters_unchanged']
                or value['weights_sha256']!=observation['weights_sha256']
                or value['weights_sha256']!=aliases[name]['declared_weights_sha256']):
            raise ValueError('Capture identity, parameter, device or coverage facts differ: '+name)
        # These are exact source definitions, not independently reinterpreted math.
        reference=json.loads(base64.b64decode(original_sources['C107']['data_base64']))
        if any(value[key]!=reference[key] for key in ('pressure_metric','equation_metric')):
            raise ValueError('Pressure metric meaning differs: '+name)
        if file_sha(path.parent/value['output_archive']['path'])!=value['output_archive']['sha256']:
            raise ValueError('Captured pressure output archive changed: '+name)
        for index,row in enumerate(value['rows']):
            original=context['cases'][index]
            if {key:row[key] for key in case_keys}!=original['original_case']:
                raise ValueError('Original case/input/reference identity differs: '+name)
            for key in denominator_keys:
                if row['metrics'][key]!=original['dimensions'][key]:raise ValueError('Normalization evidence differs: '+name)
        for metric in metric_ids:
            if metric=='pressure_call_ms':continue
            common='common_equation_relative_l2' if metric=='equation_relative_l2' else metric
            actual=max(row['metrics'][metric] for row in value['rows'])
            if struct.pack('!d',actual)!=struct.pack('!d',observation['worst'][common]):
                raise ValueError('Retained maximum differs from common qualification: '+name)
        captures[name]=value;capture_bytes[name]=raw;receipts[name]=receipt
    context['capture_receipts']=receipts
    context['unavailable_subjects']=score['unavailable']
    context['worklist_source']=dict(path=str(work_path.relative_to(ROOT)),sha256=hashlib.sha256(work_raw).hexdigest(),
                                  interpretation='Derived alias/fit metadata; original model capture remains authoritative.')
    context_source=source('context','Exact shared case/scope/metric definitions and capture inventory',encode(context))
    extra_dimensions=[dict(id=identity,label=label,type='string',units=None,scope='subject') for identity,label in
        [('execution_variant','Recorded execution variant'),('fit_item_id','Retained fit-card identity'),
         ('model_name','Recorded model name'),('dashboard_family','Recorded dashboard family')]]
    dimensions=copy.deepcopy(template['dimensions'])+extra_dimensions
    subject_dimensions=[d['id'] for d in template['dimensions'] if d['scope']=='subject']
    output.mkdir();packets=[];measured=missing=0;subject_index=[]
    for start in range(0,len(identifiers),16):
        names=identifiers[start:start+16];subjects=[];rows=[]
        sources=[context_source,original_sources['packet'],original_sources['common-score']]
        for name in names:
            alias=aliases[name];capture=captures.get(name)
            if capture is not None:
                sources.append(source(name,'Original '+name+' GPU-backed pressure accuracy capture',capture_bytes[name]))
                values={key:capture[key] for key in subject_dimensions}
                values['execution_variant']=capture.get('execution_variant')
                reference=dict(source_id=name,pointer='')
            else:
                index,absent=unavailable[name]
                if absent['weights_sha256']!=alias['declared_weights_sha256']:raise ValueError('Unavailable checkpoint identity differs')
                values={key:None for key in subject_dimensions};values['weights_sha256']=absent['weights_sha256']
                values['execution_variant']=None
                reference=dict(source_id='common-score',pointer='/unavailable/'+str(index))
            values.update(fit_item_id=alias['fit_item_id'],model_name=alias['record']['name'],dashboard_family=alias['record']['family'])
            subjects.append(dict(id=name,label=name,dimensions=values,reference=reference))
            subject_index.append(dict(identifier=name,alias_item_id=alias['item_id'],fit_item_id=alias['fit_item_id'],
                checkpoint_sha256=alias['declared_weights_sha256'],state='captured_accuracy' if capture else 'checkpoint_not_located',
                actual_pressure_requests=128 if capture else 0))
            for index in range(128):
                for metric in metric_ids:
                    unavailable_value=capture is None or metric=='pressure_call_ms'
                    if capture is None:
                        reference_value=reference
                        reason='Original checkpoint not located; no substitute parameters and no prediction performed.'
                    elif metric=='pressure_call_ms':
                        reference_value=dict(source_id=name,pointer='/latency_claim')
                        reason='Accuracy capture explicitly has latency_claim=false; no qualified call timing.'
                    else:reference_value=dict(source_id=name,pointer=f'/rows/{index}/metrics/{metric}');reason=None
                    value=None if unavailable_value else capture['rows'][index]['metrics'][metric]
                    rows.append(dict(id=name+f'-case-{index:03d}-'+metric,subject_id=name,case_id=f'case-{index:03d}',metric_id=metric,
                        state='unavailable' if unavailable_value else 'measured',value=value,reason=reason,dimensions={},reference=reference_value))
                    if unavailable_value:missing+=1
                    else:measured+=1
        packet=dict(format='retained_measurements_v1',schema_version=1,title=f'{names[0]}–{names[-1]} · original pressure cases ({len(names)} subjects)',
            scope=copy.deepcopy(template['scope']),dimensions=dimensions,subjects=subjects,cases=copy.deepcopy(template['cases']),
            metrics=copy.deepcopy(template['metrics']),observations=rows,sources=sources)
        raw=encode(packet);table=MeasurementTable(raw)
        if table.metadata()['observation_count']!=len(names)*128*5:raise ValueError('Incomplete partition')
        relative=f'pressure-part-{len(packets)+1:02d}.json'
        (output/relative).write_bytes(raw)
        packets.append(dict(path=relative,sha256=table.sha256,bytes=len(raw),subjects=names,
            observations=len(rows),measured=sum(r['state']=='measured' for r in rows),
            unavailable=sum(r['state']=='unavailable' for r in rows),scope_definition_sha256=packet['scope']['definition']['sha256'],
            metric_definition_sha256={m['id']:m['definition']['sha256'] for m in packet['metrics']}))
    if (measured,missing)!=(65024,17536):raise ValueError('Full named pressure scalar denominator differs')
    if len({p['scope_definition_sha256'] for p in packets})!=1 or len({json.dumps(p['metric_definition_sha256'],sort_keys=True) for p in packets})!=1:
        raise ValueError('Partitions do not share exact comparison definitions')
    bundle=dict(schema='rheon.native-pressure-portfolio-bundle.v1',status='prepared_not_native_imported',
        source_score=dict(path=str(score_path.relative_to(ROOT)),sha256=hashlib.sha256(score_raw).hexdigest()),
        source_worklist=dict(path=str(work_path.relative_to(ROOT)),sha256=hashlib.sha256(work_raw).hexdigest()),
        template_sha256=TEMPLATE_SHA,scope_definition_sha256=packets[0]['scope_definition_sha256'],
        metric_definition_sha256=packets[0]['metric_definition_sha256'],case_binding_sha256=context['scope']['case_binding_sha256'],
        cases_per_subject=128,subjects=subject_index,available_capture_subjects=127,unlocated_checkpoint_subjects=2,
        actual_pressure_requests_retained=16256,measured_scalar_values=measured,explicit_unavailable_observations=missing,
        packets=packets,new_neural_forwards=0,new_scoring=False,new_reference_generation=False,
        qualified_latency=False,complete_model_portfolio=False)
    (output/'bundle.json').write_bytes(encode(bundle))
    data=[dict(id=f'pressure-part-{index+1:02d}',label=f"{part['subjects'][0]}–{part['subjects'][-1]} · full per-case observations",
        path=(output/part['path']).relative_to(ROOT).as_posix(),sha256=part['sha256']) for index,part in enumerate(packets)]
    manifest=dict(schema_version=1,id=digest(bundle)[:32],name='Complete named pressure comparison · retained observations',
                  data=data,recipes=[],evidence=[],requirements=[])
    source_manifest(manifest);(output/'source-project.json').write_bytes(encode(manifest))
    if any(name=='torch' or name.startswith('torch.') or name=='numpy' for name in sys.modules):raise ValueError('No numerical/model runtime belongs in packaging')
    print(json.dumps(dict(status=bundle['status'],subjects=len(subject_index),packets=len(packets),
        measured_scalars=measured,explicit_unavailable=missing,total_packet_bytes=sum(p['bytes'] for p in packets))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
