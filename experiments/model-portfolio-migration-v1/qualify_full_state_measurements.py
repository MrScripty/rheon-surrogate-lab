"""Audit every packaged scalar against original captures, then native reductions.

This is metadata-only qualification, not a model benchmark or native ingestion.
The original JSON files, independently selected metric paths and native query
results are the deciding evidence; no numerical/runtime package is imported.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
FIELDS={'pressure_relative_rms':('relative','pressure'),'velocity_relative_rms':('relative','velocity'),
    'tracer_relative_rms':('relative','tracer'),'pressure_equation_relative_l2':('relative','equation'),
    'pressure_absolute_rms':('absolute_rms','pressure'),'velocity_absolute_rms':('absolute_rms','velocity'),
    'tracer_absolute_rms':('absolute_rms','tracer')}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def same(a,b):return struct.pack('!d',a)==struct.pack('!d',b)


def qualify(bundle_path,source,report):
    require(not report.exists(),'Fresh qualification report required')
    started=time.monotonic();sys.path.insert(0,str(source))
    from retained_measurements import MeasurementTable,MeasurementComparison
    bundle=json.loads(bundle_path.read_bytes());base=bundle_path.parent
    tables={};subjects={};originals={};row_indices={};counts=dict(measured=0,unavailable=0);table_parts=[]
    for part in bundle['packets']:
        path=base/part['path'];require(sha(path)==part['sha256'],'Prepared packet bytes changed')
        raw=path.read_bytes();doc=json.loads(raw);table=MeasurementTable(raw)
        tables.setdefault(part['scope'],[]).append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
        table_parts.append(dict(path=part['path'],sha256=table.sha256,bytes=len(raw)))
        context=json.loads(base64.b64decode(doc['sources'][0]['data_base64']))
        inventory={c['source_id']:c for c in context['capture_inventory']}
        for value in doc['sources'][1:]:
            evidence=inventory[value['id']];path=ROOT/evidence['path']
            require(sha(path)==evidence['sha256']==value['sha256'] and path.read_bytes()==base64.b64decode(value['data_base64']),
                    'Packet does not retain the complete exact original replay bytes')
            if value['sha256'] not in originals:
                original=json.loads(path.read_bytes());originals[value['sha256']]=original
                index={}
                for n,row in enumerate(original['rows']):
                    k=(row['identifier'],row.get('start_fine_step',row.get('start')),row['K'],row['lead'],
                       row.get('output_steps_per_s',60),row.get('mode'))
                    require(k not in index,'Original replay output identity repeats');index[k]=(n,row)
                row_indices[value['sha256']]=index
        case_map={c['id']:c['dimensions'] for c in doc['cases']};source_map={s['id']:s for s in doc['sources']}
        local={s['id']:s for s in doc['subjects']}
        for identity,s in local.items():
            require(identity not in subjects,'Execution subject repeats across partitions')
            original=originals[source_map[s['reference']['source_id']]['sha256']]
            mode=s['dimensions']['execution_variant'] if context['temporal'] else None
            subjects[identity]=dict(original=original,mode=mode,expected=[],scope=part['scope'])
        for observation in doc['observations']:
            s=local[observation['subject_id']];c=case_map[observation['case_id']];source_id=s['reference']['source_id']
            original_sha=source_map[source_id]['sha256'];mode=s['dimensions']['execution_variant'] if context['temporal'] else None
            k=(c['trajectory_id'],c['start_index'],c['K'],c['lead'],c['output_steps_per_s'],mode)
            found=row_indices[original_sha].get(k);metric=observation['metric_id'];counts[observation['state']]+=1
            if found is None:
                require(observation['state']=='unavailable' and observation['value'] is None
                        and observation['reference']==dict(source_id=source_id,pointer='/status'), 'Unconsumed case must remain explicitly unavailable')
                require(all(v is None for v in observation['dimensions'].values()),'Missing output cannot claim executed work')
                continue
            n,row=found
            require([c[x] for x in ('grid_x','grid_y','grid_z')]==row['counts_xyz'] and c['cells_per_metre']==row['cells_per_metre']
                    and c['split']==row['split'],'Packaged density/domain/split differs from original physical output')
            total=row.get('actual_internal_advances',row.get('actual_internal_advances_per_block'))
            require(observation['dimensions']['advances_block']==total
                    and observation['dimensions']['advances_output']==total//row['K']
                    and observation['dimensions']['diagnostic_pass']==row['all_accuracy_pass'],'Packaged actual work/diagnostic state differs')
            if metric in FIELDS:
                group,field=FIELDS[metric]
                require(observation['state']=='measured' and same(observation['value'],row['metrics'][group][field])
                        and observation['reference']==dict(source_id=source_id,pointer=f'/rows/{n}/metrics/{group}/{field}'),
                        'Scalar or its meaning/provenance differs from original capture')
            else:
                require(metric in ('complete_block_ms','per_output_ms') and observation['state']=='unavailable'
                        and observation['value'] is None and observation['reference']==dict(source_id=source_id,pointer='/latency_claim'),
                        'Unqualified latency must not become a measured speed')
            if metric=='pressure_relative_rms':subjects[s['id']]['expected'].append((row,c))
    require(counts==dict(measured=129626,unavailable=46954),'Whole retained scalar population differs')
    queries=[]
    for scope,inputs in tables.items():
        # This construction is the actual combined-view capacity gate. The
        # old engine refuses171720rows even though each packet is valid.
        comparison=MeasurementComparison(inputs)
        selections=[{}]
        if scope=='old-diversity-full-state-2120':
            selections += [dict(shared_anchor_case=True),dict(cells_per_metre=100,K=4),dict(grid_x=64)]
        else:selections += [dict(output_steps_per_s=r) for r in (30,60,120)]
        for filters in selections:
            value=comparison.read('pressure_relative_rms',filters=filters,limit=1)['query']
            for summary in value['summary']:
                wanted=[row['metrics']['relative']['pressure'] for row,c in subjects[summary['subject_id']]['expected']
                        if all(c.get(k)==v for k,v in filters.items())]
                require(summary['states']['measured']==len(wanted),'Native reduction dropped or duplicated measured cases')
                maximum=summary['maximum']
                require((maximum is None and not wanted) or (maximum is not None and wanted and same(maximum['value'],max(wanted))),
                        'Native worst-case scalar differs from complete original population')
            queries.append(dict(scope=scope,filters=filters,total=value['total'],summary=value['summary']))
        for metric in ('complete_block_ms','per_output_ms'):
            missing=comparison.read(metric,limit=1)['query']
            require(all(s['maximum'] is None and not s['states']['measured'] for s in missing['summary']),
                    'Native view invented qualified timing')
    mixed=[item for inputs in tables.values() for item in inputs]
    try:MeasurementComparison(mixed)
    except Exception as error:require(getattr(error,'code',None)=='incompatible','Wrong rejection of scientifically different scopes')
    else:raise ValueError('Temporal/diversity populations were incorrectly pooled')
    result=dict(status='qualified_exact_full_state_packets_and_native_comparisons',created_unix=time.time(),
        bundle=str(bundle_path.relative_to(ROOT)),bundle_sha256=sha(bundle_path),packets=table_parts,
        measured_scalars_exact=counts['measured'],explicit_unavailable_exact=counts['unavailable'],execution_subjects=len(subjects),
        original_replay_sources_exact=len(originals),queries=queries,all_case_filters_and_maxima_match_originals=True,
        raw_capture_bytes_preserved=True,latency_measured=False,new_neural_forwards=0,native_ingestion_or_UI=False,
        scope_mixing_rejected=True,source_path=str(source.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),source/'retained_measurements.py']},
        elapsed_seconds=time.monotonic()-started)
    with report.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','measured_scalars_exact','explicit_unavailable_exact','execution_subjects']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--native-source',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();qualify(args.bundle.resolve(),args.native_source.resolve(),args.report.resolve())
