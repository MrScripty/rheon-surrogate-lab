"""Independently check complete native portfolio preparation against originals.

No model, reference solver, new metric evaluation or checkpoint deserializer.
Missing observations are checked explicitly rather than excluded from coverage.
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
BASE=ROOT/'runs/model-portfolio-migration-v1'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def verify(bundle_directory,cards_directory,report_path):
    require(not report_path.exists(),'Fresh independent verification report required')
    sys.path.insert(0,str(BASE/'sources/Tuldok'))
    from retained_measurements import MeasurementTable
    from project_files import MAX_FILE_BYTES
    bundle=json.loads((bundle_directory/'bundle.json').read_bytes())
    template_raw=(BASE/'native-measurements-packet-05/measurements.json').read_bytes()
    require(sha(template_raw)=='c78bbc342cad9f3d9af5116f2c272e85f1d01344f7af0f33f4dce000de828400','Qualified template changed')
    template=json.loads(template_raw)
    template_sources={entry['id']:entry for entry in template['sources']}
    template_context=json.loads(base64.b64decode(template_sources['context']['data_base64']))
    preparation=json.loads((cards_directory/'preparation.json').read_bytes())
    work=json.loads((BASE/'discovery/model-worklist-05.json').read_bytes())
    common_path=ROOT/bundle['source_score']['path'];common_raw=common_path.read_bytes();common=json.loads(common_raw)
    require(sha(common_raw)==bundle['source_score']['sha256'],'Common-score original changed')
    original={name:json.loads((ROOT/item['result']['path']).read_bytes()) for name,item in common['models'].items()}
    unavailable={entry['identifier']:entry for entry in common['unavailable']}
    wanted=set(original)|set(unavailable)
    seen_subjects=set();seen_rows=set();states={};exact=0;missing=0;packet_receipts=[];case_documents=None;context_sha=None
    for part in bundle['packets']:
        path=bundle_directory/part['path'];raw=path.read_bytes()
        require(sha(raw)==part['sha256'] and len(raw)==part['bytes'],'Partition identity changed')
        table=MeasurementTable(raw);packet=json.loads(raw)
        sources={entry['id']:entry for entry in packet['sources']}
        decoded={name:base64.b64decode(entry['data_base64']) for name,entry in sources.items()}
        context=json.loads(decoded['context'])
        require(packet['scope']==template['scope'] and packet['metrics']==template['metrics'],'Accepted metric/scope definitions were changed')
        require(packet['cases']==template['cases'] and context['cases']==template_context['cases'],'Accepted original case definitions were changed')
        require(context['scope']==template_context['scope'] and context['metrics']==template_context['metrics'],'Scientific definition source changed')
        require(decoded['packet']==base64.b64decode(template_sources['packet']['data_base64']),'Original consumed dataset manifest bytes changed')
        if case_documents is None:case_documents=context['cases'];context_sha=sources['context']['sha256']
        require(case_documents==context['cases'] and context_sha==sources['context']['sha256'],'Partition case/context definitions differ')
        require(packet['scope']['definition']['sha256']==bundle['scope_definition_sha256'],'Partition scope differs')
        require({m['id']:m['definition']['sha256'] for m in packet['metrics']}==bundle['metric_definition_sha256'],'Partition metric meaning differs')
        require(decoded['common-score']==common_raw,'Original common-score bytes changed in a partition')
        for subject in packet['subjects']:
            name=subject['id'];require(name in wanted and name not in seen_subjects,'Missing, duplicate or foreign subject')
            seen_subjects.add(name);states[name]={state:0 for state in ('measured','unavailable')}
            if name in original:
                evidence=ROOT/common['models'][name]['result']['path']
                require(decoded[name]==evidence.read_bytes(),'Original capture bytes changed')
                require(subject['dimensions']['weights_sha256']==original[name]['weights_sha256'],'Subject checkpoint identity changed')
                require(subject['dimensions']['execution_variant']==original[name].get('execution_variant'),'Recorded execution boundary changed')
            else:
                require(subject['dimensions']['weights_sha256']==unavailable[name]['weights_sha256'],'Unavailable checkpoint identity changed')
                require(all(v is None for k,v in subject['dimensions'].items() if k in
                    ('inference_device','gpu_model','network_dtype','pressure_dtype','tf32','autocast','torch_version','cuda_version','execution_variant')),
                    'Unexecuted subject acquired invented execution metadata')
        for row in packet['observations']:
            name=row['subject_id'];index=int(row['case_id'].split('-')[-1]);metric=row['metric_id']
            identity=(name,index,metric);require(identity not in seen_rows,'Duplicate Cartesian observation');seen_rows.add(identity)
            require(row['case_id']==case_documents[index]['id'],'Case identity changed')
            should_measure=name in original and metric!='pressure_call_ms'
            require(row['state']==('measured' if should_measure else 'unavailable'),'Observation missingness changed')
            states[name][row['state']]+=1
            if should_measure:
                raw_row=original[name]['rows'][index]
                case=case_documents[index]['original_case']
                require({k:raw_row[k] for k in case}==case,'Original input/reference pairing changed')
                require(row['reference']==dict(source_id=name,pointer=f'/rows/{index}/metrics/{metric}'),'Scalar pointer changed case/metric meaning')
                require(struct.pack('!d',row['value'])==struct.pack('!d',raw_row['metrics'][metric]),'Binary64 scalar changed')
                exact+=1
            else:
                require(row['value'] is None and bool(row['reason']),'Missing value was dropped or replaced')
                missing+=1
        packet_receipts.append(dict(path=part['path'],sha256=part['sha256'],subjects=len(packet['subjects']),
            observations=table.metadata()['observation_count'],bytes=len(raw)))
    metrics={'equation_relative_l2','pressure_mean_free_relative_l2','pressure_absolute_rms','pressure_absolute_max','pressure_call_ms'}
    require(seen_subjects==wanted,'Portfolio subject coverage incomplete')
    require(len(seen_rows)==len(wanted)*128*len(metrics),'Full per-subject case/metric Cartesian denominator incomplete')
    for name in wanted:
        require(states[name]==dict(measured=512 if name in original else 0,unavailable=128 if name in original else 640),
                'Per-subject measured/missing support is wrong')
    require((exact,missing)==(65024,17536),'Full scalar totals differ')
    fits={fit['item_id']:fit for fit in work['fit_cards']};fit_notes={}
    for note in preparation['notes']:
        raw=(cards_directory/note['path']).read_bytes()
        require(len(raw)==note['bytes'] and sha(raw)==note['sha256'] and len(raw)<=MAX_FILE_BYTES,'Native note identity/bound changed')
        raw.decode('utf-8')
        if 'fit_item_id' not in note:continue
        identity=note['fit_item_id'];require(identity in fits and identity not in fit_notes,'Fit card duplicated or foreign')
        offset=note['embedded_receipt_offset'];length=note['embedded_receipt_bytes'];receipt=raw[offset:offset+length]
        require(sha(receipt)==fits[identity]['receipt_sha256']==note['embedded_receipt_sha256'],'Embedded original receipt bytes changed')
        require(any((ROOT/p).is_file() and (ROOT/p).read_bytes()==receipt for p in fits[identity]['receipt_paths']),
                'Original retained receipt does not match the native card')
        start=raw.index(b'```json\n')+len(b'```json\n');end=raw.index(b'\n```\n\n## Exact original receipt bytes',start)
        facts=json.loads(raw[start:end]);require(facts['fit']==fits[identity],'Original fit facts were rewritten')
        if 'embedded_config_sha256' in note:
            offset=note['embedded_config_offset'];length=note['embedded_config_bytes'];config=raw[offset:offset+length]
            require(sha(config)==note['embedded_config_sha256']==json.loads(receipt)['closure']['config_sha256'],
                'Embedded configuration differs from its direct original receipt binding')
            require(facts['bound_training_configuration']['config_sha256']==sha(config),'Configuration fact binding differs')
            require(json.dumps(facts['bound_training_configuration']['configuration'],sort_keys=True)==json.dumps(json.loads(config),sort_keys=True),
                'Readable configuration facts differ from exact embedded configuration')
        fit_notes[identity]=note['path']
    require(set(fit_notes)==set(fits) and len(fits)==205,'Fit coverage incomplete')
    index=json.loads((cards_directory/'notes/portfolio-index.json').read_bytes())
    require({entry['item_id'] for entry in index['aliases']}=={entry['item_id'] for entry in work['named_entries']},'Dashboard aliases were merged or omitted')
    require(len(index['aliases'])==379,'Alias denominator changed')
    require({entry['item_id']:entry['card_path'] for entry in index['fits']}==fit_notes,'Native fit index/card links differ')
    checkpoint_source=json.loads((BASE/'discovery/checkpoint-identities-01.json').read_bytes())
    groups={entry['sha256']:entry for entry in checkpoint_source['groups']};seen_checkpoints=set()
    for part in index['checkpoint_partitions']:
        raw=(cards_directory/part['path']).read_bytes();require(sha(raw)==part['sha256'],'Checkpoint metadata partition changed')
        for entry in json.loads(raw)['rows']:
            digest=entry['sha256'];require(digest in groups and digest not in seen_checkpoints,'Checkpoint byte identity merged/omitted')
            seen_checkpoints.add(digest);require(entry['original_inventory']==groups[digest],'Historical checkpoint inventory rewritten')
    require(seen_checkpoints==set(groups) and len(seen_checkpoints)==524,'Checkpoint coverage incomplete')
    require(not any(n=='torch' or n.startswith('torch.') or n=='numpy' for n in sys.modules),'Numerical/model runtime imported')
    method_gaps=sum(not fit['method'] for fit in work['fit_cards'])
    require(preparation['missing_structured_method_descriptions']==method_gaps,'Structured method gap count differs from original worklist')
    config_count=sum('embedded_config_sha256' in note for note in preparation['notes'])
    require(config_count==preparation['exact_training_configurations_embedded']==51,'Recovered configuration coverage differs')
    report=dict(schema='rheon.native-portfolio-preparation-verification.v3',status='verified_complete_named_pressure_and_fit_checkpoint_index_preparation',
        created_unix=time.time(),verifier_sha256=sha(Path(__file__).read_bytes()),bundle=str(bundle_directory.relative_to(ROOT)),cards=str(cards_directory.relative_to(ROOT)),
        bundle_sha256=sha((bundle_directory/'bundle.json').read_bytes()),card_preparation_sha256=sha((cards_directory/'preparation.json').read_bytes()),
        subjects=129,available_capture_subjects=127,explicit_unlocated_subjects=['C01','C02'],cases_per_subject=128,metrics=5,
        original_pressure_requests=16256,exact_binary64_scalars=exact,explicit_unavailable_observations=missing,packets=packet_receipts,
        common_case_scope_metric_definitions_exact=True,original_capture_bytes_exact=True,fit_cards=205,original_receipts_exact=205,
        dashboard_aliases=379,checkpoint_byte_identities=524,native_notes=len(preparation['notes']),
        missing_structured_method_descriptions=method_gaps,new_neural_forwards=0,new_scoring=False,new_reference_generation=False,
        exact_training_configurations_embedded=config_count,
        native_import_qualified=False,complete_model_portfolio_accounting=False)
    report_path.write_bytes(encoded(report))
    print(json.dumps({key:report[key] for key in ('status','subjects','exact_binary64_scalars','explicit_unavailable_observations','fit_cards','checkpoint_byte_identities')}))


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--cards',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();verify(args.bundle.resolve(),args.cards.resolve(),args.report.resolve())
