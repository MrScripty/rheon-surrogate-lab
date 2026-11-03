"""Prepare evidence drafts and a typed portfolio index for native note adoption.

Receipt bytes are embedded exactly, with recorded offsets/hashes for extraction.
Checkpoint files are hashed without deserialization, copying or execution. Model,
fit, alias, byte identity and execution-variant counts stay distinct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'
SOURCE=BASE/'sources/Tuldok'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def prepare(output,bundle_path,method_path):
    if output.exists() or not output.is_relative_to(BASE):raise ValueError('Fresh task-owned card output required')
    sys.path.insert(0,str(SOURCE))
    from project_files import MAX_FILE_BYTES
    inputs={}
    def load(name,path):
        raw=path.read_bytes();inputs[name]=dict(path=str(path.relative_to(ROOT)),sha256=sha(raw))
        return json.loads(raw)
    work=load('worklist',BASE/'discovery/model-worklist-05.json')
    checkpoints=load('checkpoint_inventory',BASE/'discovery/checkpoint-identities-01.json')
    contexts=load('checkpoint_contexts',BASE/'discovery/checkpoint-contexts-01.json')
    score=load('common_pressure',ROOT/'docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json')
    bundle=load('pressure_bundle',bundle_path)
    recovered=load('method_configurations',method_path)
    if recovered['worklist_sha256']!=inputs['worklist']['sha256']:
        raise ValueError('Method configurations belong to a different original worklist')
    configurations={row['fit_item_id']:row for row in recovered['rows'] if row['state']=='verified_original_config_bytes'}
    if len(work['fit_cards'])!=205 or len(checkpoints['groups'])!=524 or len(contexts['rows'])!=524:
        raise ValueError('Expected complete reconciled inputs; never silently truncate a portfolio')
    output.mkdir();notes=[]
    def write(relative,raw,**metadata):
        if len(raw)>MAX_FILE_BYTES:raise ValueError('Explicit note partition required: '+relative)
        raw.decode('utf-8')
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(raw)
        record=dict(kind='note',path=relative,sha256=sha(raw),bytes=len(raw),**metadata)
        notes.append(record);return record
    aliases={entry['item_id']:entry for entry in work['named_entries']}
    fit_index=[]
    for fit in work['fit_cards']:
        receipts=[]
        for relative in fit['receipt_paths']:
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT):raise ValueError('Receipt path leaves the retained workspace')
            if path.is_file() and hash_file(path)==fit['receipt_sha256']:receipts.append((relative,path))
        if not receipts:raise ValueError('No exact original receipt retained: '+fit['item_id'])
        relative,receipt=receipts[0];raw=receipt.read_bytes();raw.decode('utf-8')
        config_record=configurations.get(fit['item_id']);config_raw=None;bound_config=None
        if config_record is not None:
            config_path=(ROOT/config_record['recovered_path']).resolve()
            if not config_path.is_relative_to(ROOT):raise ValueError('Configuration leaves the retained workspace')
            config_raw=config_path.read_bytes()
            if (sha(config_raw)!=config_record['config_sha256'] or fit['receipt_sha256']!=config_record['receipt_sha256']
                    or json.loads(raw)['closure']['config_sha256']!=config_record['config_sha256']):
                raise ValueError('Configuration is not bound to this exact original receipt')
            bound_config={key:config_record[key] for key in ('config_sha256','graph_id','graph_document_sha256','recipe_path','recipe_sha256','config_path')}
            bound_config['configuration']=json.loads(config_raw)
        named=[aliases[key] for key in fit['named_entries']]
        pressure={entry['label']:score['models'][entry['label']] for entry in named
                  if entry['item_id'].startswith('cohorts/') and entry['label'] in score['models']}
        facts=dict(fit=fit,original_receipt=dict(path=relative,sha256=fit['receipt_sha256'],bytes=len(raw)),
            bound_training_configuration=bound_config,
            qualified_common_pressure=dict(scope='128 original consumed pressure requests; not a full-state, independent-validation or drift qualification',
                capture_and_common_score=inputs['common_pressure'],subjects=pressure,qualified_latency=False),
            interpretation=dict(card_status='retrospective_evidence_draft',architecture_description='Recorded fields; source semantics still require review.',
                purpose_and_lesson='No causal lesson or completion claim is inferred from a training receipt.',
                reproduction='Source/data/checkpoint identities retained. Clean reproduction remains unverified.',
                historical_original_Run_observations='References only; no native Run, Trial, ancestry or current execution authority is created.'))
        method=fit['method'] or (bound_config['configuration'] if bound_config else {})
        details=[]
        if isinstance(method,dict):
            for key in ('architecture','schema','role','dtype','width','horizons','encoder_widths','decoder_widths'):
                if key in method:
                    value=json.dumps(method[key],ensure_ascii=False,separators=(',',':'))
                    details.append(key+'='+value[:120]+('…' if len(value)>120 else ''))
        summary=('Recorded method: '+('; '.join(details) if details else 'description pending source review')+'\n\n'
            'Declared parameters: '+('not recorded' if fit['declared_parameter_count'] is None else str(fit['declared_parameter_count']))
            +'; optimizer updates: '+('not recorded' if fit['optimizer_updates'] is None else str(fit['optimizer_updates']))+'.\n\n')
        for name,item in list(pressure.items())[:2]:
            summary+=(name+' · same 128 pressure requests: worst mean-free pressure relative L2 '
                +format(item['worst']['pressure_mean_free_relative_l2'],'.6g')+', equation relative L2 '
                +format(item['worst']['common_equation_relative_l2'],'.6g')+'. Timing remains unqualified.\n\n')
        if len(pressure)>2:summary+=str(len(pressure)-2)+' further joined cohort results are retained below.\n\n'
        prefix=('# '+fit['name']+'\n\nEvidence draft · '+fit['item_id']+'\n\n'+summary+
            'Model, training attempt, checkpoint bytes and execution variant are separate identities. '
            'Method/purpose/lesson review and clean reproduction remain pending. '
            'Any historical timing in the original receipt remains historical.\n\n'
            '## Retained facts and comparison evidence\n\n```json\n').encode()+encoded(facts)+b'```\n\n## Exact original receipt bytes\n\n```json\n'
        payload=prefix+raw+b'\n```\n';config_metadata={}
        if config_raw is not None:
            payload+=b'\n## Exact hash-bound training configuration\n\n```json\n'
            config_metadata=dict(embedded_config_offset=len(payload),embedded_config_bytes=len(config_raw),embedded_config_sha256=sha(config_raw))
            payload+=config_raw+b'\n```\n'
        slug=re.sub('[^a-z0-9]+','-',fit['name'].lower()).strip('-')[:96] or 'fit'
        note=write('notes/fits/'+slug+'--'+fit['receipt_sha256'][:16]+'.md',payload,
            fit_item_id=fit['item_id'],embedded_receipt_offset=len(prefix),embedded_receipt_bytes=len(raw),embedded_receipt_sha256=sha(raw),**config_metadata)
        actual=(output/note['path']).read_bytes()[note['embedded_receipt_offset']:note['embedded_receipt_offset']+len(raw)]
        if actual!=raw:raise ValueError('Original receipt bytes changed during card authoring')
        fit_index.append(dict(item_id=fit['item_id'],kind=fit['kind'],name=fit['name'],task=fit['task'],
            card_path=note['path'],card_sha256=note['sha256'],receipt_sha256=fit['receipt_sha256'],
            method_fields_present=bool(fit['method']),Run_observation_count=len(fit['recorded_Run_observations']),
            verified_configuration_embedded=config_raw is not None,
            named_entries=fit['named_entries'],checkpoints=fit['checkpoints'],status='evidence_draft_review_required'))
    by_hash={row['sha256']:row for row in contexts['rows']}
    checkpoint_rows=[]
    for group in checkpoints['groups']:
        context=by_hash[group['sha256']]
        candidates=[context['retained_path'],*group['paths']]
        verified=None
        for relative in dict.fromkeys(candidates):
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT):continue
            if path.is_file() and path.stat().st_size==group['bytes'] and hash_file(path)==group['sha256']:
                verified=relative;break
        if verified is None:raise ValueError('Known checkpoint byte identity no longer has an exact retained file: '+group['sha256'])
        checkpoint_rows.append(dict(kind='checkpoint_byte_identity',sha256=group['sha256'],bytes=group['bytes'],
            verified_retained_path=verified,original_inventory=group,context=context,
            interpretation='Historical alias paths are inert evidence, not current availability or a model-count claim. No payload deserialized.'))
    checkpoint_parts=[];part=[]
    def checkpoint_partition(rows):
        value=dict(schema='rheon.portfolio-checkpoint-index-part.v1',sources={key:inputs[key] for key in ('checkpoint_inventory','checkpoint_contexts')},
                   interpretation='Byte identities and retained contexts; model identity and applicable benchmark review remain incomplete.',rows=rows)
        entry=write(f'notes/checkpoints/part-{len(checkpoint_parts)+1:02d}.json',encoded(value),checkpoint_rows=len(rows))
        checkpoint_parts.append(dict(path=entry['path'],sha256=entry['sha256'],rows=len(rows)))
    for row in checkpoint_rows:
        if part and len(encoded(part+[row]))>700*1024:checkpoint_partition(part);part=[]
        part.append(row)
    if part:checkpoint_partition(part)
    alias_index=[{key:entry.get(key) for key in ('item_id','kind','label','source_pointer','fit_item_id',
        'declared_weights_sha256','checkpoint_disposition','fit_link_disposition','candidate_fit_item_ids',
        'applicable_benchmark')} for entry in work['named_entries']]
    pressure={subject['identifier']:subject for subject in bundle['subjects']}
    for projected,original in zip(alias_index,work['named_entries'],strict=True):
        projected['source_worklist_fresh_benchmark_status']=original['fresh_benchmark_status']
        projected['retained_common_pressure']=pressure.get(original['label']) if original['item_id'].startswith('cohorts/') else None
    index=dict(schema='rheon.native-model-portfolio-index.v1',status='evidence_index_not_complete_model_accounting',
        sources=inputs,definitions=dict(fit='One retained training receipt identity; not necessarily a complete semantic model.',
            alias='One dashboard reference; aliases can share fits/bytes and preserve distinct execution methods.',
            checkpoint='One exact serialized byte identity; equal or different bytes do not prove semantic equivalence.',
            execution_variant='A distinct recorded inference procedure; a display label alone does not prove its actual work.'),
        fits=fit_index,aliases=alias_index,checkpoint_partitions=checkpoint_parts,
        source_candidates=work['source_candidates'],pressure_bundle=dict(source=inputs['pressure_bundle'],subjects=len(bundle['subjects']),
            available_capture_subjects=bundle['available_capture_subjects'],unlocated_checkpoint_subjects=bundle['unlocated_checkpoint_subjects'],
            actual_requests=bundle['actual_pressure_requests_retained'],measured_scalars=bundle['measured_scalar_values'],
            missing_observations=bundle['explicit_unavailable_observations'],packet_count=len(bundle['packets'])),
        limitations=['55 fit drafts lack a structured worklist method description; original receipts retain training/mode facts and architecture source review remains pending.',
            'Original current/archived Run history is not reconstructed by these cards.',
            'Complete intermediate/prototype/full-state benchmarks, current timings, clean reproduction and new-Rheon generation remain pending.'])
    write('notes/portfolio-index.json',encoded(index))
    overview=('# Rheon model portfolio · evidence migration\n\n'
        '205 training-receipt card drafts, 379 dashboard aliases and 524 checkpoint byte identities are separately indexed. '
        'These are not interchangeable model counts. 55 fit drafts lack a structured method description; original receipts still retain mode/training facts. '
        '51 exact training configurations are now recovered and embedded alongside their original receipts. '
        'Architecture source review remains pending. No automatic causal lessons or clean reproduction are claimed.\n\n'
        'Pressure results: all 129 named entries are accounted for. 127 have the same 128 original consumed pressure requests each; '
        'C01/C02 retain explicit unavailable checkpoint observations. Nine bounded scalar packets retain all 65,024 measured values and '
        '17,536 unavailable observations. Missing timings are not zero. Density and XYZ domain counts remain distinct. '
        'Execution variants and device/precision facts are retained per subject. These pressure proposals do not advance a complete simulation.\n\n'
        'Both previously quoted full-state anchors reproduced exactly under their actual work contracts. Their historical timings and '
        'single-case pressure values cannot be substituted for these common-pressure populations. Full-state/own-history drift and new timings remain separate work.\n\n'
        'Open notes/portfolio-index.json for card/alias/checkpoint links. Each card embeds exact original receipt bytes and source-linked facts. '
        'Captured historical paths and Run IDs grant no current filesystem/execution authority. No unique model weights are copied into these notes or retired. '
        'New Rheon has not been adopted; its different physics requires a separate reference epoch.\n\n'
        '## Fit cards\n\n'+''.join('- ['+row['name']+']('+row['card_path'].removeprefix('notes/')+')\n' for row in fit_index)).encode()
    write('notes/portfolio-overview.md',overview)
    report=dict(schema='rheon.native-portfolio-cards-preparation.v3',status='prepared_not_native_adopted',sources=inputs,
        producer_sha256=hash_file(Path(__file__)),
        fit_cards=205,aliases=379,checkpoint_byte_identities=524,checkpoint_files_rehashed=524,
        original_receipt_bytes_exact=205,original_receipt_total_bytes=sum(note.get('embedded_receipt_bytes',0) for note in notes),
        missing_structured_method_descriptions=sum(not fit['method'] for fit in work['fit_cards']),
        exact_training_configurations_embedded=sum('embedded_config_sha256' in note for note in notes),
        notes=notes,new_neural_forwards=0,new_scoring=False,payloads_deserialized=False,
        unique_weights_copied_or_deleted=False,full_portfolio_accounted=False)
    (output/'preparation.json').write_bytes(encoded(report))
    print(json.dumps({key:report[key] for key in ('status','fit_cards','aliases','checkpoint_byte_identities','original_receipt_bytes_exact','missing_structured_method_descriptions')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--method-configs',type=Path,required=True)
    args=parser.parse_args();prepare(args.output.resolve(),args.bundle.resolve(),args.method_configs.resolve())
