"""Publish the26qualified pressure notes once, with compact exact custody."""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,CONTROLLER,header,payloads,read,digest,encode,binding,require,save
from adopt_matched_generator_outcome_notes_02 import target_records

PREPARED = RUN/'native-pressure-outcome-notes-01'
QUALIFICATION = DOC/'reports/pressure-outcome-notes-qualification-01.json'
IMPORT_CUSTODY = DOC/'reports/native-all150-pressure-import-custody-01.json'
JOB = RUN/'native-pressure-outcome-adoption-01'
BEFORE = JOB/'before-custody.json'
STATUS = JOB/'status.json'
REQUEST = JOB/'adopt.request.json'
RESPONSE = JOB/'adopt.response.json'
REPORT = DOC/'reports/native-pressure-outcome-adoption-01.json'


def qualified():
    prep,proof = read(PREPARED/'preparation.json'),read(QUALIFICATION)
    require(proof['status'] == 'pressure26outcome_note_proposal_actual_UI_codec_and_original_tails_qualified'
            and proof['notes'] == len(prep['notes']) == 26 and proof['pressure_fits'] == 23 and proof['models'] == 205
            and proof['native_batch_items_bound_passed'] and proof['native_max_items'] == 40
            and proof['literal_summary_paragraphs'] == 1230 and proof['authored_comparisons'] == 26,
            'Complete26note native codec/source qualification required')
    for spec in [proof['producer'],proof['preparation'],proof['native_UI_codec'],*proof['evidence'],prep['producer'],*prep['evidence']]:
        require(binding(ROOT/spec['path']) == spec,'Qualified note preparation/source changed')
    require(len({note['path'] for note in prep['notes']}) == 26
            and sum(note['bytes'] for note in prep['notes']) <= proof['native_max_bytes'],'Distinct note/byte batch bounds required')
    return prep,proof


def before():
    require(not JOB.exists() and not REPORT.exists(),'Fresh original pressure-note owner required')
    prep,_ = qualified()
    custody = read(IMPORT_CUSTODY)
    import_owner = RUN/'native-all150-pressure-import-01/status.json'
    require(custody['status'] == 'all150_pressure_tables_native_custody_exact'
            and binding(import_owner) == custody['original_owner'],'Accepted original150pressure import custody required')
    owner = read(import_owner)
    require(owner['status'] == 'all150_pressure_imports_native_metadata_and_maxima_passed'
            and owner['active_request'] is None and not Path(f'/proc/{owner["pid"]}').exists(),
            'Original pressure import must be terminal before note adoption')
    raw,snapshot = header()
    require(snapshot['checkpoint'] == custody['checkpoint'] == owner['last_acknowledged_checkpoint']
            and digest(raw) == custody['canonical_header_sha256'] and payloads() == custody['payloads'],
            'Current project differs from accepted pressure import custody')
    current = target_records(snapshot)
    targets = []
    for note in prep['notes']:
        entry,doc = current[note['path']]
        require(entry['id'] == note['file_id'] and doc['current']['revision'] == note['expected_revision']
                and doc['current']['sha256'] == note['previous_sha256']
                and (PROJECT/note['path']).read_bytes() == doc['current']['text'].encode(),'A target note changed after qualification')
        targets.append(dict(entry=entry,note_path=note['path']))
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Project changed during before-custody')
    JOB.mkdir()
    save(BEFORE,dict(status='compact_pressure_note_before_custody_captured',created_unix=time.time(),project_id=snapshot['project_id'],
        checkpoint=snapshot['checkpoint'],header_sha256=digest(raw),canonical_records=len(snapshot['inventory']),
        old_target_entries=targets,payloads=payloads(),preparation=binding(PREPARED/'preparation.json'),
        pressure_table_custody=binding(IMPORT_CUSTODY),no_full_header_project_or_payload_backup=True))
    print('Compact26note before-custody captured; no working notes changed.',flush=True)


def adopt():
    require(BEFORE.exists() and not STATUS.exists() and not REQUEST.exists(),'Original before-custody required; never replay note adoption')
    prep,proof = qualified()
    previous = read(BEFORE)
    raw,snapshot = header()
    require(snapshot['checkpoint'] == previous['checkpoint'] and digest(raw) == previous['header_sha256'],
            'Canonical checkpoint changed before note adoption')
    controller = read(CONTROLLER/'controller.json')
    source = Path(controller['source'])
    require(controller['status'] == 'ready' and controller['source_commit'] == 'f8ba4cedd22580f273f8c4fcbc06bca53797a7a7'
            and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == controller['source_commit']
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'static/project-model-portfolio.js') == proof['native_UI_codec'],'Exact qualified clean live source required')
    state = dict(status='starting_original_pressure_note_owner',pid=os.getpid(),created_unix=time.time(),project_id=snapshot['project_id'],
        previous_checkpoint=snapshot['checkpoint'],preparation=binding(PREPARED/'preparation.json'),new_model_forwards=0,
        unique_weights_deleted=0,payload_copies=0,Trial_lineage_modified=False,automatic_retries=False,full_portfolio_complete=False)
    save(STATUS,state)
    sys.path.insert(0,str(source))
    import agent_api
    client = agent_api.ProjectFilesOwnerClient(controller['origin'],(CONTROLLER/'private/owner-session.txt').read_text().strip(),
        controller['catalog_id'],timeout=7200)
    try:
        client.bootstrap()
        response = client.call('project_files.status',{})
        save(JOB/'before-status.response.json',response)
        require(response['status'] == 200,'Native pre-adoption status rejected')
        selected = response['result']
        require(selected['state'] == 'open' and selected['project']['project_id'] == snapshot['project_id']
                and selected['project']['checkpoint'] == snapshot['checkpoint'],'Actual native selection changed')
        current = target_records(snapshot)
        pending = []
        for note in prep['notes']:
            relative = Path(note['path'])
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'notes','Authored path escapes notes')
            target,proposed = PROJECT/relative,ROOT/note['prepared_path']
            require(proposed.is_relative_to(PREPARED) and not any(path.is_symlink() for path in (target,*target.parents,proposed)),
                    'Authored source/target aliasing forbidden')
            body,old = proposed.read_bytes(),target.read_bytes()
            doc = current[note['path']][1]
            require(len(body) == note['bytes'] and digest(body) == note['sha256']
                    and digest(old) == note['previous_sha256'] and old == doc['current']['text'].encode()
                    and doc['current']['revision'] == note['expected_revision'],'Prepared,working or canonical note changed')
            pending.append((note,target,body,stat.S_IMODE(target.stat().st_mode)))
        require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Canonical state changed before staging')
        state.update(status='staging_guarded_pressure_note_revisions',notes=26,updated_unix=time.time())
        save(STATUS,state)
        for note,target,body,mode in pending:
            require(digest(target.read_bytes()) == note['previous_sha256'],'Working note changed before guarded replacement')
            temporary = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.pending')
            descriptor = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
            with os.fdopen(descriptor,'wb') as stream:
                stream.write(body);stream.flush();os.fchmod(stream.fileno(),mode);os.fsync(stream.fileno())
            temporary.replace(target)
        arguments = dict(binding=selected['binding'],expected_checkpoint=snapshot['checkpoint'],request_key=uuid.uuid4().hex,
            items=[dict(kind='note',path=note['path'],expected_revision=note['expected_revision'],expected_sha256=note['sha256']) for note in prep['notes']])
        save(REQUEST,dict(operation='project_files.files_adopt',arguments=arguments,origin=controller['origin'],controller_catalog_id=controller['catalog_id']))
        state.update(status='original_pressure26note_adoption_in_flight',request=binding(REQUEST),updated_unix=time.time())
        save(STATUS,state)
        print('Dispatched original26note adoption once.',flush=True)
        tick = time.monotonic()
        response = client.call('project_files.files_adopt',arguments)
        save(RESPONSE,response)
        require(response['status'] == 200,'Native note outcome requires reconciliation; no automatic retry')
        envelope,value = response['result'],response['result']['value']
        require(envelope['binding'] == selected['binding'] and envelope['project_id'] == snapshot['project_id']
                and value['changed_count'] == len(value['items']) == 26,'Native note acknowledgement membership differs')
        for actual,expected in zip(value['items'],prep['notes'],strict=True):
            require(all(actual[key] == expected[key] for key in ('path','sha256','bytes')) and actual['revision'] == expected['revision'],
                    'Adopted note differs from exact proposal')
        state.update(status='native_pressure26note_adoption_acknowledged',checkpoint=envelope['checkpoint'],adopted_notes=26,
            native_operation_wall_seconds=time.monotonic()-tick,project_left_open=True,finished_unix=time.time())
    except BaseException as error:
        state.update(status='original_pressure_note_outcome_requires_reconciliation',error_type=type(error).__name__,error=str(error),updated_unix=time.time())
        raise
    finally:
        save(STATUS,state)


def after():
    require(not REPORT.exists(),'Preserve original pressure-note custody')
    prep,_ = qualified()
    previous,state,response = read(BEFORE),read(STATUS),read(RESPONSE)
    require(state['status'] == 'native_pressure26note_adoption_acknowledged'
            and not Path(f'/proc/{state["pid"]}').exists() and response['status'] == 200,'Original note owner must be terminal with ACK')
    raw,snapshot = header()
    arguments = read(REQUEST)['arguments']
    require(snapshot['checkpoint'] == state['checkpoint'] == response['result']['checkpoint']
            and arguments['expected_checkpoint'] == previous['checkpoint']
            and binding(PREPARED/'preparation.json') == previous['preparation'],'Pressure note publication authority differs')
    operation = ('project_operation',snapshot['project_id']+'.'+digest(arguments['request_key'].encode()))
    indexed = {(entry['family'],entry['id']):entry for entry in snapshot['inventory']}
    require(len(indexed) == len(snapshot['inventory']) == previous['canonical_records']+1 and operation in indexed,
            'Only one note-operation record may be added')
    total = 0
    for entry in snapshot['inventory']:
        path = PROJECT/'.tuldok'/entry['path']
        info,body = path.lstat(),path.read_bytes()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and len(body) == entry['bytes'] and digest(body) == entry['sha256'],
                'Current canonical body differs')
        total += len(body)
    old_entries = {(row['entry']['family'],row['entry']['id']):row for row in previous['old_target_entries']}
    expected = {note['path']:note for note in prep['notes']}
    for key,old in old_entries.items():
        doc = read(PROJECT/'.tuldok'/indexed[key]['path'])
        note = expected[old['note_path']]
        body = (ROOT/note['prepared_path']).read_bytes()
        require(doc['path'] == note['path'] and doc['current'] == doc['history'][-1]
                and doc['current']['revision'] == note['revision'] and doc['current']['sha256'] == note['sha256'] == digest(body)
                and doc['current']['text'].encode() == body == (PROJECT/note['path']).read_bytes(),'Saved pressure note differs from proposal')
        restored = dict(doc,current=doc['history'][-2],history=doc['history'][:-1])
        require(len(encode(restored)) == old['entry']['bytes'] and digest(encode(restored)) == old['entry']['sha256'],
                'Original complete note history changed')
    old = dict(snapshot,checkpoint=previous['checkpoint'])
    old['inventory'] = [old_entries[key]['entry'] if key in old_entries else entry for entry in snapshot['inventory']
        if (key:=(entry['family'],entry['id'])) != operation]
    old['roots'] = [entry for entry in snapshot['roots'] if (entry['family'],entry['id']) != operation]
    old['receipts'] = [row for row in snapshot['receipts'] if row['request_key'] != arguments['request_key']]
    require(len(old['receipts'])+1 == len(snapshot['receipts']) and digest(encode(old)) == previous['header_sha256']
            and payloads() == previous['payloads'],'Prior scientific records,note histories or payload identities changed')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw,'Project changed during custody')
    save(REPORT,dict(status='native26pressure_outcome_notes_exact_and_scientific_records_unchanged',created_unix=time.time(),
        checkpoint=snapshot['checkpoint'],canonical_header_sha256=digest(raw),canonical_records=len(indexed),canonical_record_bytes=total,
        changed_note_records=26,pressure_fit_cards=23,total_models=205,guide_revision=prep['guide_revision'],
        all_canonical_bodies_freshly_hashed=True,original_note_histories_exact=True,all_non_note_scientific_records_exact=True,
        payload_preservation=payloads(),payload_bodies_rehashed=False,actual_UI_pending=True,project_left_open=True,
        before=binding(BEFORE),original_owner=binding(STATUS),producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('All26pressure outcome notes and prior scientific custody exact;actual UI pending.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('before','adopt','after'))
    args = parser.parse_args()
    {'before':before,'adopt':adopt,'after':after}[args.phase]()
