"""One native owner for the qualified twenty-component outcome note batch.

Use compact before/after custody, preserving every prior note revision and all
non-note records. An uncertain operation is reconciled from its saved request;
this tool never retries publication or creates a project/payload backup.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid

from import_common_generator_measurements import ROOT, RUN, DOC, PROJECT, CONTROLLER, header, payloads, read, digest, encode, binding, require, save
from adopt_matched_generator_outcome_notes_02 import target_records

PREPARED = RUN/'native-component-outcome-notes-01'
QUALIFICATION = DOC/'reports/component-outcome-notes-qualification-01.json'
ACCEPTED = DOC/'reports/native-pressure-publication-UI-01.json'
JOB = RUN/'native-component-outcome-adoption-01'
BEFORE = JOB/'before-custody.json'
STATUS = JOB/'status.json'
REQUEST = JOB/'adopt.request.json'
RESPONSE = JOB/'adopt.response.json'
REPORT = DOC/'reports/native-component-outcome-adoption-01.json'
SOURCE_COMMIT = '00a02ee25cb30d254c059a9b372036f9f6f36696'


def qualified():
    prep, proof = read(PREPARED/'preparation.json'), read(QUALIFICATION)
    require(proof['status'] == 'component23note_proposal_native_codec_original_tails_and_physical_leads_qualified'
            and proof['notes'] == len(prep['notes']) == 23 and proof['component_fits'] == prep['component_fits'] == 20
            and proof['models'] == 205 and proof['literal_summary_paragraphs'] == 1230
            and proof['physical_lead_metric_rows'] == 540 and proof['original_tails_preserved'] == 20,
            'Complete exact component note qualification required')
    for spec in [proof['producer'], proof['preparation'], proof['native_UI_codec'], *proof['evidence'], prep['producer'], *prep['evidence']]:
        require(binding(ROOT/spec['path']) == spec, 'Qualified proposal or source changed')
    require(len({n['path'] for n in prep['notes']}) == 23 and len(prep['notes']) <= proof['native_max_items']
            and sum(n['bytes'] for n in prep['notes']) <= proof['native_max_bytes'], 'Native batch bounds differ')
    return prep, proof


def before():
    require(not JOB.exists() and not REPORT.exists(), 'Fresh component note owner required')
    prep, _ = qualified(); accepted = read(ACCEPTED); raw, snapshot = header()
    require(snapshot['checkpoint'] == prep['source_checkpoint_at_read'] == accepted['checkpoint']
            and digest(raw) == accepted['canonical_header_sha256'] and accepted['guide_revision'] == 9,
            'Current accepted publication changed')
    current = target_records(snapshot); targets = []
    for note in prep['notes']:
        entry, doc = current[note['path']]
        require(entry['id'] == note['file_id'] and doc['current']['revision'] == note['expected_revision']
                and doc['current']['sha256'] == note['previous_sha256']
                and (PROJECT/note['path']).read_bytes() == doc['current']['text'].encode(), 'Target note changed')
        targets.append(dict(entry=entry, note_path=note['path']))
    fingerprint = payloads()
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw, 'Project changed during before-custody')
    JOB.mkdir()
    save(BEFORE, dict(status='compact_component_note_before_custody_captured', created_unix=time.time(),
        project_id=snapshot['project_id'], checkpoint=snapshot['checkpoint'], header_sha256=digest(raw),
        canonical_records=len(snapshot['inventory']), old_target_entries=targets, payloads=fingerprint,
        preparation=binding(PREPARED/'preparation.json'), accepted_publication=binding(ACCEPTED),
        no_full_header_project_or_payload_backup=True))
    print('Captured compact23note before-custody;no working notes changed.', flush=True)


def adopt():
    require(BEFORE.exists() and not STATUS.exists() and not REQUEST.exists(), 'Original before-custody required; reconcile, never repeat publication')
    prep, proof = qualified(); previous = read(BEFORE); raw, snapshot = header()
    require(snapshot['checkpoint'] == previous['checkpoint'] and digest(raw) == previous['header_sha256'], 'Canonical checkpoint changed')
    controller = read(CONTROLLER/'controller.json'); source = Path(controller['source'])
    require(controller['status'] == 'ready' and controller['source_commit'] == SOURCE_COMMIT
            and all(Path(f'/proc/{controller[k]}').exists() for k in ('launcher_pid','app_pid'))
            and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == SOURCE_COMMIT
            and not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
            and binding(source/'static/project-model-portfolio.js') == proof['native_UI_codec'], 'Qualified clean live source required')
    state = dict(status='starting_original_component_note_owner', pid=os.getpid(), created_unix=time.time(),
        project_id=snapshot['project_id'], previous_checkpoint=snapshot['checkpoint'],
        preparation=binding(PREPARED/'preparation.json'), new_model_forwards=0, unique_weights_deleted=0,
        payload_copies=0, Trial_lineage_modified=False, automatic_retries=False, full_portfolio_complete=False)
    save(STATUS, state); sys.path.insert(0,str(source)); import agent_api
    client = agent_api.ProjectFilesOwnerClient(controller['origin'], (CONTROLLER/'private/owner-session.txt').read_text().strip(),
                                               controller['catalog_id'], timeout=7200)
    try:
        client.bootstrap(); response = client.call('project_files.status', {}); save(JOB/'before-status.response.json', response)
        require(response['status'] == 200, 'Native pre-adoption status rejected')
        selected = response['result']
        require(selected['state'] == 'open' and selected['project']['project_id'] == snapshot['project_id']
                and selected['project']['checkpoint'] == snapshot['checkpoint'], 'Actual native selection changed')
        current = target_records(snapshot); pending = []
        for note in prep['notes']:
            relative = Path(note['path']); target, proposed = PROJECT/relative, ROOT/note['prepared_path']
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'notes'
                    and proposed.is_relative_to(PREPARED)
                    and not any(p.is_symlink() for p in (target,*target.parents,proposed)), 'Authored path/alias boundary differs')
            body, old = proposed.read_bytes(), target.read_bytes(); doc = current[note['path']][1]
            require(len(body) == note['bytes'] and digest(body) == note['sha256'] and digest(old) == note['previous_sha256']
                    and old == doc['current']['text'].encode() and doc['current']['revision'] == note['expected_revision'], 'Prepared/working/canonical note changed')
            pending.append((note,target,body,stat.S_IMODE(target.stat().st_mode)))
        require((PROJECT/'.tuldok/project.json').read_bytes() == raw, 'Canonical state changed before staging')
        state.update(status='staging_guarded_component_note_revisions', notes=23, updated_unix=time.time()); save(STATUS,state)
        for note,target,body,mode in pending:
            require(digest(target.read_bytes()) == note['previous_sha256'], 'Working note changed before replacement')
            temporary = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.pending')
            descriptor = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
            with os.fdopen(descriptor,'wb') as stream:
                stream.write(body); stream.flush(); os.fchmod(stream.fileno(),mode); os.fsync(stream.fileno())
            temporary.replace(target)
        arguments = dict(binding=selected['binding'], expected_checkpoint=snapshot['checkpoint'], request_key=uuid.uuid4().hex,
            items=[dict(kind='note',path=n['path'],expected_revision=n['expected_revision'],expected_sha256=n['sha256']) for n in prep['notes']])
        save(REQUEST, dict(operation='project_files.files_adopt', arguments=arguments, origin=controller['origin'], controller_catalog_id=controller['catalog_id']))
        state.update(status='original_component23note_adoption_in_flight', request=binding(REQUEST), updated_unix=time.time()); save(STATUS,state)
        print('Dispatched original23note adoption once.',flush=True); tick = time.monotonic()
        response = client.call('project_files.files_adopt',arguments); save(RESPONSE,response)
        require(response['status'] == 200, 'Native note outcome requires reconciliation; no automatic retry')
        envelope, value = response['result'], response['result']['value']
        require(envelope['binding'] == selected['binding'] and envelope['project_id'] == snapshot['project_id']
                and value['changed_count'] == len(value['items']) == 23, 'Native note acknowledgement membership differs')
        for actual,expected in zip(value['items'],prep['notes'],strict=True):
            require(all(actual[k] == expected[k] for k in ('path','sha256','bytes')) and actual['revision'] == expected['revision'], 'Native note differs from proposal')
        state.update(status='native_component23note_adoption_acknowledged', checkpoint=envelope['checkpoint'], adopted_notes=23,
            native_operation_wall_seconds=time.monotonic()-tick, project_left_open=True, finished_unix=time.time())
    except BaseException as error:
        state.update(status='original_component_note_outcome_requires_reconciliation', error_type=type(error).__name__, error=str(error), updated_unix=time.time())
        raise
    finally:
        save(STATUS,state)


def after():
    require(not REPORT.exists(), 'Preserve original note custody')
    prep, _ = qualified(); previous, state, response = read(BEFORE), read(STATUS), read(RESPONSE)
    require(state['status'] == 'native_component23note_adoption_acknowledged'
            and not Path(f'/proc/{state["pid"]}').exists() and response['status'] == 200, 'Original note owner must be terminal with ACK')
    raw, snapshot = header(); arguments = read(REQUEST)['arguments']
    require(snapshot['checkpoint'] == state['checkpoint'] == response['result']['checkpoint']
            and arguments['expected_checkpoint'] == previous['checkpoint']
            and binding(PREPARED/'preparation.json') == previous['preparation'], 'Publication authority differs')
    operation = ('project_operation',snapshot['project_id']+'.'+digest(arguments['request_key'].encode()))
    indexed = {(e['family'],e['id']):e for e in snapshot['inventory']}
    require(len(indexed) == len(snapshot['inventory']) == previous['canonical_records']+1 and operation in indexed, 'Only one operation record may be added')
    total = 0
    for entry in snapshot['inventory']:
        path = PROJECT/'.tuldok'/entry['path']; info, body = path.lstat(), path.read_bytes()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and len(body) == entry['bytes'] and digest(body) == entry['sha256'], 'Canonical body differs')
        total += len(body)
    old_entries = {(r['entry']['family'],r['entry']['id']):r for r in previous['old_target_entries']}; expected = {n['path']:n for n in prep['notes']}
    for key,old in old_entries.items():
        doc = read(PROJECT/'.tuldok'/indexed[key]['path']); note = expected[old['note_path']]; body = (ROOT/note['prepared_path']).read_bytes()
        require(doc['path'] == note['path'] and doc['current'] == doc['history'][-1]
                and doc['current']['revision'] == note['revision'] and doc['current']['sha256'] == note['sha256'] == digest(body)
                and doc['current']['text'].encode() == body == (PROJECT/note['path']).read_bytes(), 'Saved note differs from proposal')
        restored = dict(doc,current=doc['history'][-2],history=doc['history'][:-1])
        require(len(encode(restored)) == old['entry']['bytes'] and digest(encode(restored)) == old['entry']['sha256'], 'Prior complete note history changed')
    old = dict(snapshot,checkpoint=previous['checkpoint'])
    old['inventory'] = [old_entries[key]['entry'] if key in old_entries else e for e in snapshot['inventory'] if (key:=(e['family'],e['id'])) != operation]
    old['roots'] = [e for e in snapshot['roots'] if (e['family'],e['id']) != operation]
    old['receipts'] = [r for r in snapshot['receipts'] if r['request_key'] != arguments['request_key']]
    fingerprint = payloads()
    require(len(old['receipts'])+1 == len(snapshot['receipts']) and digest(encode(old)) == previous['header_sha256']
            and fingerprint == previous['payloads'], 'Prior scientific records, histories or payload identities changed')
    require((PROJECT/'.tuldok/project.json').read_bytes() == raw, 'Project changed during custody')
    save(REPORT, dict(status='native23component_outcome_notes_exact_and_scientific_records_unchanged', created_unix=time.time(),
        checkpoint=snapshot['checkpoint'], canonical_header_sha256=digest(raw), canonical_records=len(indexed), canonical_record_bytes=total,
        changed_note_records=23, component_fit_cards=20, total_models=205, guide_revision=prep['guide_revision'],
        all_canonical_bodies_freshly_hashed=True, original_note_histories_exact=True, all_non_note_scientific_records_exact=True,
        payload_preservation=fingerprint, payload_bodies_rehashed=False, actual_UI_pending=True, project_left_open=True,
        before=binding(BEFORE), original_owner=binding(STATUS), producer=binding(Path(__file__)),
        native_chart_publication_pending=True, portable_scalar_source_closure_complete=False, full_portfolio_complete=False))
    print('All23component notes and prior scientific custody exact;actual UI pending.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',choices=('before','adopt','after')); args = parser.parse_args()
    {'before':before,'adopt':adopt,'after':after}[args.phase]()
