"""Reconcile the interrupted note request against its unchanged canonical source.

Require a later host boot and absent original owners, rehash every canonical
record, then undo only our exact proposed working-note bytes. Never replay the
mutation, modify canonical history or touch model/reference/prediction fields.
"""
import os
from pathlib import Path
import stat
import time
import uuid

from import_common_generator_measurements import ROOT, RUN, DOC, PROJECT, CONTROLLER, header, payloads, read, save, binding, digest, require

JOB = RUN/'native-component-outcome-adoption-01'
REPORT = DOC/'reports/component-note-reboot-reconciliation-01.json'


def run():
    require(not REPORT.exists(), 'Preserve original reboot reconciliation')
    before, owner, controller = read(JOB/'before-custody.json'), read(JOB/'status.json'), read(CONTROLLER/'controller.json')
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    pids = [owner['pid'],controller['launcher_pid'],controller['app_pid']]
    require(owner['status']=='original_component23note_adoption_in_flight' and owner['created_unix']<boot
            and controller['ready_unix']<boot and not any(Path('/proc',str(pid)).exists() for pid in pids)
            and not (JOB/'adopt.response.json').exists(), 'A later boot and absent original owners required')
    raw, snapshot = header(); request = read(JOB/'adopt.request.json'); key = request['arguments']['request_key']
    require(digest(raw)==before['header_sha256'] and snapshot['checkpoint']==before['checkpoint']
            and len(snapshot['inventory'])==before['canonical_records']
            and not any(r.get('request_key')==key for r in snapshot['receipts']), 'Canonical outcome differs; explicit native recovery required')
    total = 0
    for entry in snapshot['inventory']:
        path = PROJECT/'.tuldok'/entry['path']; info = path.lstat(); body = path.read_bytes()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and len(body)==entry['bytes'] and digest(body)==entry['sha256'], 'Canonical body differs after reboot')
        total += len(body)
    fingerprint = payloads(); require(fingerprint==before['payloads'], 'Payload identity changed')
    prep_path = RUN/'native-component-outcome-notes-01/preparation.json'
    require(binding(prep_path)==before['preparation'], 'Original prepared notes changed')
    notes = {n['path']:n for n in read(prep_path)['notes']}; plans = []
    for item in before['old_target_entries']:
        doc = read(PROJECT/'.tuldok'/item['entry']['path']); name = item['note_path']; note = notes[name]
        require(binding(PROJECT/'.tuldok'/item['entry']['path'])['sha256']==item['entry']['sha256'], 'Original note history changed')
        old = doc['current']['text'].encode(); target = PROJECT/name; current = target.read_bytes(); current_hash = digest(current)
        require(digest(old)==note['previous_sha256'] and current_hash in (note['previous_sha256'],note['sha256'])
                and not target.is_symlink(), 'Unexpected working-note edit; preserve it')
        plans.append((target,old,current_hash==note['sha256'],stat.S_IMODE(target.stat().st_mode)))
    require((PROJECT/'.tuldok/project.json').read_bytes()==raw, 'Canonical project changed during reconciliation')
    restored = []
    for target,old,changed,mode in plans:
        if not changed:
            continue
        temporary = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.pending')
        descriptor = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(old);stream.flush();os.fchmod(stream.fileno(),mode);os.fsync(stream.fileno())
        temporary.replace(target);restored.append(str(target.relative_to(PROJECT)))
    require((PROJECT/'.tuldok/project.json').read_bytes()==raw and payloads()==fingerprint, 'Canonical/payload state changed')
    save(REPORT,dict(status='interrupted_note_request_reconciled_canonical_effect_absent_after_host_reboot',created_unix=time.time(),
        host_boot_unix=boot,host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        original_processes_absent=pids,original_request=binding(JOB/'adopt.request.json'),original_owner=binding(JOB/'status.json'),
        before_custody=binding(JOB/'before-custody.json'),checkpoint=snapshot['checkpoint'],canonical_header_sha256=digest(raw),
        canonical_records_verified=len(snapshot['inventory']),canonical_record_bytes_verified=total,original_request_receipt_present=False,
        payload_preservation=fingerprint,payload_bodies_rehashed=False,working_notes_restored_to_canonical=restored,
        prepared_proposal_preserved=binding(prep_path),native_mutation_replayed=False,native_restart_and_Open_pending=True,
        SQL_projection_outcome_not_inferred_from_filesystem=True,producer=binding(Path(__file__)),
        project_or_payload_copies=0,prediction_files_deleted=0,new_model_forwards=0,full_portfolio_complete=False))
    print('Reconciled host reboot:',len(snapshot['inventory']),'canonical records exact;',len(restored),'staged working notes restored;no mutation replay.',flush=True)


if __name__=='__main__':
    run()
