"""Remove only verified duplicate payloads from the fixed retired rehearsal04.

Keep its distinct SQL snapshot, receipt and coordinator journal. Reconstruct
ordinary independent files before using the compacted snapshot for restoration.
The retained rehearsal03 backup has already passed actual native restoration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from qualify_native_workspace import ROOT
from compact_backups import digest, payload, write_new

BASE=ROOT/'runs/model-portfolio-migration-v1'
DOCS=ROOT/'docs/plans/model-portfolio-migration-v1/reports'
OLD=BASE/'isolated-history-metadata-restore-01/private/extraction-exchange-native-history-extraction-rehearsal-04/stopped-copy'
KEEPER=BASE/'native-backup-compaction-03/private/backup'
PLAN=BASE/'discovery/rehearsal04-payload-compaction-01.json'
MARKER=OLD/'COMPACTED-BACKUP.json'


def verified(root,receipt):
    memo={}
    for relative,row in receipt['files'].items():
        path=payload(root,relative);info=path.lstat()
        if row['kind']=='directory':
            if not stat.S_ISDIR(info.st_mode):raise ValueError('Backup directory differs')
            continue
        if row['kind']!='file' or not stat.S_ISREG(info.st_mode) or info.st_size!=row['bytes']:
            raise ValueError('Backup file kind or size differs')
        key=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        if key not in memo:memo[key]=digest(path)
        if memo[key]!=row['sha256']:raise ValueError('Backup payload SHA differs')
    if digest(root/'metadata.json')!=receipt['metadata_sha256']:
        raise ValueError('Original SQL snapshot bytes differ')


def compact(report):
    if report.exists() or PLAN.exists() or MARKER.exists() or not report.is_relative_to(DOCS):
        raise ValueError('Fresh fixed-scope compaction evidence required')
    terminal=json.loads((DOCS/'native-history-extraction-rehearsal-04.json').read_bytes())
    proof_path=DOCS/'native-backup-compaction-03.json';proof=json.loads(proof_path.read_bytes())
    if (not terminal['postgres_stopped'] or not terminal['project_extracted']
            or proof['status']!='qualified_compact_backup_and_native_restore'
            or not all(proof[k] for k in ('native_restore_verified','compact_copy_verified','postgres_stopped','temporary_restore_removed'))):
        raise ValueError('Retired source and completed keeper restore qualification required')
    original=json.loads((OLD/'backup-receipt.json').read_bytes())
    keeper=json.loads((KEEPER/'backup-receipt.json').read_bytes())
    old_receipt=digest(OLD/'backup-receipt.json');keep_receipt=digest(KEEPER/'backup-receipt.json')
    if keep_receipt!=proof['source_receipt_sha256']:raise ValueError('Qualified keeper receipt changed')
    os.umask(0o077)
    result=dict(schema='rheon.rehearsal04-payload-compaction-result.v1',status='verifying_all_original_and_retained_bytes',
                created_unix=time.time(),original=str(OLD.relative_to(ROOT)),keeper=str(KEEPER.relative_to(ROOT)),
                metadata_deleted=False,unique_payloads_deleted=0,original_service_changed=False,
                available_bytes_before=shutil.disk_usage(ROOT).free)
    def save():
        pending=report.with_suffix('.pending');pending.write_text(json.dumps(result,indent=2)+'\n');pending.replace(report)
    save();print(result['status'],flush=True)
    verified(OLD,original);verified(KEEPER,keeper)
    duplicates=[];unique=[]
    for relative,row in original['files'].items():
        if row['kind']!='file':continue
        path=payload(OLD,relative);kept=keeper['files'].get(relative)
        mode=stat.S_IMODE(path.stat().st_mode)
        if kept==row and stat.S_IMODE(payload(KEEPER,relative).stat().st_mode)==mode:
            duplicates.append(dict(path=relative,**row,mode=mode))
        else:unique.append(dict(path=relative,**row,mode=mode))
    plan=dict(schema='rheon.rehearsal04-payload-compaction.v1',created_unix=time.time(),
              authorization='User authorized removal of identical backup files; reiterated disk-space concern.',
              original=str(OLD.relative_to(ROOT)),keeper=str(KEEPER.relative_to(ROOT)),
              original_receipt_sha256=old_receipt,keeper_receipt_sha256=keep_receipt,
              original_metadata_sha256=original['metadata_sha256'],
              keeper_native_restore_proof=str(proof_path.relative_to(ROOT)),
              keeper_native_restore_proof_sha256=digest(proof_path),duplicates=duplicates,unique=unique)
    write_new(PLAN,plan)
    marker=dict(schema=plan['schema'],status='compacting_duplicate_payloads',
                manifest=str(PLAN.relative_to(ROOT)),manifest_sha256=digest(PLAN),
                metadata_and_receipt_retained=True,standalone_restore_input=False,
                materialize_with='experiments/model-portfolio-migration-v1/compact_rehearsal04_backup.py materialize')
    write_new(MARKER,marker)
    result.update(status='removing_verified_duplicate_payloads',duplicate_files=len(duplicates),
                  duplicate_logical_bytes=sum(r['bytes'] for r in duplicates),
                  unique_payloads_retained=len(unique),plan=str(PLAN.relative_to(ROOT)),plan_sha256=digest(PLAN))
    save();print(result['status'],flush=True)
    for row in duplicates:
        old=payload(OLD,row['path']);kept=payload(KEEPER,row['path'])
        if (old.stat().st_size!=row['bytes'] or kept.stat().st_size!=row['bytes']
                or digest(old)!=row['sha256'] or digest(kept)!=row['sha256']):
            raise ValueError('Duplicate payload changed before unlink; retain remaining originals')
        old.unlink()
    verified(KEEPER,keeper)
    for row in unique:
        if digest(payload(OLD,row['path']))!=row['sha256']:raise ValueError('Unique original payload changed')
    if digest(OLD/'metadata.json')!=original['metadata_sha256'] or digest(OLD/'backup-receipt.json')!=old_receipt:
        raise ValueError('Distinct original metadata or receipt changed')
    marker['status']='compacted_requires_materialization'
    pending=MARKER.with_suffix('.pending');pending.write_text(json.dumps(marker,indent=2)+'\n');pending.replace(MARKER)
    result.update(status='completed_verified_duplicate_payload_compaction',
                  available_bytes_after=shutil.disk_usage(ROOT).free,completed_unix=time.time(),
                  original_metadata_and_receipt_exact=True,keeper_still_verified=True)
    save();print(json.dumps(result),flush=True)


def materialize(destination):
    if (destination.exists() or destination.is_relative_to(OLD) or destination.is_relative_to(KEEPER)):
        raise ValueError('Fresh destination outside retained backup roots required')
    os.umask(0o077)
    marker=json.loads(MARKER.read_bytes());plan=json.loads(PLAN.read_bytes())
    if (marker['status']!='compacted_requires_materialization' or digest(PLAN)!=marker['manifest_sha256']
            or plan['original']!=str(OLD.relative_to(ROOT)) or plan['keeper']!=str(KEEPER.relative_to(ROOT))
            or digest(OLD/'backup-receipt.json')!=plan['original_receipt_sha256']
            or digest(KEEPER/'backup-receipt.json')!=plan['keeper_receipt_sha256']):
        raise ValueError('Fixed compaction plan or source identities differ')
    receipt=json.loads((OLD/'backup-receipt.json').read_bytes())
    if shutil.disk_usage(destination.parent).free<sum(r.get('bytes',0) for r in receipt['files'].values())+1024**3:
        raise ValueError('Materialization space unavailable')
    destination.mkdir(mode=0o700)
    (destination/'catalog').mkdir(mode=0o700)
    (destination/'catalog').chmod(stat.S_IMODE((OLD/'catalog').stat().st_mode))
    duplicate_names={r['path'] for r in plan['duplicates']}
    for relative,row in sorted(receipt['files'].items(),key=lambda p:(p[0].count('/'),p[0])):
        target=payload(destination,relative)
        if row['kind']=='directory':
            target.mkdir(mode=0o700)
            target.chmod(stat.S_IMODE(payload(OLD,relative).stat().st_mode));continue
        source=payload(KEEPER if relative in duplicate_names else OLD,relative)
        if digest(source)!=row['sha256']:raise ValueError('Materialization source changed')
        shutil.copyfile(source,target);target.chmod(stat.S_IMODE(source.stat().st_mode))
    for name in ('metadata.json','backup-receipt.json'):
        shutil.copyfile(OLD/name,destination/name)
        (destination/name).chmod(stat.S_IMODE((OLD/name).stat().st_mode))
    verified(destination,receipt)
    print(json.dumps(dict(status='materialized_exact_original_rehearsal04',destination=str(destination))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('compact');p.add_argument('--report',type=Path,required=True)
    p=sub.add_parser('materialize');p.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='compact':compact(args.report.resolve())
    else:materialize(args.destination.resolve())
