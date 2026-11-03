"""Compact verified payload duplicates in explicit inactive rehearsal catalogs.

The original complete backup is unchanged. Distinct bytes and all root metadata
remain. The per-file manifest restores an ordinary independent catalog copy.
"""
import hashlib
import json
import os
from pathlib import Path
import argparse
import shutil
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/model-portfolio-migration-v1'
OLD=None
KEEPER=ROOT/'runs/dependency-migration-v1/completed-current-backup-04'
PLAN=None
REPORT=None
MARKER=None


def digest(path):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('Ordinary file required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    after=path.lstat()
    if signature(before)!=signature(after):raise ValueError('File changed during verification')
    return h.hexdigest()


def signature(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode,info.st_nlink)


def write(path,value,fresh=False):
    with path.open('x' if fresh else 'w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def inactive(directory=None):
    root=str(OLD if directory is None else directory)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:descriptors=list((proc/'fd').iterdir())
        except (PermissionError,FileNotFoundError):continue
        for fd in descriptors:
            try:name=os.readlink(fd)
            except OSError:continue
            if name==root or name.startswith(root+'/'):raise ValueError('Retired target has an open file descriptor')
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError):continue
        if any(v==root.encode() or v.startswith(root.encode()+b'/') for v in args):
            raise ValueError('A process selects the retired target')


def safe(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
    target=root/p
    for part in (target,*target.parents):
        if part.is_symlink():raise ValueError('Symlink in selected tree')
        if part==root:break
    return target


def verify_keeper():
    receipt=json.loads((KEEPER/'backup-receipt.json').read_bytes())
    if receipt['dataset_id']!='a7da4a8de53541a2a0e2aad08005bb71' or digest(KEEPER/'metadata.json')!=receipt['metadata_sha256']:
        raise ValueError('Complete keeper identity differs')
    rows={};memo={}
    for relative,row in receipt['files'].items():
        path=safe(KEEPER/'catalog',relative);info=path.lstat()
        if row['kind']=='directory':
            if not stat.S_ISDIR(info.st_mode):raise ValueError('Keeper directory differs')
            continue
        if row['kind']!='file' or not stat.S_ISREG(info.st_mode) or info.st_size!=row['bytes']:
            raise ValueError('Keeper file differs')
        key=signature(info)
        if key not in memo:memo[key]=digest(path)
        if memo[key]!=row['sha256']:raise ValueError('Keeper file digest differs')
        rows[relative]=dict(bytes=row['bytes'],sha256=row['sha256'],mode=stat.S_IMODE(info.st_mode))
    return rows


def compact():
    if any(p.exists() for p in (PLAN,REPORT,MARKER)):raise ValueError('Fresh fixed-scope compaction required')
    inactive();identity=OLD.stat();keeper=verify_keeper();by_hash={}
    for relative,row in sorted(keeper.items()):by_hash.setdefault((row['bytes'],row['sha256'],row['mode']),relative)
    files=[];directories=[dict(path='.',mode=stat.S_IMODE(identity.st_mode))]
    for path in sorted(OLD.rglob('*')):
        info=path.lstat();relative=str(path.relative_to(OLD))
        if info.st_uid!=os.getuid():raise ValueError('Target ownership differs')
        if stat.S_ISDIR(info.st_mode):
            directories.append(dict(path=relative,mode=stat.S_IMODE(info.st_mode)));continue
        if not stat.S_ISREG(info.st_mode):raise ValueError('Unexpected target file kind')
        row=dict(path=relative,bytes=info.st_size,sha256=digest(path),mode=stat.S_IMODE(info.st_mode),allocated_bytes=info.st_blocks*512)
        # Preserve every original root header/restore/migration receipt even if
        # identical; only subordinate scientific/runtime payloads are retired.
        kept=None if len(Path(relative).parts)==1 else by_hash.get((row['bytes'],row['sha256'],row['mode']))
        row['keeper_path']=kept;files.append(row)
    plan=dict(schema='rheon.retired-catalog-payload-compaction.v1',created_unix=time.time(),source=str(OLD.relative_to(ROOT)),
        keeper=str(KEEPER.relative_to(ROOT)),keeper_receipt_sha256=digest(KEEPER/'backup-receipt.json'),
        source_root_identity=[identity.st_dev,identity.st_ino],directories=directories,files=files,
        runner_sha256=digest(Path(__file__)),authorization='User authorized removing identical backup files and requested space recovery; fixed inactive migration rehearsal only.')
    write(PLAN,plan,True)
    write(MARKER,dict(schema=plan['schema'],status='compacting',manifest=str(PLAN.relative_to(ROOT)),manifest_sha256=digest(PLAN),
                      source_restart_requires_materialization=True,keeper=str(KEEPER.relative_to(ROOT))),True)
    result=dict(schema=plan['schema'],status='removing_verified_duplicate_files',created_unix=time.time(),manifest=str(PLAN.relative_to(ROOT)),
        manifest_sha256=digest(PLAN),keeper=str(KEEPER.relative_to(ROOT)),duplicate_files=sum(r['keeper_path'] is not None for r in files),
        duplicate_logical_bytes=sum(r['bytes'] for r in files if r['keeper_path'] is not None),
        distinct_or_root_metadata_files_retained=sum(r['keeper_path'] is None for r in files),
        free_bytes_before=shutil.disk_usage(ROOT).free,unique_payloads_deleted=0,original_service_changed=False)
    write(REPORT,result,True);print(json.dumps(result),flush=True)
    inactive()
    if (OLD.stat().st_dev,OLD.stat().st_ino)!=(identity.st_dev,identity.st_ino):raise ValueError('Target root changed')
    for row in files:
        path=safe(OLD,row['path'])
        if row['keeper_path'] is None:continue
        kept=safe(KEEPER/'catalog',row['keeper_path'])
        if digest(path)!=row['sha256'] or digest(kept)!=row['sha256']:raise ValueError('Duplicate identity changed')
        path.unlink()
    verify_keeper()
    for row in files:
        if row['keeper_path'] is None and digest(safe(OLD,row['path']))!=row['sha256']:raise ValueError('Distinct or metadata file changed')
    if digest(KEEPER/'backup-receipt.json')!=plan['keeper_receipt_sha256']:raise ValueError('Keeper receipt changed')
    marker=json.loads(MARKER.read_bytes());marker['status']='compacted_requires_materialization';write(MARKER,marker)
    result.update(status='completed_verified_retired_catalog_compaction',completed_unix=time.time(),free_bytes_after=shutil.disk_usage(ROOT).free,
                  keeper_complete_and_verified_after=True,distinct_and_root_metadata_verified_after=True)
    write(REPORT,result);print(json.dumps(result),flush=True)


def materialize(destination):
    destination=destination.resolve()
    if destination.exists() or any(destination.is_relative_to(p) for p in (OLD,KEEPER)):raise ValueError('Absent external destination required')
    marker=json.loads(MARKER.read_bytes());plan=json.loads(PLAN.read_bytes())
    if marker['status']!='compacted_requires_materialization' or digest(PLAN)!=marker['manifest_sha256'] or digest(KEEPER/'backup-receipt.json')!=plan['keeper_receipt_sha256']:
        raise ValueError('Compaction bindings differ')
    verify_keeper()
    if shutil.disk_usage(destination.parent).free<sum(r['bytes'] for r in plan['files'])+1024**3:raise ValueError('Insufficient free materialization space')
    destination.mkdir(mode=0o700)
    for row in plan['directories']:
        p=destination/row['path'];p.mkdir(mode=0o700,exist_ok=True);p.chmod(row['mode'])
    for row in plan['files']:
        source=safe(KEEPER/'catalog',row['keeper_path']) if row['keeper_path'] is not None else safe(OLD,row['path'])
        if digest(source)!=row['sha256']:raise ValueError('Materialization source changed')
        target=destination/row['path'];shutil.copyfile(source,target);target.chmod(row['mode'])
        if digest(target)!=row['sha256']:raise ValueError('Materialized file differs')
    print(json.dumps(dict(status='materialized_exact_retired_catalog',destination=str(destination))))


def select_catalog(index):
    global OLD,PLAN,REPORT,MARKER
    if type(index) is not int or index not in (1,2,3):
        raise ValueError('Only the three fixed inactive migration rehearsals are eligible')
    OLD=ROOT/('runs/dependency-migration-v1/catalog-rehearsal-%02d'%index)
    PLAN=BASE/('discovery/retired-catalog%02d-compaction-01.json'%index)
    REPORT=ROOT/('docs/plans/model-portfolio-migration-v1/reports/retired-catalog%02d-compaction-01.json'%index)
    MARKER=OLD/'COMPACTED-SOURCE.json'


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('compact','materialize'))
    parser.add_argument('--catalog-index',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args();select_catalog(args.catalog_index)
    if args.command=='compact':
        if args.destination is not None:parser.error('Compact has no destination option')
        compact()
    else:
        if args.destination is None:parser.error('Materialize requires an absent external destination')
        materialize(args.destination)
