"""Deduplicate one complete private backup without removing any retained path.

Only identical bytes and permission modes may share inodes inside the backup.
The original catalog and published project are never linked or modified.
"""
import argparse
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
BACKUP=ROOT/'runs/dependency-migration-v1/completed-current-backup-04'
RUN=ROOT/'runs/model-portfolio-migration-v1'
LINKS_PER_DURABILITY_BATCH=100


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    pending=path.with_suffix('.pending')
    with pending.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def inactive(backup):
    """Refuse another local user process with this backup selected or open."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            cwd=(proc/'cwd').resolve()
            if cwd.is_relative_to(backup):raise ValueError('Backup is another process working directory')
            for raw in (proc/'cmdline').read_bytes().split(b'\0'):
                value=os.fsdecode(raw)
                if value.startswith(('/', 'runs/', './runs/')):
                    candidate=Path(os.path.abspath(cwd/value))
                    if candidate.is_relative_to(backup):raise ValueError('Backup selected by another process')
            for fd in (proc/'fd').iterdir():
                try:target=Path(os.readlink(fd))
                except OSError:continue
                if target.is_absolute() and target.is_relative_to(backup):raise ValueError('Backup has another process open file')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue


def inventory(backup,receipt):
    catalog=backup/'catalog';expected=receipt['files'];found={};groups=defaultdict(list)
    for path in catalog.rglob('*'):
        name=path.relative_to(catalog).as_posix();info=path.lstat();entry=expected.get(name)
        if entry is None or path.is_symlink():raise ValueError('Unexpected or symbolic backup member')
        if stat.S_ISDIR(info.st_mode):
            if entry['kind']!='directory':raise ValueError('Backup directory type differs')
            found[name]=dict(kind='directory',mode=stat.S_IMODE(info.st_mode));continue
        if not stat.S_ISREG(info.st_mode) or entry['kind']!='file':raise ValueError('Nonregular backup member')
        if info.st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('Backup content differs from original receipt')
        after=path.stat()
        if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Backup member changed while hashing')
        row=dict(kind='file',sha256=entry['sha256'],bytes=info.st_size,mode=stat.S_IMODE(info.st_mode),
                 device=info.st_dev,inode=info.st_ino,links=info.st_nlink,allocated_bytes=info.st_blocks*512)
        found[name]=row;groups[(row['sha256'],row['bytes'],row['mode'])].append(name)
    if found.keys()!=expected.keys():raise ValueError('Backup member population differs')
    return found,groups


def semantic(rows):
    return {name:{k:v for k,v in row.items() if k not in ('device','inode','links','allocated_bytes')} for name,row in rows.items()}


def allocated(rows):
    return sum({(row['device'],row['inode']):row['allocated_bytes'] for row in rows.values() if row['kind']=='file'}.values())


def deduplicate(backup,output):
    if backup.is_symlink() or (backup/'catalog').is_symlink():raise ValueError('Symbolic backup root refused')
    backup=backup.resolve()
    if output.exists():raise ValueError('Fresh operation output required; inspect original owner before another attempt')
    if backup.is_symlink() or not (backup/'catalog').is_dir():raise ValueError('Complete private backup required')
    lock=os.open(backup,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    state=dict(status='verifying_original_backup',created_unix=time.time(),pid=os.getpid(),backup=str(backup),
               files_relinked=0,payload_bytes_copied=0,original_catalog_linked=False,live_project_modified=False,
               original_payload_paths_removed=0,scientific_content_retired=False)
    output.mkdir(parents=True);report=output/'report.json';save(report,state)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inactive(backup)
        receipt_path=backup/'backup-receipt.json';receipt_raw=receipt_path.read_bytes();receipt=json.loads(receipt_raw)
        if receipt['format_version']!=1 or receipt['kind']!='postgresql' or digest(backup/'metadata.json')!=receipt['metadata_sha256']:
            raise ValueError('Original standalone PostgreSQL backup receipt required')
        headers={name:digest(backup/name) for name in ('metadata.json','backup-receipt.json')}
        before,groups=inventory(backup,receipt)
        names=Counter((r['device'],r['inode']) for r in before.values() if r['kind']=='file')
        pairs=[]
        for entries in groups.values():
            private=[name for name in entries if before[name]['links']==names[(before[name]['device'],before[name]['inode'])]]
            distinct={(before[n]['device'],before[n]['inode']) for n in private}
            if len(distinct)<2:continue
            keeper=min(private,key=lambda n:(before[n]['allocated_bytes'],n));identity=(before[keeper]['device'],before[keeper]['inode'])
            pairs.extend(dict(keeper=keeper,target=name,original=before[name]) for name in private
                         if (before[name]['device'],before[name]['inode'])!=identity)
        state.update(status='verified_pending_atomic_relinks',receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                     allocated_bytes_before=allocated(before),candidate_paths=len(pairs),files_before=sum(r['kind']=='file' for r in before.values()))
        save(output/'plan.json',dict(backup=str(backup),receipt_sha256=state['receipt_sha256'],pairs=pairs))
        save(report,state);inactive(backup)
        dirty=set()
        def flush():
            for path in sorted(dirty):
                parent=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:os.fsync(parent)
                finally:os.close(parent)
            os.fsync(lock);dirty.clear()
            state['durable_relinks']=state['files_relinked'];save(report,state)
        for row in pairs:
            keeper=backup/'catalog'/row['keeper'];target=backup/'catalog'/row['target'];old=row['original'];known=before[row['keeper']]
            first=target.lstat();saved=keeper.lstat()
            if ((first.st_dev,first.st_ino,first.st_size,stat.S_IMODE(first.st_mode))!=(old['device'],old['inode'],old['bytes'],old['mode'])
                    or (saved.st_dev,saved.st_ino,saved.st_size,stat.S_IMODE(saved.st_mode))!=(known['device'],known['inode'],known['bytes'],known['mode'])
                    or digest(target)!=old['sha256'] or digest(keeper)!=old['sha256']):
                raise ValueError('Private duplicate or keeper changed before atomic relink')
            pending=backup/('.dedup-pending-'+uuid.uuid4().hex)
            try:
                os.link(keeper,pending,follow_symlinks=False)
                os.replace(pending,target)
                dirty.add(target.parent)
            finally:
                if os.path.lexists(pending):
                    info=pending.lstat()
                    if (info.st_dev,info.st_ino)==(saved.st_dev,saved.st_ino):pending.unlink()
                    else:raise ValueError('Unexpected pending-link identity; retain for inspection')
            state['files_relinked']+=1
            if state['files_relinked']%LINKS_PER_DURABILITY_BATCH==0:flush()
        flush()
        after,_groups=inventory(backup,receipt)
        if semantic(before)!=semantic(after) or headers!={name:digest(backup/name) for name in headers}:
            raise ValueError('Original backup bytes, modes or headers changed')
        inside=Counter((r['device'],r['inode']) for r in after.values() if r['kind']=='file')
        touched={row['keeper'] for row in pairs}|{row['target'] for row in pairs}
        if any(after[name]['links']!=inside[(after[name]['device'],after[name]['inode'])] for name in touched):
            raise ValueError('Deduplicated inode has an alias outside this private backup')
        state.update(status='complete_private_backup_bytes_modes_paths_exact',allocated_bytes_after=allocated(after),
                     allocated_bytes_reclaimed=allocated(before)-allocated(after),original_metadata_and_receipt_exact=True,
                     all_original_paths_present=True,shared_inodes_confined_to_private_backup=True,completed_unix=time.time())
    except BaseException as error:
        state.update(status='failed_or_incomplete',error_type=type(error).__name__,error=str(error));raise
    finally:
        save(report,state);os.close(lock)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(RUN):raise ValueError('Task-owned output directory required')
    print(json.dumps(deduplicate(BACKUP,output)),flush=True)
