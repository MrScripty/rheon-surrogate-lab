"""Compact one fixed, retired rehearsal backup and prove a native SQL restore.

Preparation never changes the original backup or live service. Per-file content
and modes, all SQL table hashes and sequences, and independent restored inodes
must match before a public success receipt is written. Temporary restored files
are removed only after verification and a confirmed private PostgreSQL stop.
No Dataset, model runtime, service credential or original database is opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback

from qualify_native_workspace import ROOT, SOURCE, PG


RUN = ROOT / 'runs/model-portfolio-migration-v1'
REHEARSAL = RUN / 'isolated-history-metadata-restore-01/private'


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def usage(directory):
    inodes = {}
    files = 0
    logical = 0
    for path in directory.rglob('*'):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Backup contains a special file or symlink')
        files += 1
        logical += info.st_size
        inodes[(info.st_dev, info.st_ino)] = info.st_blocks * 512
    return dict(files=files, logical_bytes=logical,
                distinct_file_inodes=len(inodes), allocated_file_bytes=sum(inodes.values()))


def run(attempt, output, report):
    source = REHEARSAL / f'extraction-exchange-native-history-extraction-rehearsal-{attempt:02}' / 'stopped-copy'
    terminal = ROOT / f'docs/plans/model-portfolio-migration-v1/reports/native-history-extraction-rehearsal-{attempt:02}.json'
    if (attempt not in (3, 4) or output.exists() or report.exists()
            or not output.is_relative_to(RUN) or not source.is_dir() or source.is_symlink()
            or not json.loads(terminal.read_text())['postgres_stopped']):
        raise ValueError('Fixed retired backup, stopped rehearsal and fresh owned outputs required')
    os.umask(0o077)
    output.mkdir(mode=0o700)
    private = output / 'private'
    private.mkdir(mode=0o700)
    destination = private / 'backup'
    cluster = private / 'postgres-data'
    restored = private / 'restored-catalog'
    socket = Path(tempfile.mkdtemp(prefix='rheon-backup-restore-', dir='/tmp'))
    dsn = f'dbname=postgres host={socket} port=56978'
    environment = dict(os.environ, TULDOK_DATABASE_URL=dsn, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path.insert(0, str(SOURCE))
    from backup_metadata import (tree_manifest, tree_modes, copy_tree, restore_catalog,
        read_catalog, read_sequences, summaries, _remove_partial_contents)
    from metadata import Metadata
    modules = ('backup_metadata.py', 'metadata.py')
    code = {name: file_sha(SOURCE/name) for name in modules}
    value = dict(format='rheon_native_backup_compaction_restore_v1', status='starting',
        created_unix=time.time(), attempt=attempt, source=str(source.relative_to(ROOT)),
        compact_backup=str(destination.relative_to(ROOT)),
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        source_sha256=code, runner_sha256=file_sha(Path(__file__)),
        original_source_modified=False, original_database_opened=False,
        source_credentials_reused=False, raw_backup_modified=False, models_executed=0,
        compact_copy_verified=False, native_restore_verified=False,
        postgres_start_attempted=False, postgres_started=False,
        postgres_stopped=False, temporary_restore_removed=False)

    def save():
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(report)

    def phase(name):
        value['status'] = name
        save()
        print(name, flush=True)

    save()
    try:
        phase('verifying_retired_source_backup_read_only')
        receipt = json.loads((source/'backup-receipt.json').read_text())
        if (receipt['format_version'] != 1 or receipt['kind'] != 'postgresql'
                or file_sha(source/'metadata.json') != receipt['metadata_sha256']
                or tree_manifest(source/'catalog') != receipt['files']):
            raise ValueError('Original native backup receipt or bytes differ')
        before = tree_manifest(source)
        modes = tree_modes(source, before)
        value['source_usage'] = usage(source)
        value['source_receipt_sha256'] = file_sha(source/'backup-receipt.json')
        phase('copying_verified_bytes_into_new_compact_backup')
        copy_tree(source, destination, before, deduplicate=True)
        if (tree_manifest(destination) != before or tree_manifest(source) != before
                or tree_modes(destination, before) != modes or tree_modes(source, before) != modes):
            raise ValueError('Compact copy changed bytes, permissions or original source')
        value['compact_copy_verified'] = True
        value['compact_usage'] = usage(destination)
        phase('restoring_compact_backup_on_fresh_private_PostgreSQL')
        with (private/'postgres-setup.log').open('x') as log:
            subprocess.run([str(PG/'initdb'),'-D',str(cluster),'--auth-local=trust',
                '--auth-host=reject','--no-locale','-E','UTF8'],env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
            value['postgres_start_attempted'] = True
            save()
            subprocess.run([str(PG/'pg_ctl'),'-D',str(cluster),'-l',str(private/'postgres.log'),
                '-o',f"-h '' -k {socket} -p 56978 -c max_connections=20 -c shared_buffers=32MB",
                '-w','-t','45','start'],env=environment,stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=60)
        value['postgres_started'] = True
        save()
        marker = restore_catalog(destination, restored, dsn=dsn)
        marker_names = [p.name for p in restored.glob('.restore-receipt-*.json')]
        if len(marker_names) != 1:
            raise ValueError('Native restore marker is unavailable')
        db = Metadata(restored, dsn=dsn, create=False)
        try:
            actual = summaries(read_catalog(db))
            if actual != receipt['tables'] or read_sequences(db) != receipt['sequences']:
                raise ValueError('Native restored tables or sequences differ')
        finally:
            db.close()
        if (tree_manifest(restored, marker_names) != receipt['files']
                or tree_modes(restored, receipt['files']) != tree_modes(destination/'catalog', receipt['files'])):
            raise ValueError('Native restored ordinary files or modes differ')
        inodes = set()
        count = 0
        for name, entry in receipt['files'].items():
            if entry['kind'] != 'file':
                continue
            restored_info = (restored/name).stat()
            source_info = (destination/'catalog'/name).stat()
            identity = (restored_info.st_dev, restored_info.st_ino)
            if (identity in inodes or restored_info.st_nlink != 1
                    or identity == (source_info.st_dev, source_info.st_ino)):
                raise ValueError('Restored ordinary files share source or sibling inodes')
            inodes.add(identity)
            count += 1
        (private/'native-restore-marker.json').write_text(json.dumps(marker,indent=2)+'\n')
        value.update(native_restore_verified=True, restored_independent_files=count,
                     exact_SQL_tables=len(actual), exact_SQL_rows=sum(t['count'] for t in actual.values()),
                     exact_sequences=True, neural_runtime_imported='torch' in sys.modules)
        if value['neural_runtime_imported'] or code != {name:file_sha(SOURCE/name) for name in modules}:
            raise ValueError('Runtime or source changed during compact-backup qualification')
        phase('native_restore_verified_pending_private_shutdown')
    except BaseException as error:
        value.update(status='failed_or_incomplete_compaction', error_type=type(error).__name__,
            error_frames=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno)
                          for f in traceback.extract_tb(error.__traceback__)])
        raise
    finally:
        if value['postgres_start_attempted']:
            stopped = subprocess.run([str(PG/'pg_ctl'),'-D',str(cluster),'-m','fast',
                '-w','-t','30','stop'],env=environment,capture_output=True,text=True,timeout=45)
            (private/'postgres-stop.log').write_text(stopped.stdout+stopped.stderr)
            value['postgres_stopped'] = stopped.returncode == 0
            if not value['postgres_stopped']:
                observed = subprocess.run([str(PG/'pg_ctl'),'-D',str(cluster),'status'],
                    env=environment,capture_output=True,text=True,timeout=15)
                (private/'postgres-status.log').write_text(observed.stdout+observed.stderr)
                value['postgres_stopped'] = observed.returncode == 3 and not (cluster/'postmaster.pid').exists()
        else:
            value['postgres_stopped'] = True
        if value['postgres_stopped']:
            shutil.rmtree(socket)
        save()
    if not value['postgres_stopped']:
        raise ValueError('Private restore database did not stop; retain all temporary files')
    phase('removing_only_verified_temporary_restore')
    descriptor = os.open(restored,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    held = os.fstat(descriptor)
    try:
        _remove_partial_contents(descriptor)
        current = restored.lstat()
        if (held.st_dev,held.st_ino,held.st_uid) != (current.st_dev,current.st_ino,os.getuid()):
            raise ValueError('Owned temporary restore directory identity changed')
        restored.rmdir()
    finally:
        os.close(descriptor)
    value['temporary_restore_removed'] = True
    phase('qualified_compact_backup_and_native_restore')
    print(json.dumps({key:value[key] for key in ('status','exact_SQL_tables','exact_SQL_rows',
        'restored_independent_files','source_usage','compact_usage','postgres_stopped')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,choices=(3,4),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    try:
        run(args.attempt,args.output.resolve(),args.report.resolve())
    except BaseException as error:
        print('Native backup compaction incomplete:',type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
