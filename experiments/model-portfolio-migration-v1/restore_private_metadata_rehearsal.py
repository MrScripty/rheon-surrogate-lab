"""Verify private metadata restoration on a fresh physical PostgreSQL instance.

This is an isolated SQL rehearsal, not a filesystem restore, source-stop proof
or project extraction. No Dataset, Jobs, Runs or model runtime is constructed.
Only the snapshot already authorized for private retention is consumed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

from qualify_native_workspace import ROOT, SOURCE, PG


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def interrupted(signum, frame):
    raise InterruptedError('Task-owned rehearsal interrupted')


def run(snapshot, output, report, *, archived_backup=False, upgrade=False,
        temporary_postgres=False, source=SOURCE):
    scope = (ROOT / 'runs/model-portfolio-migration-v1').resolve()
    if (os.geteuid() == 0 or output.exists() or report.exists()
            or (not archived_backup and not snapshot.is_relative_to(scope))
            or not output.is_relative_to(scope) or not output.parent.is_dir()
            or not source.is_relative_to(scope / 'sources') or not (source / 'metadata.py').is_file()):
        raise ValueError('Existing snapshot and fresh task-owned rehearsal paths required')
    if archived_backup:
        observed = json.loads((ROOT / 'docs/plans/model-portfolio-migration-v1/reports/catalog-archive-summary-01.json').read_bytes())
        if (snapshot != (ROOT / observed['source_metadata']['path']).parent
                or sha(snapshot / 'backup-receipt.json') != observed['backup_receipt']['sha256']):
            raise ValueError('Exact previously verified archived backup required')
        receipt = json.loads((snapshot / 'backup-receipt.json').read_bytes())
        if (receipt['kind'] != 'postgresql' or receipt['schema_version'] != 8
                or receipt['metadata_sha256'] != observed['source_metadata']['sha256']):
            raise ValueError('Archived native metadata receipt differs')
        summary = dict(metadata_sha256=receipt['metadata_sha256'], schema_version=8,
                       catalog_id=receipt['dataset_id'])
    else:
        if stat.S_IMODE(snapshot.stat().st_mode) != 0o700:
            raise ValueError('Private snapshot directory required')
        for name in ('metadata.json', 'summary.json'):
            path = snapshot / name
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise ValueError('Private independent snapshot file required')
        summary = json.loads((snapshot / 'summary.json').read_bytes())
    if sha(snapshot / 'metadata.json') != summary['metadata_sha256']:
        raise ValueError('Private snapshot bytes changed')
    data = json.loads((snapshot / 'metadata.json').read_bytes())
    if (data['format_version'] != 1 or data['schema_version'] != summary['schema_version']
            or data['dataset_id'] != summary['catalog_id']
            or (not archived_backup and not summary['transaction_read_only'])):
        raise ValueError('Unsupported private observation identity')
    sys.path.insert(0, str(source))
    from metadata import Metadata
    from backup_metadata import populate_catalog, restore_sequences, read_catalog, read_sequences, summaries

    os.umask(0o077)
    output.mkdir(mode=0o700)
    private = output / 'private'
    private.mkdir(mode=0o700)
    storage = Path('/tmp') if temporary_postgres else private
    if shutil.disk_usage(storage).free < 6 * (snapshot / 'metadata.json').stat().st_size + 3 * 1024 ** 3:
        raise ValueError('Insufficient space for generated SQL and its reserve')
    cluster = (Path(tempfile.mkdtemp(prefix='rheon-portfolio-archive-pg-', dir='/tmp'))
               if temporary_postgres else private / 'postgres-data')
    catalog = private / 'catalog'
    socket = Path(tempfile.mkdtemp(prefix='rheon-history-restore-', dir='/tmp'))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    dsn = f'dbname=postgres host={socket} port=56975'
    environment['TULDOK_DATABASE_URL'] = dsn
    value = dict(schema='rheon.private-metadata-SQL-rehearsal.v1', status='starting',
                 source_snapshot_sha256=summary['metadata_sha256'],
                 source_catalog_id=data['dataset_id'], schema_version=data['schema_version'],
                 private_rows_printed=False, original_database_opened=False,
                 live_service_stopped=False, source_files_copied=False,
                 Dataset_constructed=False, models_executed=0, project_extracted=False,
                 postgres_stopped=False, exact_table_identity=False, exact_sequences=False,
                 tables=len(data['tables']), rows=sum(t['count'] for t in data['tables'].values()),
                 source_kind='retained_schema8_backup_metadata_only' if archived_backup else 'authorized_private_readonly_snapshot',
                 physical_cluster=str(cluster), physical_cluster_temporary=temporary_postgres,
                 source_filesystem_restored=False, native_schema_upgrade_requested=upgrade,
                 existing_table_data_preserved_after_upgrade=None,
                 source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                 source_path=str(source.relative_to(ROOT)),
                 runner_sha256=sha(Path(__file__)))

    def save():
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(report)

    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, interrupted)
    db = None
    started = time.monotonic()
    save()
    try:
        with (private / 'postgres-setup.log').open('x') as stream:
            subprocess.run([str(PG / 'initdb'), '-D', str(cluster), '--auth-local=trust',
                            '--auth-host=reject', '--no-locale', '-E', 'UTF8'],
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(private / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56975 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        value['status'] = 'restoring_private_metadata'
        save()
        db = Metadata(catalog, dsn=dsn, dataset_id=data['dataset_id'], schema_version=data['schema_version'])
        with db.lock, db.transaction():
            populate_catalog(db, data['tables'])
            restore_sequences(db, data['sequences'])
        value['status'] = 'verifying_exact_restoration'
        save()
        with db.lock, db.transaction():
            value['exact_table_identity'] = summaries(read_catalog(db)) == summaries(data['tables'])
            value['exact_sequences'] = read_sequences(db) == data['sequences']
        if not value['exact_table_identity'] or not value['exact_sequences']:
            raise ValueError('Restored private rows or sequences differ')
        if upgrade:
            value['status'] = 'upgrading_owned_SQL_without_domain_rewriting'
            save()
            db.close();db = None
            from upgrade_metadata import upgrade_catalog
            value['native_schema_upgrade'] = upgrade_catalog(catalog, dsn=dsn, quiesced=True)
            db = Metadata(catalog, dsn=dsn, create=False)
            with db.lock, db.transaction():
                actual = read_catalog(db)
                value['existing_table_data_preserved_after_upgrade'] = all(
                    actual[name] == expected for name, expected in data['tables'].items()
                    if name != 'catalog_version')
                value['new_tables_empty'] = all(table['count'] == 0 for name, table in actual.items()
                                                if name not in data['tables'])
                sequences = read_sequences(db)
                value['original_sequences_preserved_after_upgrade'] = all(
                    sequences.get(name) == original for name, original in data['sequences'].items())
            if not all(value[name] for name in ('existing_table_data_preserved_after_upgrade',
                       'new_tables_empty', 'original_sequences_preserved_after_upgrade')):
                raise ValueError('Schema upgrade changed retained domain rows or sequences')
            value['final_schema_version'] = db.schema_version
        value.update(status='verified_isolated_SQL_metadata_restore', elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        value.update(status='failed_or_incomplete_private_restore', error_type=type(error).__name__,
                     sqlstate=getattr(error, 'sqlstate', None))
        raise
    finally:
        # Drain this owned database even if a caller interrupts the restore.
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, signal.SIG_IGN)
        if db is not None:
            db.close()
        if cluster.exists():
            result = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-m', 'fast',
                                     '-w', '-t', '30', 'stop'], env=environment,
                                    capture_output=True, text=True, timeout=45)
            (private / 'postgres-stop.log').write_text(result.stdout + result.stderr)
            value['postgres_stopped'] = result.returncode == 0 or 'no server running' in result.stdout + result.stderr
        if value['postgres_stopped']:
            shutil.rmtree(socket)
        save()
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--snapshot', type=Path)
    inputs.add_argument('--archive-backup', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--upgrade', action='store_true')
    parser.add_argument('--temporary-postgres', action='store_true')
    args = parser.parse_args()
    try:
        run((args.archive_backup or args.snapshot).resolve(), args.output.resolve(), args.report.resolve(),
            archived_backup=args.archive_backup is not None, upgrade=args.upgrade,
            temporary_postgres=args.temporary_postgres, source=args.source.resolve())
    except BaseException as error:
        # PostgreSQL exceptions can include retained SQL values; never export them.
        print('Private SQL rehearsal failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
