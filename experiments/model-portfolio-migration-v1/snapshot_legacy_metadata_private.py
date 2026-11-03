"""Capture private catalog metadata for an isolated handoff rehearsal.

Requires separate permission to retain private metadata, including access rows.
This is not a complete filesystem backup, stopped-source proof or extraction.
The verified live connection is used only through PostgreSQL READ ONLY; it is
never written to the output. Ordinary-file/model payloads are not opened here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from catalog_metadata_snapshot import ROOT, SOURCE, CATALOG, CATALOG_ID, configuration
from preview_legacy_history import owner


def run(output, authorized):
    if not authorized:
        print(json.dumps(dict(status='prepared_not_executed', catalog_id=CATALOG_ID,
            retains='Complete opaque SQL values, including private access rows, in mode0600 files under a mode0700 directory.',
            database_mode='READ ONLY REPEATABLE READ', source_files_opened=False,
            live_service_stopped=False, model_execution=False)))
        return
    if output.exists() or not output.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1'):
        raise ValueError('Fresh task-owned private snapshot directory required')
    owner()
    sys.path.insert(0, str(SOURCE))
    from project_transfer import ReadonlyCatalog
    from metadata import connection_info
    from backup_metadata import read_catalog, read_sequences, canonical
    db = ReadonlyCatalog(connection_info(configuration()), 'catalog_' + CATALOG_ID, CATALOG_ID, CATALOG)
    try:
        assert db.execute("SELECT current_setting('transaction_read_only')").fetchone()[0] == 'on'
        assert db.execute("SELECT current_setting('transaction_isolation')").fetchone()[0] == 'repeatable read'
        tables = read_catalog(db)
        data = dict(format_version=1, schema_version=db.schema_version, dataset_id=CATALOG_ID,
                    tables=tables, sequences=read_sequences(db))
        transaction = db.execute('SELECT txid_current_snapshot()::text').fetchone()[0]
        raw = canonical(data) + b'\n'
    finally:
        db.close()
    owner()
    output.mkdir(parents=True, mode=0o700)
    descriptor = os.open(output / 'metadata.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    summary = dict(format='rheon_private_metadata_rehearsal_snapshot_v1',
        catalog_id=CATALOG_ID, schema_version=data['schema_version'],
        metadata_sha256=hashlib.sha256(raw).hexdigest(), metadata_bytes=len(raw),
        tables={name: value['count'] for name, value in tables.items()},
        transaction_read_only=True, transaction_isolation='repeatable read',
        transaction_snapshot=transaction, metadata_contains_private_access_rows=True,
        filesystem_payloads_opened=0, database_writes=0, models_executed=0,
        stopped_source_proof=False, complete_filesystem_backup=False, imported=False,
        limits='Rows are a repeatable-read observation; sequence high-water marks may include concurrent allocations. Private source custody is excluded from portable project records.')
    descriptor = os.open(output / 'summary.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='captured_private_metadata_only', output=str(output.relative_to(ROOT)),
                         metadata_bytes=len(raw), metadata_sha256=summary['metadata_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorized-private-snapshot', action='store_true')
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.authorized_private_snapshot)
    except Exception as error:
        print('Private metadata observation failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
