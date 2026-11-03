"""Preview all current-catalog model history using an owned SQL copy.

The original filesystem is opened through native read capabilities; its service
and database are untouched. A retained private snapshot proves the copied
scientific tables before preview. No Dataset, worker, stopped lease or import is
constructed. Native blockers remain blockers, including absent original bytes.
"""
import argparse
from collections import Counter
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from qualify_native_workspace import ROOT, SOURCE, PG
from catalog_metadata_snapshot import CATALOG, CATALOG_ID, OLD_SERVICE
from preview_legacy_history import selection as anchor_selection

BASE = ROOT / 'runs/model-portfolio-migration-v1'
SNAPSHOT = BASE / 'private-history-rehearsal-snapshot-01'
REHEARSAL = BASE / 'isolated-history-metadata-restore-01'
TABLES = ('workspace_runs', 'model_studies', 'model_trials', 'model_definition_drafts',
          'model_materials', 'workspace_recipe_materials', 'workspace_graphs', 'workspace_projects')
ROOTS = {
    'current-lab': ROOT,
    'catalog-source': CATALOG,
    'historical': ROOT / 'runs/dependency-migration-v1/domain-original-stage-01',
    'lab': ROOT / 'runs/dependency-migration-v1/domain-original-stage-01/external/lab',
    'pressure': ROOT,
    'long-training': Path('/tmp/rheon-shared-long-training-candidate-01'),
    'migration-host': OLD_SERVICE,
    'migration-original-host': ROOT / 'runs/dependency-migration-v1/Tuldok',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


def interrupted(signum, frame):
    raise InterruptedError('Owned preview interrupted')


def run(output, report, source_path=SOURCE, host_recovery=None, payload_bindings=None):
    cluster = REHEARSAL / 'private/postgres-data'
    if (os.geteuid() == 0 or output.exists() or report.exists()
            or not output.is_relative_to(BASE) or not cluster.is_dir()
            or (cluster / 'postmaster.pid').exists()
            or not source_path.is_relative_to(BASE / 'sources')
            or not (source_path / 'project_extraction.py').is_file()):
        raise ValueError('Fresh outputs and confirmed stopped task-owned SQL copy required')
    summary = json.loads((SNAPSHOT / 'summary.json').read_bytes())
    if sha(SNAPSHOT / 'metadata.json') != summary['metadata_sha256']:
        raise ValueError('Retained original metadata identity changed')
    data = json.loads((SNAPSHOT / 'metadata.json').read_bytes())
    if (data['dataset_id'] != CATALOG_ID or data['schema_version'] != 10
            or json.loads((CATALOG / '.tuldok.json').read_bytes())['dataset_id'] != CATALOG_ID):
        raise ValueError('Original source namespace/schema differs')
    os.umask(0o077)
    output.mkdir(mode=0o700)
    socket = Path(tempfile.mkdtemp(prefix='rheon-portfolio-preview-', dir='/tmp'))
    dsn = f'dbname=postgres host={socket} port=56976'
    environment = dict(os.environ, TULDOK_DATABASE_URL=dsn, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = dict(schema='rheon.full-current-history-preview.v1', status='starting_owned_SQL_copy',
                  created_unix=time.time(), source_snapshot_sha256=summary['metadata_sha256'],
                  source_path=str(source_path.relative_to(ROOT)),
                  source_commit=subprocess.check_output(['git', '-C', str(source_path), 'rev-parse', 'HEAD'], text=True).strip(),
                  source_clean=not bool(subprocess.check_output(['git', '-C', str(source_path), 'status', '--porcelain'], text=True)),
                  runner_sha256=sha(Path(__file__)), original_service_changed=False,
                  original_database_opened=False, copied_database_writes=0,
                  source_files_written=0, new_neural_forwards=0, Dataset_constructed=False,
                  stopped_source_lease_issued=False, native_import_performed=False,
                  postgres_stopped=False, selected_public_tables_exact=False)
    save(report, result)
    db = source = None
    root_fd = None
    roots = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        with (output / 'postgres-start.log').open('x') as log:
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(output / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56976 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        sys.path.insert(0, str(source_path))
        from project_transfer import ReadonlyCatalog, SourceView, ProjectTransfer
        from metadata import connection_info
        from backup_metadata import table_state
        from psycopg import sql
        db = ReadonlyCatalog(connection_info(dsn), 'catalog_' + CATALOG_ID, CATALOG_ID, CATALOG)
        if tuple(db.execute("SELECT current_setting('transaction_read_only'),current_setting('transaction_isolation')").fetchone()) != ('on', 'repeatable read'):
            raise ValueError('Source SQL transaction is not pinned read-only')
        result['status'] = 'checking_original_scientific_rows'
        save(report, result)
        for table in TABLES:
            cursor = db.connection.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table)))
            actual = table_state([column.name for column in cursor.description], cursor.fetchall())
            expected = data['tables'][table]
            if actual != expected:
                result['mismatched_scientific_table'] = table
                raise ValueError('Disposable scientific rows differ from original snapshot')
        result['selected_public_tables_exact'] = True
        selection = anchor_selection()
        selection['roots'] = {
            root: sorted(row[0] for row in db.execute(statement).fetchall())
            for root, statement in (
                ('runs', 'SELECT id FROM workspace_runs ORDER BY id'),
                ('studies', 'SELECT id FROM model_studies ORDER BY id'),
                ('model_definitions', 'SELECT id FROM model_definition_drafts ORDER BY id'),
                ('graphs', 'SELECT id FROM workspace_graphs ORDER BY id'),
                ('model_materials', 'SELECT sha256 FROM model_materials ORDER BY sha256'),
                ('recipe_materials', 'SELECT sha256 FROM workspace_recipe_materials ORDER BY sha256'))}
        # Each source root is an explicit operator binding. Stored old paths or
        # root identities do not create capabilities for missing directories.
        result['unavailable_operator_roots'] = [name for name, path in ROOTS.items() if not path.is_dir()]
        for name, path in ROOTS.items():
            if not path.is_dir():
                continue
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(fd)
            roots[name] = fd, [info.st_dev, info.st_ino]
        root_fd = os.open(CATALOG, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        actor = db.execute("SELECT id FROM access_actors WHERE kind='owner' ORDER BY id LIMIT 1").fetchone()
        if actor is None:
            raise ValueError('Retained catalog lacks original owner identity')
        source = SourceView(db, root_fd, CATALOG, roots, actor_id=actor[0])
        db = None
        root_fd = None
        roots = {}
        if payload_bindings is not None:
            if not payload_bindings.is_relative_to(BASE):
                raise ValueError('Explicit task-owned payload bindings required')
            binding_raw = payload_bindings.read_bytes()
            binding_document = json.loads(binding_raw)
            if (set(binding_document) != {'schema', 'bindings'}
                    or binding_document['schema'] != 'rheon.explicit-history-payload-bindings.v1'
                    or not isinstance(binding_document['bindings'], list)
                    or not 1 <= len(binding_document['bindings']) <= 100):
                raise ValueError('Unsupported explicit history bindings')
            result.update(status='verifying_explicit_original_payloads',
                          explicit_binding_source=dict(path=str(payload_bindings.relative_to(ROOT)),
                              sha256=hashlib.sha256(binding_raw).hexdigest()),
                          verified_explicit_bindings=[])
            save(report, result)
            for row in binding_document['bindings']:
                if set(row) != {'role', 'sha256', 'bytes', 'root', 'relative'}:
                    raise ValueError('Unknown original payload binding fields')
                source.bind_payload(row['sha256'], row['bytes'], row['root'], row['relative'])
                result['verified_explicit_bindings'].append(row)
                save(report, result)
        runs = [json.loads(row[0]) for row in source.db.execute('SELECT document FROM workspace_runs ORDER BY id')]
        recovered = {}
        if host_recovery is not None:
            if not host_recovery.is_relative_to(BASE):
                raise ValueError('Explicit task-owned host recovery required')
            recovery_raw = host_recovery.read_bytes()
            recovery = json.loads(recovery_raw)
            for row in recovery['rows']:
                path = (ROOT / row['recovered_path']).resolve()
                if (not path.is_relative_to(BASE) or row['state'] != 'verified_exact_original_host_bytes'
                        or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']):
                    raise ValueError('Recovered host source no longer matches its exact identity')
                recovered[row['sha256']] = path
            result['host_recovery'] = dict(path=str(host_recovery.relative_to(ROOT)),
                                           sha256=hashlib.sha256(recovery_raw).hexdigest())
        host_matches, host_gaps, bound = [], [], set()
        for run in runs:
            for name, expected in sorted(run.get('material', {}).get('host_files', {}).items()):
                match = None
                for alias in ('migration-host', 'migration-original-host'):
                    path = ROOTS[alias] / name
                    if path.is_file() and not path.is_symlink() and sha(path) == expected:
                        match = alias, path
                        break
                if match is None and expected in recovered:
                    match = 'current-lab', recovered[expected]
                if match is None:
                    host_gaps.append(dict(run_id=run['id'], name=name, expected_sha256=expected))
                    continue
                alias, path = match
                if expected not in bound:
                    relative = str(path.relative_to(ROOTS[alias]))
                    source.bind_payload(expected, path.stat().st_size, alias, relative)
                    bound.add(expected)
                host_matches.append(dict(run_id=run['id'], name=name, sha256=expected))
        capture_matches = []
        for run in runs:
            material_sha = run.get('approval_material_sha256')
            if material_sha is None or run.get('execution_kind') != 'external_recipe':
                continue
            row = source.db.execute('SELECT document FROM workspace_recipe_materials WHERE sha256=%s', (material_sha,)).fetchone()
            if row is None:
                continue
            item = json.loads(row[0])['material']['recipe_document']
            folder = 'workspace-recipe-materials/' + material_sha + '/'
            raw = source.read_catalog(folder + item['name'], 2 * 1024 * 1024)
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Original approved recipe snapshot changed')
            for entry in json.loads(raw)['capture_paths']:
                original = run['material']['captures'][entry['path']]
                blob = entry['blob']
                if any(original[key] != blob[key] for key in ('sha256', 'bytes')):
                    raise ValueError('Original Run capture disagrees with approved material')
                if blob['sha256'] not in bound:
                    source.bind_payload(blob['sha256'], blob['bytes'], 'catalog-source', folder + blob['name'])
                    bound.add(blob['sha256'])
                capture_matches.append(dict(run_id=run['id'], name=entry['path'], sha256=blob['sha256']))
        save(output / 'selection.json', selection)
        save(output / 'payload-bindings.json', dict(host_matches=host_matches, host_gaps=host_gaps, capture_matches=capture_matches))
        result.update(status='native_full_current_preview_in_flight',
                      selected_roots={name: len(values) for name, values in selection['roots'].items()},
                      host_source_observations_bound=len(host_matches), host_source_gaps=len(host_gaps),
                      approved_capture_observations_bound=len(capture_matches))
        save(report, result)
        print(json.dumps({key: result[key] for key in ('status', 'selected_roots', 'host_source_gaps')}), flush=True)
        del data
        # A live stack can be sampled without debugger privileges or stopping a
        # read-only operation. This file contains frames only, never local data.
        with (output / 'python-stacks.log').open('x') as stack_log:
            faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=False)
            try:
                transfer = ProjectTransfer(source)
                preview = transfer.preview(selection)
            finally:
                faulthandler.unregister(signal.SIGUSR1)
        save(output / 'native-preview.json', preview)
        from project_store import _encode
        selected_records = transfer._reports[preview['report_sha256']]['records']
        result['selected_canonical_record_bytes'] = sum(len(_encode(row['document'])) for row in selected_records)
        result['selected_canonical_record_count'] = len(selected_records)
        result['verified_payload_bytes'] = sum(row['bytes'] for row in preview['artifacts'] if row['state'] == 'verified')
        result['footprint_excludes_publication_overlays_directory_blocks_SQL_projection_and_backup'] = True
        if 'torch' in sys.modules:
            raise ValueError('Metadata preview imported neural runtime')
        result.update(status='completed_native_full_current_preview',
                      native_preview_path=str((output / 'native-preview.json').relative_to(ROOT)),
                      native_preview_sha256=sha(output / 'native-preview.json'),
                      included_records=len(preview['included']),
                      included_families=dict(Counter(row['family'] for row in preview['included'])),
                      blockers=dict(Counter(row['code'] for row in preview['blockers'])),
                      can_extract=preview['can_extract'],
                      verified_payloads=sum(item['state'] == 'verified' for item in preview['artifacts']),
                      missing_unique_payloads=sum(item['state'] != 'verified' and item['preservation'] == 'unique' for item in preview['artifacts']),
                      missing_regenerable_payloads=sum(item['state'] != 'verified' and item['preservation'] == 'regenerable' for item in preview['artifacts']))
    except BaseException as error:
        result.update(status='failed_or_incomplete_native_full_current_preview', error_type=type(error).__name__,
                      native_error_code=getattr(error, 'code', None),
                      error_frames=[dict(file=Path(frame.filename).name, line=frame.lineno, function=frame.name)
                                    for frame in traceback.extract_tb(error.__traceback__)])
        # Raw SQL exceptions can contain retained private rows. Export no values.
        if type(error).__name__ in ('WorkbenchError', 'ProjectStoreError'):
            result['native_error_message'] = str(error)
        raise
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if source is not None:
            source.close()
        if db is not None:
            db.close()
        if root_fd is not None:
            os.close(root_fd)
        for fd, _ in roots.values():
            os.close(fd)
        stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-m', 'fast', '-w', '-t', '30', 'stop'],
                                 env=environment, capture_output=True, text=True, timeout=45)
        (output / 'postgres-stop.log').write_text(stopped.stdout + stopped.stderr)
        result['postgres_stopped'] = stopped.returncode == 0 or 'no server running' in stopped.stdout + stopped.stderr
        if result['postgres_stopped']:
            shutil.rmtree(socket)
        save(report, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--host-recovery', type=Path)
    parser.add_argument('--payload-bindings', type=Path)
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.report.resolve(), args.source.resolve(),
            args.host_recovery.resolve() if args.host_recovery else None,
            args.payload_bindings.resolve() if args.payload_bindings else None)
    except BaseException as error:
        print('Full current-history preview failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
