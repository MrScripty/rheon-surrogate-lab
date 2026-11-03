"""Read-only native preview of the retained schema8 archive after SQL upgrade.

The archive receipt and exact retained-file index select the source. The keeper
is an explicit read capability, never a writable Dataset or a stopped lease.
Native codecs own closure and diagnostics; missing bytes remain missing.
"""
import argparse
from collections import Counter
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from qualify_native_workspace import ROOT, PG
from preview_portfolio_history import ROOTS, save, sha, interrupted

BASE = ROOT / 'runs/model-portfolio-migration-v1'
DOCS = ROOT / 'docs/plans/model-portfolio-migration-v1/reports'


def run(output, report, native_source=None, manifest_recovery=None):
    restored = json.loads((DOCS / 'archived-history-metadata-restore-01.json').read_bytes())
    proof = json.loads((DOCS / 'archive-retained-file-bindings-01.json').read_bytes())
    cluster = Path(restored['physical_cluster'])
    source_path = native_source or ROOT / restored['source_path']
    archive = ROOT / 'runs/dependency-migration-v1/original-schema8-backup-02'
    keeper = Path(proof['keeper'])
    binding_path = ROOT / proof['binding_index']
    if (os.geteuid() == 0 or output.exists() or report.exists()
            or not output.is_relative_to(BASE) or not report.is_relative_to(DOCS)
            or not source_path.is_relative_to(BASE / 'sources')
            or not cluster.is_dir() or (cluster / 'postmaster.pid').exists()
            or not cluster.name.startswith('rheon-portfolio-archive-pg-')
            or cluster.parent != Path('/tmp')
            or restored['final_schema_version'] != 10 or not restored['postgres_stopped']
            or proof['status'] != 'all_original_catalog_file_bytes_verified'
            or proof['missing'] or proof['unsafe']
            or sha(archive / 'metadata.json') != restored['source_snapshot_sha256']
            or sha(archive / 'backup-receipt.json') != proof['original_receipt_sha256']
            or sha(binding_path) != proof['binding_index_sha256']):
        raise ValueError('Verified stopped archive SQL, original receipt and fresh owned outputs required')
    commit = subprocess.check_output(['git', '-C', str(source_path), 'rev-parse', 'HEAD'], text=True).strip()
    if ((native_source is None and commit != restored['source_commit']) or
            subprocess.check_output(['git', '-C', str(source_path), 'status', '--porcelain'], text=True)):
        raise ValueError('Qualified native source changed')
    original = json.loads((archive / 'metadata.json').read_bytes())
    bindings = json.loads(binding_path.read_bytes())
    if bindings['root'] != str(keeper):
        raise ValueError('Retained binding root differs')
    by_sha = {}
    for row in bindings['bindings']:
        if not row['single_link'] or row['root'] != 'archive-catalog':
            raise ValueError('Retained file is not a native single-file capability')
        by_sha.setdefault(row['sha256'], row)
    # Only explicit original digest references become capabilities. Binding
    # rechecks current byte identity through SourceView; the index is a locator.
    referenced = set(re.findall(r'[a-f0-9]{64}', json.dumps(original['tables'], separators=(',', ':'))))
    os.umask(0o077)
    output.mkdir(mode=0o700)
    socket = Path(tempfile.mkdtemp(prefix='rheon-archive-preview-', dir='/tmp'))
    dsn = f'dbname=postgres host={socket} port=56978'
    environment = dict(os.environ, TULDOK_DATABASE_URL=dsn, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = dict(schema='rheon.archived-history-native-preview.v1', status='starting_owned_SQL',
                  created_unix=time.time(), source_path=str(source_path.relative_to(ROOT)),
                  source_commit=commit, runner_sha256=sha(Path(__file__)),
                  archive_metadata_sha256=restored['source_snapshot_sha256'],
                  retained_file_binding_sha256=proof['binding_index_sha256'],
                  archive_original_regular_files=proof['verified_files'],
                  source_catalog=str(keeper.relative_to(ROOT)),
                  source_catalog_includes_later_retained_files=True,
                  original_database_opened=False, original_service_changed=False,
                  source_files_written=0, copied_database_writes=0,
                  Dataset_constructed=False, stopped_source_lease_issued=False,
                  native_import_performed=False, new_neural_forwards=0, postgres_stopped=False)
    save(report, result)
    db = source = None
    descriptors = {}
    root_fd = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        with (output / 'postgres-start.log').open('x') as log:
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(output / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56978 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        sys.path.insert(0, str(source_path))
        from project_transfer import ReadonlyCatalog, SourceView, ProjectTransfer
        from metadata import connection_info
        from backup_metadata import table_state
        from psycopg import sql
        catalog_id = restored['source_catalog_id']
        db = ReadonlyCatalog(connection_info(dsn), 'catalog_' + catalog_id, catalog_id, keeper)
        if tuple(db.execute("SELECT current_setting('transaction_read_only'), current_setting('transaction_isolation')").fetchone()) != ('on', 'repeatable read'):
            raise ValueError('Pinned read-only repeatable-read transaction required')
        for name, expected in original['tables'].items():
            if name == 'catalog_version':
                continue
            cursor = db.connection.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(name)))
            if table_state([c.name for c in cursor.description], cursor.fetchall()) != expected:
                raise ValueError('Original archived table differs after SQL upgrade')
        result['all_original_nonversion_tables_exact'] = True
        registrations = [json.loads(r[0]) for r in db.execute('SELECT document FROM workspace_projects ORDER BY id')]
        if not registrations:
            raise ValueError('An exact original archive registration is required')
        selected = registrations[0]
        selection = dict(project_snapshot={k: selected[k] for k in ('id', 'manifest_sha256')}, decisions=[])
        selection['roots'] = {
            name: sorted(row[0] for row in db.execute(query).fetchall()) for name, query in (
                ('runs', 'SELECT id FROM workspace_runs'),
                ('studies', 'SELECT id FROM model_studies'),
                ('model_definitions', 'SELECT id FROM model_definition_drafts'),
                ('graphs', 'SELECT id FROM workspace_graphs'),
                ('model_materials', 'SELECT sha256 FROM model_materials'),
                ('recipe_materials', 'SELECT sha256 FROM workspace_recipe_materials'))}
        roots = dict(ROOTS, **{'archive-catalog': keeper,
                              'archive-original-host': ROOT / 'runs/live-integration/Tuldok'})
        for alias, path in roots.items():
            if not path.is_dir():
                continue
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(fd)
            descriptors[alias] = fd, [info.st_dev, info.st_ino]
        root_fd = os.open(keeper, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        actor = db.execute("SELECT id FROM access_actors WHERE kind='owner' ORDER BY id LIMIT 1").fetchone()
        if actor is None:
            raise ValueError('Original archive owner unavailable')
        source = SourceView(db, root_fd, keeper, descriptors, actor_id=actor[0])
        db = root_fd = None
        descriptors = {}
        bound = set()
        result.update(status='binding_exact_original_payloads',
                      selected_roots={k: len(v) for k, v in selection['roots'].items()},
                      retained_referenced_payloads=len(referenced & by_sha.keys()))
        save(report, result)
        for digest in sorted(referenced & by_sha.keys()):
            row = by_sha[digest]
            source.bind_payload(digest, row['bytes'], 'archive-catalog', row['relative'])
            bound.add(digest)
        if manifest_recovery is not None:
            if not manifest_recovery.is_relative_to(DOCS):
                raise ValueError('Explicit task-owned archive recovery report required')
            recovery = json.loads(manifest_recovery.read_bytes())
            index = ROOT / recovery['binding_index']
            if (recovery['archive_metadata_sha256'] != restored['source_snapshot_sha256']
                    or not index.is_relative_to(BASE) or sha(index) != recovery['binding_index_sha256']):
                raise ValueError('Recovered archive manifest provenance differs')
            for row in json.loads(index.read_bytes())['bindings']:
                if row['sha256'] not in referenced or row['root'] != 'current-lab':
                    raise ValueError('Manifest binding is not named by the original archive')
                source.bind_payload(row['sha256'], row['bytes'], row['root'], row['relative'])
                bound.add(row['sha256'])
            result['manifest_recovery'] = dict(path=str(manifest_recovery.relative_to(ROOT)),
                                              sha256=sha(manifest_recovery))
        # Known recovered manifests/host versions are explicit operator input,
        # and are bound only if the original archive actually names that digest.
        recovered = json.loads((BASE / 'full-current-host-recovery-01/recovery.json').read_bytes())
        recovered_by_sha = {r['sha256']: ROOT / r['recovered_path'] for r in recovered['rows']
                            if r['state'] == 'verified_exact_original_host_bytes'}
        hosts, gaps = [], []
        for row in source.db.execute('SELECT document FROM workspace_runs ORDER BY id'):
            run = json.loads(row[0])
            for name, digest in run.get('material', {}).get('host_files', {}).items():
                if digest not in bound:
                    candidates = [('current-lab', recovered_by_sha[digest])] if digest in recovered_by_sha else []
                    candidates += [(alias, roots[alias] / name) for alias in (
                        'archive-original-host', 'migration-host', 'migration-original-host')]
                    for alias, file in candidates:
                        if file.is_file() and not file.is_symlink() and sha(file) == digest:
                            source.bind_payload(digest, file.stat().st_size, alias, str(file.relative_to(roots[alias])))
                            bound.add(digest)
                            break
                (hosts if digest in bound else gaps).append(dict(run_id=run['id'], name=name, sha256=digest))
        save(output / 'selection.json', selection)
        save(output / 'host-bindings.json', dict(bound=hosts, gaps=gaps))
        result.update(status='native_archive_preview_in_flight', bound_payloads=len(bound),
                      host_observations_bound=len(hosts), host_observations_missing=len(gaps))
        save(report, result)
        del original
        with (output / 'python-stacks.log').open('x') as stack_log:
            faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=False)
            try:
                preview = ProjectTransfer(source).preview(selection)
            finally:
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
                faulthandler.unregister(signal.SIGUSR1)
                signal.signal(signal.SIGUSR1, signal.SIG_IGN)
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
        save(output / 'native-preview.json', preview)
        result.update(status='native_archive_readonly_preview_completed',
                      native_preview_path=str((output / 'native-preview.json').relative_to(ROOT)),
                      native_preview_sha256=sha(output / 'native-preview.json'),
                      can_extract=preview['can_extract'],
                      included_records=len(preview['included']),
                      included_families=dict(Counter(r['family'] for r in preview['included'])),
                      blockers=dict(Counter(r['code'] for r in preview['blockers'])),
                      artifact_states=dict(Counter(r['state'] for r in preview['artifacts'])))
    except BaseException as error:
        result.update(status='failed_or_incomplete_archive_preview', error_type=type(error).__name__,
                      native_error_code=getattr(error, 'code', None),
                      error_frames=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                                    for f in traceback.extract_tb(error.__traceback__)])
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
        for fd, _ in descriptors.values():
            os.close(fd)
        stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-m', 'fast', '-w', '-t', '30', 'stop'],
                                 capture_output=True, text=True, timeout=45)
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
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest-recovery', type=Path)
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.report.resolve(),
            args.source.resolve() if args.source else None,
            args.manifest_recovery.resolve() if args.manifest_recovery else None)
    except BaseException as error:
        print('Archived read-only preview failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
