"""Extract genuine selected history from the task-owned disposable catalog copy.

The original service is neither opened nor stopped. The running copy's real
Dataset owner must issue its own stopped-source lease. Original SQL metadata
and selected byte provenance remain bound to the authorized private snapshot.
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
import stat
import subprocess
import sys
import tempfile
import time
import traceback

from qualify_native_workspace import ROOT, SOURCE, PG
from preview_legacy_history import HOST_FILES, HOST_ROOTS, RUN_IDS, selection


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bind_hosts(source):
    """Resolve only the same exact historical sources admitted by the preview."""
    bound = set()
    matches = []
    for run_id in RUN_IDS:
        row = source.db.execute('SELECT document FROM workspace_runs WHERE id=%s', (run_id,)).fetchone()
        hosts = json.loads(row[0])['material']['host_files']
        for name in sorted(set(hosts) & HOST_FILES):
            expected = hosts[name]
            for alias, directory in HOST_ROOTS.items():
                path = directory / name
                if path.is_file() and not path.is_symlink() and sha(path) == expected:
                    if expected not in bound:
                        source.bind_payload(expected, path.stat().st_size, alias, name)
                        bound.add(expected)
                    matches.append(dict(run_id=run_id, source=name, sha256=expected, root=alias))
                    break
            else:
                raise ValueError('Exact historical host source is unavailable')
    return matches


def bind_required_inputs(source):
    hosts = bind_hosts(source)
    captures = []
    for run_id in RUN_IDS:
        run = json.loads(source.db.execute('SELECT document FROM workspace_runs WHERE id=%s',
                                          (run_id,)).fetchone()[0])
        material_sha = run['approval_material_sha256']
        composition = json.loads(source.db.execute(
            'SELECT document FROM workspace_recipe_materials WHERE sha256=%s',
            (material_sha,)).fetchone()[0])
        folder = Path('workspace-recipe-materials') / material_sha
        item = composition['material']['recipe_document']
        raw = source.read_catalog(str(folder / item['name']), 2 * 1024 ** 2)
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Exact retained recipe snapshot required')
        snapshot = json.loads(raw)
        for entry in snapshot['capture_paths']:
            original = run['material']['captures'][entry['path']]
            blob = entry['blob']
            if any(blob[key] != original[key] for key in ('sha256', 'bytes')):
                raise ValueError('Run input and retained material byte identities differ')
            # Prepared material bytes are the same original inputs as the old
            # execution directory. Bind the retained copy explicitly; no replay
            # or reconstruction supplies missing source bytes.
            relative = str(source.root_path.relative_to(ROOT) / folder / blob['name'])
            source.bind_payload(blob['sha256'], blob['bytes'], 'current-lab', relative)
            captures.append(dict(run_id=run_id, name=entry['path'], sha256=blob['sha256'],
                                 bytes=blob['bytes'], root='current-lab', relative=relative))
    return dict(hosts=hosts, captures=captures)


def bind_current_inputs(source, run_ids, host_recovery, payload_bindings):
    """Bind the exact originals qualified by the full-catalog preview.

    Historical roots are observations only. The operator supplies existing
    roots and exact recovered bytes explicitly, including the large inputs.
    Native bind_payload remains the byte and descriptor authority.
    """
    from preview_portfolio_history import ROOTS, BASE
    recovered = {}
    for row in json.loads(host_recovery.read_bytes())['rows']:
        path = (ROOT / row['recovered_path']).resolve()
        if (not path.is_relative_to(BASE) or path.is_symlink()
                or row['state'] != 'verified_exact_original_host_bytes'
                or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']):
            raise ValueError('Recovered original host bytes changed')
        recovered[row['sha256']] = path
    bindings = json.loads(payload_bindings.read_bytes())
    if (set(bindings) != {'schema', 'bindings'}
            or bindings['schema'] != 'rheon.explicit-history-payload-bindings.v1'
            or not 1 <= len(bindings['bindings']) <= 100):
        raise ValueError('Explicit original payload bindings required')
    bound = set()
    for row in bindings['bindings']:
        if set(row) != {'role', 'sha256', 'bytes', 'root', 'relative'}:
            raise ValueError('Unknown original binding fields')
        source.bind_payload(row['sha256'], row['bytes'], row['root'], row['relative'])
        bound.add(row['sha256'])
    hosts, captures, original_captures = [], [], []
    for run_id in run_ids:
        run = json.loads(source.db.execute(
            'SELECT document FROM workspace_runs WHERE id=%s', (run_id,)).fetchone()[0])
        # The owned filesystem copy intentionally omits most execution staging
        # inputs. They are original unique evidence, including runtime manifests
        # and bootstrap proofs, beyond the approved recipe's input subset.
        for name, blob in sorted(run.get('material', {}).get('captures', {}).items()):
            if blob['sha256'] not in bound:
                source.bind_payload(blob['sha256'], blob['bytes'], 'catalog-source',
                                    'workspace-runs/.external-' + run_id + '/' + name)
                bound.add(blob['sha256'])
            original_captures.append(dict(run_id=run_id, name=name, sha256=blob['sha256']))
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
                raise ValueError('Exact historical host source is unavailable')
            alias, path = match
            if expected not in bound:
                source.bind_payload(expected, path.stat().st_size, alias,
                                    str(path.relative_to(ROOTS[alias])))
                bound.add(expected)
            hosts.append(dict(run_id=run_id, name=name, sha256=expected))
        material_sha = run.get('approval_material_sha256')
        if material_sha is None or run.get('execution_kind') != 'external_recipe':
            continue
        row = source.db.execute('SELECT document FROM workspace_recipe_materials WHERE sha256=%s',
                                (material_sha,)).fetchone()
        if row is None:
            raise ValueError('Original approved recipe material is unavailable')
        item = json.loads(row[0])['material']['recipe_document']
        folder = 'workspace-recipe-materials/' + material_sha + '/'
        raw = source.read_catalog(folder + item['name'], 2 * 1024 ** 2)
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Original approved recipe snapshot changed')
        for entry in json.loads(raw)['capture_paths']:
            original = run['material']['captures'][entry['path']]
            blob = entry['blob']
            if any(original[key] != blob[key] for key in ('sha256', 'bytes')):
                raise ValueError('Original capture disagrees with approved material')
            if blob['sha256'] not in bound:
                source.bind_payload(blob['sha256'], blob['bytes'], 'current-lab',
                                    str(source.root_path.relative_to(ROOT) / folder / blob['name']))
                bound.add(blob['sha256'])
            captures.append(dict(run_id=run_id, name=entry['path'], sha256=blob['sha256']))
    return dict(hosts=hosts, captures=captures, original_execution_captures=original_captures,
                explicit_binding_sha256=sha(payload_bindings), host_recovery_sha256=sha(host_recovery))


def run(rehearsal, destination, report, *, selection_path=None, host_recovery=None,
        payload_bindings=None, space_preflight=None,
        title='Rheon C106/C107 history rehearsal', request_key='rheon-c106-c107-history-rehearsal-01'):
    scope = (ROOT / 'runs/model-portfolio-migration-v1').resolve()
    if (not rehearsal.is_relative_to(scope) or not rehearsal.is_dir()
            or destination.exists() or not destination.parent.is_dir() or report.exists()):
        raise ValueError('Owned source copy and absent destination/report required')
    private = rehearsal / 'private'
    catalog = private / 'catalog'
    cluster = private / 'postgres-data'
    exchange = private / ('extraction-exchange-' + report.stem)
    if stat.S_IMODE(private.stat().st_mode) != 0o700 or exchange.exists():
        raise ValueError('Private source copy and fresh exchange required')
    current = selection_path is not None
    if current and not all(path is not None and path.is_file() and path.is_relative_to(scope)
                           for path in (selection_path, host_recovery, payload_bindings)):
        raise ValueError('Qualified full selection and explicit recovery inputs required')
    selected = json.loads(selection_path.read_bytes()) if current else selection()
    run_ids = selected['roots']['runs'] if current else RUN_IDS
    budget = json.loads(space_preflight.read_bytes()) if space_preflight else None
    if budget:
        sys.path.insert(0, str(SOURCE))
        from project_store import MAX_TOTAL_BYTES
        if budget['selected_record_bytes'] > MAX_TOTAL_BYTES:
            raise ValueError('Qualified original records exceed the native aggregate project budget')
    if current and (budget is None or shutil.disk_usage(destination.parent).free
                    < budget['estimated_peak_required_bytes']):
        raise ValueError('Full native extraction space reserve is unavailable')
    if (cluster / 'postmaster.pid').exists():
        raise ValueError('Copied SQL owner must be confirmed stopped before startup')
    exchange.mkdir(mode=0o700)
    socket = Path(tempfile.mkdtemp(prefix='rheon-history-owner-', dir='/tmp'))
    dsn = f'dbname=postgres host={socket} port=56975'
    environment = dict(os.environ, TULDOK_DATABASE_URL=dsn, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # These are explicit operator bindings for the disposable copy, not authority
    # inferred from the historical paths stored inside imported records.
    roots = {
        'current-lab': ROOT,
        'historical': ROOT / 'runs/dependency-migration-v1/domain-original-stage-01',
        'lab': ROOT / 'runs/dependency-migration-v1/domain-original-stage-01/external/lab',
        'pressure': ROOT,
        'long-training': Path('/tmp/rheon-shared-long-training-candidate-01'),
        **HOST_ROOTS,
    }
    if current:
        from preview_portfolio_history import ROOTS
        roots = {alias: path for alias, path in ROOTS.items() if path.is_dir()}
    elif not all(path.is_dir() for path in roots.values()):
        raise ValueError('Explicit copy root binding is unavailable')
    os.umask(0o077)
    value = dict(schema='rheon.native-history-extraction-rehearsal.v1', status='starting_copy_owner',
                 source_kind='isolated_partial_filesystem_copy_with_exact_original_SQL_snapshot',
                 original_service_stopped=False, original_database_opened=False,
                 original_catalog_modified=False, copy_owner_started=False,
                 copy_stop_lease_issued=False, project_extracted=False,
                 source_credentials_reused=False, models_executed=0,
                 postgres_stopped=False, source_commit=subprocess.check_output(
                     ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
                 runner_sha256=sha(Path(__file__)), phases=[], created_unix=time.time(),
                 selection_sha256=sha(selection_path) if current else None,
                 selected_roots={name: len(ids) for name, ids in selected['roots'].items()},
                 destination=str(destination.relative_to(ROOT)), request_key=request_key,
                 free_bytes_before=shutil.disk_usage(destination.parent).free,
                 space_preflight_sha256=sha(space_preflight) if budget else None)

    def save():
        pending = report.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(report)

    def phase(name):
        value['status'] = name
        value['phases'].append(name)
        save()
        print(name, flush=True)

    def bind(source):
        return (bind_current_inputs(source, run_ids, host_recovery, payload_bindings)
                if current else bind_required_inputs(source))

    def retain_preview(name, preview):
        # Native reports contain every warning and original declaration. Store
        # them losslessly without a second 370 MB plaintext representation.
        import gzip
        with gzip.open(exchange / (name + '.json.gz'), 'wt', encoding='utf-8', compresslevel=1) as stream:
            json.dump(preview, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        return dict(can_extract=preview['can_extract'], report_sha256=preview['report_sha256'],
                    included_families=dict(Counter(row['family'] for row in preview['included'])),
                    blockers=dict(Counter(row['code'] for row in preview['blockers'])),
                    artifacts=dict(Counter(row['state'] for row in preview['artifacts'])))

    save()
    dataset = lease = None
    backup_fd = destination_fd = None
    stack_log = (exchange / 'python-stacks.log').open('x')
    faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=True)
    def interrupted(signum, frame):
        raise InterruptedError('Copied-owner extraction interrupted')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        with (private / (report.stem + '-postgres-start.log')).open('x') as stream:
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(private / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56975 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
        sys.path.insert(0, str(SOURCE))
        from app import Dataset
        from project_transfer import ProjectTransfer, preview_source
        if current:
            # The copied owner may reconcile local continuations, but may not
            # silently change the original scientific tables during startup.
            from preview_portfolio_history import SNAPSHOT, TABLES
            from backup_metadata import table_state
            from psycopg import sql
            snapshot_summary = json.loads((SNAPSHOT / 'summary.json').read_bytes())
            if sha(SNAPSHOT / 'metadata.json') != snapshot_summary['metadata_sha256']:
                raise ValueError('Private original metadata snapshot changed')
            original = json.loads((SNAPSHOT / 'metadata.json').read_bytes())
        dataset = Dataset(catalog, dsn=dsn, project_roots=roots)
        value['copy_owner_started'] = True
        value['neural_runtime_imported'] = 'torch' in sys.modules
        if value['neural_runtime_imported']:
            raise ValueError('History rehearsal must not import the neural runtime')
        with dataset.lock:
            if current:
                for table in TABLES:
                    cursor = dataset.db.execute(sql.SQL('SELECT * FROM {}').format(sql.Identifier(table)))
                    if table_state([column.name for column in cursor.description], cursor.fetchall()) != original['tables'][table]:
                        value['mismatched_scientific_table'] = table
                        raise ValueError('Copied owner startup changed original scientific rows')
                value['all_original_scientific_tables_exact_after_owner_start'] = True
                value['original_metadata_sha256'] = snapshot_summary['metadata_sha256']
                for family, table, column in (
                        ('runs', 'workspace_runs', 'id'), ('studies', 'model_studies', 'id'),
                        ('graphs', 'workspace_graphs', 'id'),
                        ('model_definitions', 'model_definition_drafts', 'id'),
                        ('model_materials', 'model_materials', 'sha256'),
                        ('recipe_materials', 'workspace_recipe_materials', 'sha256')):
                    identities = sorted(row[0] for row in dataset.db.execute(
                        sql.SQL('SELECT {} FROM {}').format(sql.Identifier(column), sql.Identifier(table))))
                    if selected['roots'][family] != identities:
                        raise ValueError('Full selection omits or repeats original identities')
                del original
            before = {run_id: dataset.db.execute('SELECT document FROM workspace_runs WHERE id=%s',
                                               (run_id,)).fetchone()[0] for run_id in run_ids}
        exchange_result = dataset.access.exchange(dataset.access._local_owner_token())
        token = exchange_result['token']
        backup_fd = os.open(exchange, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        destination_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        with dataset.access.bind(token):
            phase('previewing_copy_with_real_owner')
            with preview_source(dataset) as source:
                value['source_payload_bindings'] = bind(source)
                preview = ProjectTransfer(source).preview(selected)
            value['live_preview'] = retain_preview('live-preview', preview)
            value['live_preview_blockers'] = dict(Counter(row['code'] for row in preview['blockers']))
            if value['live_preview_blockers'] != {'cessation_proof_unavailable': 1}:
                raise ValueError('Selected copy content is not ready for owner Stop')
            del preview
            phase('stopping_copy_and_capturing_native_backup')
            lease = dataset.stop_for_project_extraction(backup_fd, 'stopped-copy')
            value['copy_stop_lease_issued'] = True
            value['backup_receipt'] = lease.backup_receipt
            value['free_bytes_after_backup'] = shutil.disk_usage(destination.parent).free
            if budget and value['free_bytes_after_backup'] < sum(budget[key] for key in (
                    'selected_payload_bytes', 'selected_record_bytes',
                    'target_directory_and_publication_allowance_bytes', 'SQL_projection_allowance_bytes',
                    'minimum_free_reserve_bytes')):
                raise ValueError('Portable publication reserve no longer available after backup')
            bind(lease)
            phase('previewing_native_stopped_copy')
            transfer = ProjectTransfer(lease)
            stopped = transfer.preview(selected)
            value['stopped_preview'] = retain_preview('stopped-preview', stopped)
            value['stopped_preview_can_extract'] = stopped['can_extract']
            value['stopped_preview_blockers'] = dict(Counter(row['code'] for row in stopped['blockers']))
            if not stopped['can_extract']:
                raise ValueError('Stopped-copy native preview is not extractable')
            phase('publishing_new_native_history_project')
            receipt = transfer.extract(selected, stopped['report_sha256'], destination_fd,
                                       destination.name, title, request_key)
            value['receipt'] = receipt
            value['project_extracted'] = bool(receipt['committed'])
            after = {run_id: lease.db.execute('SELECT document FROM workspace_runs WHERE id=%s',
                                            (run_id,)).fetchone()[0] for run_id in run_ids}
            value['selected_original_Run_documents_unchanged'] = before == after
            if before != after:
                raise ValueError('Selected original Run documents changed')
            phase('completed_real_owner_copy_extraction')
    except BaseException as error:
        value.update(status='failed_or_incomplete_owner_rehearsal', error_type=type(error).__name__,
                     native_error_code=getattr(error, 'code', None),
                     error_frames=[dict(file=Path(frame.filename).name, line=frame.lineno,
                                        function=frame.name) for frame in traceback.extract_tb(error.__traceback__)])
        if type(error).__name__ in ('ProjectStoreError', 'WorkbenchError'):
            value['native_error_message'] = str(error)
        stage = getattr(error, 'staging_directory', None)
        if isinstance(stage, str) and stage.startswith('.tuldok-extraction-') and '/' not in stage:
            value['unpublished_staging_directory'] = str(destination.parent / stage)
        if getattr(error, 'receipt', None) is not None:
            value['publication_reconciliation_receipt'] = error.receipt
        raise
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        # Diagnostic samples must not acquire SIGUSR1's terminating default
        # while a terminal failure is draining its owned database.
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        faulthandler.unregister(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        stack_log.close()
        if lease is not None:
            lease.close()
        if dataset is not None:
            dataset.close()
        for descriptor in (backup_fd, destination_fd):
            if descriptor is not None:
                os.close(descriptor)
        stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-m', 'fast', '-w',
                                  '-t', '30', 'stop'], env=environment,
                                 capture_output=True, text=True, timeout=45)
        (private / (report.stem + '-postgres-stop.log')).write_text(stopped.stdout + stopped.stderr)
        value['postgres_stopped'] = stopped.returncode == 0 or 'no server running' in stopped.stdout + stopped.stderr
        value['free_bytes_final'] = shutil.disk_usage(destination.parent).free
        if value['postgres_stopped']:
            shutil.rmtree(socket)
        save()
    print(json.dumps({k: v for k, v in value.items()
                      if k not in ('host_sources', 'source_payload_bindings', 'receipt', 'backup_receipt')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehearsal', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--host-recovery', type=Path)
    parser.add_argument('--payload-bindings', type=Path)
    parser.add_argument('--space-preflight', type=Path)
    parser.add_argument('--title', default='Rheon C106/C107 history rehearsal')
    parser.add_argument('--request-key', default='rheon-c106-c107-history-rehearsal-01')
    args = parser.parse_args()
    try:
        run(args.rehearsal.resolve(), args.destination.resolve(), args.report.resolve(),
            selection_path=args.selection.resolve() if args.selection else None,
            host_recovery=args.host_recovery.resolve() if args.host_recovery else None,
            payload_bindings=args.payload_bindings.resolve() if args.payload_bindings else None,
            space_preflight=args.space_preflight.resolve() if args.space_preflight else None,
            title=args.title, request_key=args.request_key)
    except BaseException as error:
        print('Native history rehearsal failed:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
