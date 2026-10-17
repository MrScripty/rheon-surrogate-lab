"""Explicit root-operated M3 steps. Import and --help never access a catalog.

There is no automatic retry, rollback, restart, restore, or admission here. Every
invocation has a fresh durable receipt; an uncertain write requires a separate
read/reconciliation command. Private connection/session values never enter receipts.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/dependency-migration-v1'
OPS = BASE / 'ops'
SOURCE = BASE / 'Tuldok'
PYTHON = BASE / 'app-env/bin/python'
BACKUP = BASE / 'original-schema8-backup-02'
PID = 438962
TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def persist(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private(path):
    info = path.stat()
    require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0 and not path.is_symlink(),
            'Owned private nonsymlink selection/session file required')
    return path.read_text()


def selected():
    value = json.loads(private(OPS / 'live-selection.private.json'))
    original = read(OPS / 'preflight.json')
    require(original['dataset_id'] == 'a7da4a8de53541a2a0e2aad08005bb71', 'Original catalog ID differs')
    require(str(original['catalog']) in value['args'] and '--port' in value['args']
            and value['args'][value['args'].index('--port') + 1] == '8091', 'Original command selection differs')
    return value, original


def port_open():
    with socket.socket() as stream:
        return stream.connect_ex(('127.0.0.1', 8091)) == 0


def owns_endpoint(pid):
    inodes = set()
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        row = line.split()
        address, port = row[1].split(':')
        if int(port, 16) == 8091 and address in ('0100007F', '00000000') and row[3] == '0A':
            inodes.add(row[9])
    sockets = set()
    for fd in Path('/proc', str(pid), 'fd').iterdir():
        try:
            target = os.readlink(fd)
            if target.startswith('socket:['):
                sockets.add(target[8:-1])
        except FileNotFoundError:
            pass
    return bool(inodes) and inodes <= sockets


def process(pid):
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(stat[19]), 'state': stat[0],
            'command': [x.decode() for x in (path / 'cmdline').read_bytes().split(b'\0') if x],
            'cwd': str((path / 'cwd').resolve(strict=True)),
            'executable': str((path / 'exe').resolve(strict=True)),
            'children': sorted({int(child) for thread in (path / 'task').iterdir()
                                for child in (thread / 'children').read_text().split()})}


def original_absent(value):
    require(not Path('/proc', str(PID)).exists(), 'Exact original PID remains; keep writers stopped')
    require(not port_open(), 'Original endpoint still accepts connections')
    expected = b'\0'.join(x.encode() for x in value['args']) + b'\0'
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                require((path / 'cmdline').read_bytes() != expected, 'Original command remains active')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass


def interrupt_exact(pid, expected):
    # pidfd pins the observed task: PID reuse between the final identity check
    # and delivery must never redirect this signal to another process.
    descriptor = os.pidfd_open(pid, 0)
    try:
        require(process(pid) == expected, 'Original identity changed before signal')
        signal.pidfd_send_signal(descriptor, signal.SIGINT, None, 0)
    finally:
        os.close(descriptor)


def source_witness():
    helper = load_module('m3_dependency_config', BASE / 'lab/experiments/dependency-migration-v1/dependency_config.py')
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']).decode().strip()
    result = helper.source_identity(SOURCE, revision)
    return {key: result[key] for key in ('revision', 'tree', 'content_sha256')}


def gates(args):
    require(args.gates and args.gates_sha256 and sha(args.gates) == args.gates_sha256,
            'Root-supplied exact gate-manifest hash required')
    document = read(args.gates)
    require(document['format'] == 'rheon_m3_root_gates_v1', 'Unknown gate manifest')
    require(document['tuldok_source'] == source_witness(), 'Current Tuldok source differs from gates')
    required = {'q2_independent', 'q3_pipeline', 'q8_lifecycle', 'q6_browser', 'deployment_review'}
    require(required <= set(document['receipts']),
            'Required Tuldok Q2/Q3/Q6/Q8 and independent deployment-review gates missing')
    verified = {}
    for name, row in document['receipts'].items():
        path = Path(row['path']).resolve(strict=True)
        require(sha(path) == row['sha256'] and row['checks'], 'Gate receipt hash/checks missing: ' + name)
        value = read(path)
        for check in row['checks']:
            observed = value
            for key in check['path']:
                observed = observed[key]
            require(set(check) in ({'path', 'equals'}, {'path', 'length'}), 'Unknown receipt predicate')
            require(observed == check['equals'] if 'equals' in check else len(observed) == check['length'],
                    'Qualification gate failed: ' + name)
        if name == 'q7_full74':
            require(any(check.get('equals') == 74 or check.get('length') == 74 for check in row['checks']),
                    'Full Q7 gate must explicitly prove all 74 cases')
        verified[name] = {'path': str(path), 'sha256': row['sha256'], 'checks': row['checks']}
    for name, row in document['artifacts'].items():
        require(sha(row['path']) == row['sha256'], 'Gate artifact changed: ' + name)
    require({'deployment_manifest', 'rheon_binary', 'schema_rehearsal'} <= set(document['artifacts']),
            'Deployment/binary/schema-rehearsal bindings missing')
    return document, {'path': str(args.gates.resolve()), 'sha256': args.gates_sha256, 'receipts': verified}


def metadata(value, original, *, any_schema=False):
    from metadata import Metadata
    db = Metadata(Path(original['catalog']), dsn=value['dsn'], create=False,
                  schema_version=None if any_schema else 10)
    require(db.dataset_id == original['dataset_id'], 'Original target catalog ID differs')
    return db


def terminal_inventory(value, original):
    db = metadata(value, original, any_schema=True)
    try:
        db.connection.execute('SET default_transaction_read_only = on')
        result = {}
        for table, column, key in (('workspace_runs', 'document', 'state'),
                                   ('generation_jobs', 'data', 'status'), ('grounded_jobs', 'data', 'status')):
            counts = Counter(json.loads(row[0])[key] for row in db.execute('SELECT ' + column + ' FROM ' + table))
            require(set(counts) <= TERMINAL, 'Nonterminal current inventory blocks quiescence: ' + table)
            result[table] = dict(counts)
        return {'schema_version': db.schema_version, 'states': result, 'database_read_only': True}
    finally:
        db.close()


def bookend(args, original, schema_rehearsal, *, fresh=False):
    require(args.bookend and args.bookend_sha256 and sha(args.bookend) == args.bookend_sha256,
            'Exact fresh stopped-original revalidation receipt required')
    value = read(args.bookend)
    binding = value.get('schema_rehearsal_receipt')
    require(type(binding) is dict and set(binding) == {'path', 'sha256'}
            and Path(binding['path']).resolve(strict=True) == Path(schema_rehearsal['path']).resolve(strict=True)
            and binding['sha256'] == schema_rehearsal['sha256'] == sha(schema_rehearsal['path']),
            'Stopped-original schema rehearsal receipt binding differs from root gates')
    require(value['schema'] == 'rheon-fresh-stopped-original-bookend-v1'
            and value['status'] == 'exact_backup_revalidated' and value['schema_version'] == 8
            and value['dataset_id'] == original['dataset_id'] and value['stopped_original_pid'] == PID
            and Path(value['complete_backup']).resolve() == BACKUP.resolve()
            and value['backup_receipt_sha256'] == sha(BACKUP / 'backup-receipt.json'),
            'Stopped-original receipt does not bind exact backup02')
    for key in ('all_domain_tables_equal', 'all_sequences_equal', 'all_managed_hashes_equal',
                'all_file_and_directory_modes_equal', 'metadata_pre_post_stable',
                'original_port_closed', 'original_source_command_absent'):
        require(value[key] is True, 'Incomplete stopped-original proof: ' + key)
    if fresh:
        age = time.time() - args.bookend.stat().st_mtime
        require(0 <= age <= args.bookend_max_age, 'Stopped-original bookend is stale')
    return {'path': str(args.bookend.resolve()), 'sha256': args.bookend_sha256}


def stage_spec(args, db):
    from source_inventory import converter
    from metadata import definition_hash
    owner = converter()
    selection = read(OPS / 'source-selection02.json')
    require(selection['archive_receipt_sha256'] == sha(BACKUP / 'backup-receipt.json')
            and selection['dataset_id'] == db.dataset_id, 'Selection02 archive binding differs')
    specification = dict(selection, target_definition_sha256=definition_hash(),
                         target_connection_sha256=owner.target_identity(db),
                         target_application_sha256=owner.application_identity())
    return owner, specification


def schema_inventory(db):
    from backup_metadata import read_catalog, read_sequences, summaries, tree_manifest
    saved = read(BACKUP / 'backup-receipt.json')
    observed = summaries(read_catalog(db))
    for name, expected in saved['tables'].items():
        if name != 'catalog_version':
            require(observed[name] == expected, 'Schema upgrade changed original domain table: ' + name)
    require(all(row['count'] == 0 for name, row in observed.items() if name not in saved['tables']),
            'New schema tables must remain empty before domain publication')
    require(read_sequences(db) == saved['sequences'], 'Schema upgrade changed original sequences')
    require(tree_manifest(db.path) == saved['files'], 'Schema upgrade changed original managed files')
    return {'all_original_domain_tables_equal': True, 'all_sequences_equal': True,
            'all_managed_files_equal': True, 'new_tables_empty': True,
            'managed_entry_count': len(saved['files'])}


def require_prepared(args, db):
    require(args.stage and args.spec and args.prepared_sha256, 'Explicit stage, spec, prepared hash required')
    owner, expected = stage_spec(args, db)
    prepared = read(args.stage / 'prepared.json')
    # Converter identity is its canonical document digest, not the file's
    # newline/pretty-print byte hash. Its owner verifies the entire stage again.
    require(owner.digest(prepared) == args.prepared_sha256, 'Prepared document changed')
    require(read(args.spec) == prepared['spec'] == expected, 'Prepared target/source/application binding differs')
    return owner, prepared


def verify_conversion(owner, db, args, token):
    # Access.bind can audit a rejected credential before publish opens its own
    # transaction. Enforce database read-only before any credential evaluation.
    db.connection.execute('SET default_transaction_read_only = on')
    return owner.publish(db, args.stage, args.prepared_sha256, quiesced=True,
                         token=token, verify_only=True)


def cold_verify(db, prepared, token):
    from access import Access
    from projects import Projects, TEXT_LIMIT
    from runs import Runs
    db.connection.execute('SET default_transaction_read_only = on')
    access = Access(db, db.path)
    projects = Projects(SimpleNamespace(db=db, lock=db.lock, access=access), prepared['roots'])
    runs = Runs.__new__(Runs)
    runs.db, runs.lock, runs.access, runs.path = db, db.lock, access, db.path / 'workspace-runs'
    counts = {'projects': 0, 'evidence_previews': 0, 'runs': 0, 'artifacts': 0, 'artifact_bytes': 0}
    try:
        with access.bind(token):
            for saved in prepared['projects']:
                value = projects.inspect_snapshot(saved['id'])
                require(value['manifest_status'] == 'unchanged' and value['manifest_sha256'] == saved['manifest_sha256'],
                        'Current converted Project changed')
                for group in ('data', 'recipes', 'evidence'):
                    for row in value[group]:
                        require(row['status'] == 'available' and row.get('integrity') != 'mismatch', 'Converted reference unavailable')
                        if group == 'evidence' and (saved['root'] == 'historical' or row.get('bytes', 0) <= TEXT_LIMIT):
                            evidence = projects.evidence(saved['id'], row['id'])
                            require(evidence['external'] and evidence['scientific_verification'] == 'not_performed', 'Historical fact overclaimed')
                            if row.get('sha256'):
                                require(evidence['sha256'] == row['sha256'], 'Historical evidence changed')
                            counts['evidence_previews'] += 1
                counts['projects'] += 1
            for (run_id,) in db.execute('SELECT id FROM workspace_runs ORDER BY id'):
                value = runs.get(run_id)
                require(value['state'] in TERMINAL | {'unknown'}, 'Current Run is not terminal')
                if value['state'] == 'completed':
                    for item in value['artifacts']:
                        raw, _media = runs.artifact(run_id, item['name'])
                        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'Run artifact changed')
                        counts['artifacts'] += 1
                        counts['artifact_bytes'] += len(raw)
                counts['runs'] += 1
        return dict(counts, database_read_only=True, Dataset_constructed=False, workers_constructed=False)
    finally:
        projects.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'upgrade', 'reconcile-schema', 'prepare', 'publish',
                                        'reconcile-domain', 'verify-current', 'launch'))
    for name in ('receipt', 'gates', 'bookend', 'stage', 'spec', 'deployment', 'owner-session-file'):
        parser.add_argument('--' + name, type=Path, required=name == 'receipt',
                            default=OPS / 'migration-owner-session.private.txt' if name == 'owner-session-file' else None)
    for name in ('gates-sha256', 'bookend-sha256', 'prepared-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--bookend-max-age', type=int, default=600)
    parser.add_argument('--quiesce-original', action='store_true')
    parser.add_argument('--upgrade-original', action='store_true')
    parser.add_argument('--publish-original', action='store_true')
    parser.add_argument('--launch-current', action='store_true')
    args = parser.parse_args(argv)
    for flag, mode in ((args.quiesce_original, 'preflight'), (args.upgrade_original, 'upgrade'),
                       (args.publish_original, 'publish'), (args.launch_current, 'launch')):
        require(not flag or args.mode == mode, 'Action flag does not match selected mode')
    require(args.receipt.resolve().is_relative_to(OPS) and not args.receipt.exists(), 'Fresh receipt under ops required')
    # No application import or DB connection until the explicit mode is parsed.
    sys.path[:0] = [str(SOURCE), str(OPS)]
    value, original = selected()
    started = time.monotonic()
    record = {'schema': 'rheon-m3-orchestrator-v1', 'mode': args.mode, 'status': 'intent',
              'catalog': original['catalog'], 'catalog_id': original['dataset_id'],
              'original_pid': PID, 'script_sha256': sha(Path(__file__)), 'automatic_retry': False,
              'recovery': 'Keep all writers stopped. Inspect this receipt and durable state; reconcile explicitly. Never restore original in place.'}
    # Exclusive durable intent prevents a command restart from replaying writes.
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(record)); stream.flush(); os.fsync(stream.fileno())
    db = None
    try:
        if args.mode == 'preflight':
            observed = process(PID)
            require(observed['command'] == value['args'] and observed['cwd'] == value['cwd']
                    and not observed['children'], 'Original identity/descendants differ')
            environ = dict(part.split(b'=', 1) for part in Path('/proc', str(PID), 'environ').read_bytes().split(b'\0') if b'=' in part)
            require(environ.get(b'TULDOK_DATABASE_URL', b'').decode() == value['dsn'], 'Original process connection differs')
            record.update(process=observed, terminal_inventory=terminal_inventory(value, original), port_open=port_open())
            require(record['port_open'] and owns_endpoint(PID), 'Original selected PID does not own endpoint')
            record['endpoint_owned_by_selected_pid'] = True
            if args.quiesce_original:
                _document, record['gates'] = gates(args)
                require(process(PID) == observed, 'Original identity changed before signal')
                record['write_boundary'] = 'quiesce_signal_dispatch'
                persist(args.receipt, record)
                interrupt_exact(PID, observed)
                deadline = time.monotonic() + 30
                while Path('/proc', str(PID)).exists() and time.monotonic() < deadline:
                    time.sleep(0.1)
                original_absent(value)
                record.update(status='original_quiesced', original_pid_absent=True, original_port_closed=True,
                              original_command_absent=True, terminal_inventory_after=terminal_inventory(value, original))
            else:
                record['status'] = 'preflight_read_only_passed'
        else:
            document, record['gates'] = gates(args)
            original_absent(value)
            record['bookend'] = bookend(args, original, document['artifacts']['schema_rehearsal'],
                                        fresh=args.mode == 'upgrade')
            if args.mode == 'upgrade':
                require(args.upgrade_original, 'Schema writes require explicit --upgrade-original')
                from upgrade_metadata import upgrade_catalog
                db = metadata(value, original, any_schema=True)
                require(db.schema_version == 8, 'Upgrade requires known original schema8; use reconcile-schema after uncertainty')
                db.close(); db = None
                record['write_boundary'] = 'schema_upgrade_dispatch'
                persist(args.receipt, record)
                first = upgrade_catalog(original['catalog'], dsn=value['dsn'], quiesced=True)
                require(first['status'] == 'upgraded', 'Schema owner did not report upgraded')
                reentry = upgrade_catalog(original['catalog'], dsn=value['dsn'], quiesced=True)
                require(reentry['status'] == 'already_current', 'Schema reentry failed')
                db = metadata(value, original)
                db.connection.execute('SET default_transaction_read_only = on')
                record.update(status='schema10_verified_reentry', upgrade=first, reentry=reentry,
                              original_inventory=schema_inventory(db))
            else:
                db = metadata(value, original, any_schema=args.mode == 'reconcile-schema')
                if args.mode == 'reconcile-schema':
                    db.connection.execute('SET default_transaction_read_only = on')
                    db.validate_schema()
                    require(db.schema_version in (8, 10), 'Unexpected intermediate schema; stop for owner review')
                    record.update(status='schema_state_observed_read_only', schema_version=db.schema_version,
                                  schema_signature=db._signature(), domain_publication_not_replayed=True)
                elif args.mode == 'prepare':
                    require(args.stage and args.spec and args.stage.resolve().is_relative_to(BASE)
                            and args.spec.resolve().is_relative_to(OPS) and not args.stage.exists()
                            and not args.spec.exists(), 'Fresh absent stage/spec under migration/ops required')
                    owner, specification = stage_spec(args, db)
                    with args.spec.open('xb') as stream:
                        stream.write(canonical(specification))
                    db.connection.execute('SET default_transaction_read_only = on')
                    prepared_hash = owner.prepare(db, BACKUP, args.stage, specification, quiesced=True)
                    record.update(status='prepared_read_only', prepared_sha256=prepared_hash,
                                  stage=str(args.stage.resolve()), spec=str(args.spec.resolve()), authoritative_db_writes=0)
                else:
                    owner, prepared = require_prepared(args, db)
                    token = private(args.owner_session_file).strip()
                    require(32 <= len(token) <= 256, 'Owner session format differs')
                    if args.mode in ('publish', 'reconcile-domain'):
                        require(args.mode != 'publish' or args.publish_original, 'Publication requires explicit --publish-original')
                        record['write_boundary'] = 'domain_publish_dispatch' if args.mode == 'publish' else 'domain_verify_only'
                        persist(args.receipt, record)
                        if args.mode == 'reconcile-domain':
                            result = verify_conversion(owner, db, args, token)
                        else:
                            result = owner.publish(db, args.stage, args.prepared_sha256, quiesced=True, token=token)
                        record.update(status=result['status'], result=result, prepared_sha256=args.prepared_sha256)
                    elif args.mode == 'verify-current':
                        proof = verify_conversion(owner, db, args, token)
                        require(proof['status'] == 'verified-completed', 'Domain conversion is not complete')
                        record.update(status='current_read_only_verified', result=cold_verify(db, prepared, token), conversion=proof)
                    else:
                        require(args.launch_current, 'Service start requires explicit --launch-current')
                        proof = verify_conversion(owner, db, args, token)
                        require(proof['status'] == 'verified-completed', 'Domain publication must be verified before launch')
                        require(args.deployment is not None, 'Explicit --deployment required; latest deployment is never selected')
                        deployed = args.deployment.resolve(strict=True)
                        require(deployed.is_relative_to(BASE / 'Tuldok-deployed'), 'Deployment must be an explicit staged app payload')
                        manifest = deployed / 'deployment-manifest.json'
                        require(manifest == Path(document['artifacts']['deployment_manifest']['path']).resolve()
                                and sha(manifest) == document['artifacts']['deployment_manifest']['sha256'], 'Deployment identity differs')
                        deployment = read(manifest)
                        for row in deployment['files']:
                            path = deployed / row['path']
                            require(path.resolve(strict=True).is_relative_to(deployed) and not path.is_symlink()
                                    and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
                                    and stat.S_IMODE(path.stat().st_mode) == row['mode'], 'Deployed payload changed')
                            source_path = SOURCE / row['path']
                            require(source_path.resolve(strict=True).is_relative_to(SOURCE)
                                    and not source_path.is_symlink() and sha(source_path) == row['sha256'],
                                    'Deployment payload differs from selected current source')
                        actual_files = {p.relative_to(deployed).as_posix() for p in deployed.rglob('*') if p.is_file()}
                        require(actual_files == {row['path'] for row in deployment['files']} | {'deployment-manifest.json'}, 'Unexpected deployed files')
                        require(owner.application_identity() == digest({p.name: sha(p) for p in sorted(deployed.glob('*.py'))}), 'Deployed/source application differs')
                        roots = dict(prepared['roots'])
                        require('current-lab' not in roots and 'pressure' not in roots, 'Current root aliases collide')
                        roots.update({'current-lab': str(ROOT), 'pressure': str(ROOT)})
                        command = [str(PYTHON), '-B', str(deployed / 'app.py'), '--data', original['catalog'], '--host', '127.0.0.1', '--port', '8091']
                        for name, path in sorted(roots.items()):
                            command += ['--project-root', name + '=' + str(path)]
                        original_absent(value)
                        db.close(); db = None
                        record.update(status='launch_dispatch_intent', command=command, project_roots=roots)
                        persist(args.receipt, record)
                        env = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG',
                                'LC_ALL', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE',
                                'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY') if key in os.environ}
                        env.update(PATH=str(PYTHON.parent) + ':/usr/bin:/bin',
                                   TULDOK_DATABASE_URL=value['dsn'], PYTHONDONTWRITEBYTECODE='1')
                        log_path = args.receipt.with_suffix('.private.log')
                        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(log_fd, 'ab') as stream:
                            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        record['pid'] = child.pid
                        persist(args.receipt, record)
                        deadline = time.monotonic() + 30
                        while not port_open() and child.poll() is None and time.monotonic() < deadline:
                            time.sleep(0.1)
                        require(child.poll() is None and port_open(), 'Current service did not become ready; inspect exact child/log without restart')
                        observed = process(child.pid)
                        require(observed['command'] == command and owns_endpoint(child.pid), 'Launched process does not own exact command/endpoint')
                        record.update(status='current_service_launched_smoke_pending', endpoint='http://127.0.0.1:8091',
                            process=observed, project_roots=roots, app_sha256=sha(deployed / 'app.py'),
                            app_source_sha256=document['tuldok_source']['content_sha256'],
                            app_source=document['tuldok_source'], deployment=str(deployed),
                            rheon_binary_sha256=document['artifacts']['rheon_binary']['sha256'],
                            deployment_manifest_sha256=sha(manifest),
                            python_executable_sha256=sha(PYTHON.resolve()), ordinary_admissions_authorized=False,
                            endpoint_owned_by_selected_pid=True,
                            structural_service_switch_only=True, scientific_acceptance='not_asserted')
    except BaseException as error:
        record.update(status='stopped_requires_inspection', error_class=type(error).__name__,
                      uncertain_write_or_launch=record.get('write_boundary') in
                      ('schema_upgrade_dispatch', 'domain_publish_dispatch', 'quiesce_signal_dispatch')
                      or record.get('status') == 'launch_dispatch_intent')
        raise
    finally:
        if db is not None:
            db.close()
        record['elapsed_s'] = time.monotonic() - started
        persist(args.receipt, record)
    print(json.dumps({'status': record['status'], 'receipt': str(args.receipt), 'elapsed_s': record['elapsed_s']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Connection exceptions may contain private connection details. The
        # durable receipt records only the class; never print raw exceptions.
        sys.stderr.write('M3 step stopped (' + type(error).__name__ +
                         '); inspect the selected receipt and durable state. No automatic recovery ran.\n')
        sys.exit(1)
