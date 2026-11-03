"""Root-gated original-endpoint readback. Defaults to local validation only.

Never launches a Tuldok service, logs authority, or admits a Run. Live execution
requires exact completed-machine, structural-switch and current-launch witnesses.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
MIGRATION = ROOT / 'runs/dependency-migration-v1'
ENDPOINT = 'http://127.0.0.1:8091'
CATALOG = 'a7da4a8de53541a2a0e2aad08005bb71'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= 8 * 1024**2,
            'Gate must be a bounded ordinary file')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Gate hash differs: ' + path.name)
    return json.loads(raw)


def inspect_node(path):
    """Select a measured operator runtime before config or authority is read."""
    path = Path(path)
    require(path.is_absolute() and stat.S_ISREG(path.lstat().st_mode)
            and os.access(path, os.X_OK), 'Choose an absolute ordinary Node executable')
    before = sha(path)
    result = subprocess.run([str(path), '-e', 'process.stdout.write(process.versions.node)'],
        capture_output=True, timeout=5,
        env={'PATH': str(path.parent) + ':/usr/bin:/bin'})
    require(result.returncode == 0 and len(result.stdout) <= 128 and not result.stderr,
            'Selected Node version inspection failed')
    version = result.stdout.decode('ascii')
    parts = version.split('.')
    require(len(parts) == 3 and all(part.isdigit() for part in parts)
            and int(parts[0]) >= 20, 'Playwright requires explicitly selected Node 20 or later')
    require(sha(path) == before, 'Selected Node executable changed during inspection')
    return {'path': str(path), 'version': version, 'sha256': before}


def validate_documents(cutover, launch, machine, provenance, manifest_sha, app_sha):
    """Pure gate checks. This performs no original endpoint/process/catalog access."""
    require(cutover.get('schema') == 'rheon-tuldok-migration-cutover-v1'
            and cutover.get('m3_service_switched') is True
            and cutover.get('m3_complete') is False
            and cutover.get('ordinary_admissions_open') is False
            and cutover.get('smoke_authorized') == {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1}
            and cutover.get('endpoint') == ENDPOINT and cutover.get('catalog_id') == CATALOG,
            'Root structural switch with ordinary admission closed is required')
    require(launch.get('status') == 'current_service_launched_smoke_pending'
            and launch.get('endpoint') == ENDPOINT and launch.get('catalog_id') == CATALOG
            and launch.get('endpoint_owned_by_selected_pid') is True
            and launch.get('structural_service_switch_only') is True
            and launch.get('deployment_manifest_sha256') == manifest_sha
            and launch.get('app_sha256') == app_sha,
            'Actual current deployment launch is required')
    require(machine.get('format') == 'rheon_typed_pressure_qualification_v1'
            and machine.get('status') == 'machine_qualified_browser_pending'
            and machine.get('endpoint') == ENDPOINT and machine.get('catalog_id') == CATALOG
            and machine.get('optimizer_updates_executed') == 1
            and machine.get('restore_optimizer_updates') == 0
            and machine.get('ordinary_restore_bound_profile') is False
            and machine.get('scientific_acceptance') == 'not_asserted'
            and machine.get('model_promotion') is False,
            'Completed single CUDA-update and ordinary zero-update CPU machine receipt required')
    probe = machine.get('probe', {})
    require(probe.get('status') == 'exact' and probe.get('fresh_process') is True
            and probe.get('cuda_initialized') is False and probe.get('optimizer_updates') == 0
            and probe.get('field_api_cli_exact') is True,
            'Exact fresh CPU pressure/API/CLI parity required')
    require(machine['model']['catalog_id'] == CATALOG
            and machine['trial']['material']['catalog_id'] == CATALOG
            and machine['composition_sha256'] == machine['trial']['material']['sha256']
            and machine['browser']['run_id'] == machine['training_run_id']
            and machine['browser']['checkpoint_sha256'] == machine['checkpoint_document_sha256']
            and machine['training_run_id'] != machine['restore_run_id']
            and machine['source_verification']['tuldok_source']['content_sha256'] == launch['app_source_sha256'],
            'Machine Model/Trial/composition/checkpoint/launch identity differs')
    require(provenance.get('identity_relation') == 'new_current_project'
            and len(provenance.get('project_id', '')) == 32
            and len(provenance.get('current_manifest_sha256', '')) == 64,
            'Explicit selected current Project provenance required')


def verify_process(launch):
    expected = launch['process']
    directory = Path('/proc') / str(expected['pid'])
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    command = [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
    require(int(fields[19]) == expected['start_ticks'] and fields[0] != 'Z'
            and command == expected['command'], 'Selected current launch process identity changed')


def prepare(args):
    cutover = checked(args.cutover_receipt, args.cutover_sha256)
    launch = checked(args.launch_receipt, args.launch_sha256)
    machine = checked(args.machine_receipt, args.machine_sha256)
    provenance = checked(args.current_project_provenance, args.provenance_sha256)
    application = args.application.resolve(strict=True)
    require(application.is_relative_to(MIGRATION / 'Tuldok-deployed'), 'Explicit clean deployment required')
    spec = importlib.util.spec_from_file_location('live_readback_deployment', MIGRATION / 'lab/experiments/dependency-migration-v1/stage_tuldok.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest, manifest_sha = builder.verify(application)
    app_sha = sha(application / 'app.py')
    validate_documents(cutover, launch, machine, provenance, manifest_sha, app_sha)
    require(machine['m3'] == {'path': str(args.cutover_receipt.resolve()), 'sha256': args.cutover_sha256}
            and machine['app_launch'] == {'path': str(args.launch_receipt.resolve()), 'sha256': args.launch_sha256},
            'Completed machine receipt is bound to other cutover/launch witnesses')
    require(Path(launch['deployment']).resolve() == application, 'Launch uses another application directory')
    project_root = Path(provenance['root']).resolve(strict=True)
    require(project_root == ROOT and Path(launch['project_roots'][args.project_alias]).resolve() == ROOT,
            'Selected live Project root differs from actual launch')
    relative = Path(provenance['manifest_path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Project manifest must be root-relative')
    project_path = ROOT / relative
    require(sha(project_path) == provenance['current_manifest_sha256'], 'Selected current Project changed')
    project = json.loads(project_path.read_bytes())
    require(project['id'] == provenance['project_id'] and project['schema_version'] == 2,
            'Selected current Project ID/schema differs')
    required_evidence = ['dependency-migration-plan', 'dependency-migration-results',
                         'dependency-migration-scientific-parity']
    require(all(any(row['id'] == reference for row in project['evidence'])
                for reference in required_evidence), 'Selected Project lacks migration evidence references')
    selected = next(row for row in project['data'] if row['id'] == args.import_reference)
    import_path = (ROOT / selected['path']).resolve(strict=True)
    require(import_path.is_relative_to(ROOT), 'Selected field source escapes current root')
    import_sha = sha(import_path)
    require(import_sha == next(row['measured_sha256'] for row in provenance['references']
                               if row['group'] == 'data' and row['id'] == args.import_reference),
            'A5 source bytes differ from selected current Project provenance')
    require(bool(args.existing_record) != bool(args.allow_project_import),
            'Choose an existing exact field record or explicitly allow one A5 Project import')
    sys.path.insert(0, str(application))
    import agent_api
    requests = {}
    selections = {
        'model': ('models.get', {'model_id': machine['model']['id']}, 1),
        'study': ('studies.get', {'study_id': machine['trial']['study_id']}, 1),
        'trial': ('studies.trial', {'trial_id': machine['trial']['id']}, 1),
        'train': ('runs.get', {'run_id': machine['training_run_id']}, 1),
        'restore': ('runs.get', {'run_id': machine['restore_run_id']}, 1),
        'evidence': ('runs.model_evidence', {'run_id': machine['training_run_id'], 'split': 'validation'}, 2),
        'field': ('runs.model_evidence', {'run_id': machine['training_run_id'], 'split': 'validation',
            **{key: machine['browser'][key] for key in ('checkpoint_sha256', 'member_ordinal', 'field_id')},
            'axis': 'z', 'index': 0}, 2)}
    for key, (operation, arguments, version) in selections.items():
        agent_api.validate_arguments(operation, arguments, version)
        method, path, body = agent_api.project_request(operation, arguments, version)
        contract = agent_api.contract(operation, version)
        require(contract['effects'] == 'read' and (method == 'GET' and body is None
                or method == 'POST' and operation == 'runs.model_evidence' and version == 2),
                'Readback may only generate current read operations')
        requests[key] = {'method': method, 'body': body, 'path': path, 'operation': operation,
                         'version': version, 'sha256': contract['sha256']}
    return {'format': 'rheon_live_browser_readback_config_v1', 'endpoint': ENDPOINT, 'catalog_id': CATALOG,
        'root_trigger': bool(args.execute_after_m3), 'application': str(application),
        'deployment_manifest_sha256': manifest_sha, 'app_sha256': app_sha,
        'gates': [{'path': str(path.resolve()), 'sha256': digest} for path, digest in (
            (args.cutover_receipt, args.cutover_sha256), (args.launch_receipt, args.launch_sha256),
            (args.machine_receipt, args.machine_sha256), (args.current_project_provenance, args.provenance_sha256))],
        'machine': machine, 'provenance': provenance, 'requests': requests,
        'required_evidence': required_evidence,
        'project_alias': args.project_alias, 'import_reference': args.import_reference,
        'import_sha256': import_sha, 'allow_project_import': args.allow_project_import,
        'existing_record': args.existing_record, 'owner_session_file': str(args.owner_session_file.resolve()),
        'chromium': '/home/jeremy/.cache/ms-playwright/chromium_headless_shell-1246/chrome-headless-shell-linux64/chrome-headless-shell',
        'launch': launch, 'output': str(args.output.resolve())}


def child_subreaper(value=None):
    """Linux ownership: adopt detached browser descendants if Node exits first."""
    library = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if library.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot inspect child ownership')
    if value is not None and library.prctl(36, int(value), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot establish child ownership')
    return previous.value


def direct_children():
    raw = (Path('/proc/self/task') / str(os.getpid()) / 'children').read_bytes()
    require(len(raw) <= 65536, 'Owned child inventory exceeds bound')
    return {int(value) for value in raw.split()}


def cleanup_children(child, baseline, timeout=15):
    """Kill/reap only the launched handle and its newly adopted descendants.

    This dedicated launcher spawns no unrelated work. Existing direct children
    are excluded. An unreaped direct child cannot have its PID reused, so each
    signal remains bound to this launcher's current child ownership.
    """
    deadline = time.monotonic() + timeout
    errors = []
    reaped = False
    adopted = []
    if child is not None:
        if child.returncode is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append(type(error).__name__)
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
            reaped = True
        except BaseException as error:
            errors.append(type(error).__name__)
    while time.monotonic() < deadline:
        pending = direct_children() - baseline
        if child is not None:
            pending.discard(child.pid)
        if not pending:
            return reaped, adopted, errors
        for pid in pending:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                result, status = os.waitpid(pid, os.WNOHANG)
                if result:
                    adopted.append({'pid': pid, 'wait_status': status})
            except ChildProcessError:
                pass
        time.sleep(0.01)
    errors.append('DescendantCleanupDeadline')
    return reaped, adopted, errors


def run_browser(command, output, timeout=240, node_identity=None):
    """Own the launched session until exit, including interrupted/failed waits.

    After dispatch, failure always requires inspection: killing this client does
    not roll back a collection import already received by the HTTP owner.
    """
    started = time.monotonic()
    child = None
    returncode = None
    failure = None
    cleanup_error = None
    group_cleanup_attempted = False
    reaped = False
    adopted = []
    baseline = direct_children()
    prior_subreaper = None
    prior_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError('Owned browser launcher interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        prior_subreaper = child_subreaper(True)
        child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
        returncode = child.wait(timeout=timeout)
        reaped = True
        if returncode != 0:
            failure = 'NonzeroExit'
    except BaseException as error:
        # Never copy protocol exception text: it could contain private authority.
        failure = type(error).__name__
    finally:
        # Cleanup cannot be interrupted by another terminal signal. SIGKILL or
        # host loss cannot produce a receipt; the pre-dispatch import claim still
        # blocks automatic replay in that case.
        prior_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if child is not None or prior_subreaper is not None:
                group_cleanup_attempted = True
                try:
                    reaped, adopted, errors = cleanup_children(child, baseline)
                    if child is not None:
                        returncode = child.returncode
                    if errors:
                        cleanup_error = errors
                except BaseException as error:
                    cleanup_error = type(error).__name__
            if cleanup_error is not None and failure is None:
                failure = 'CleanupIncomplete'
            summary = {
                'status': 'passed' if failure is None and returncode == 0
                          else 'stopped_inspect_receipt_before_retry',
                'returncode': returncode, 'timed_out': failure == 'TimeoutExpired',
                'error_kind': failure, 'cleanup_error_kind': cleanup_error,
                'owned_process_group_cleanup_attempted': group_cleanup_attempted,
                'owned_child_reaped': reaped, 'elapsed_s': time.monotonic() - started,
                'detached_descendants_reaped': adopted,
                'node_executable': node_identity,
                'endpoint': ENDPOINT, 'catalog_id': CATALOG,
                'original_service_started_or_stopped': False,
                'import_dispatch_may_have_committed': (output / 'import-intent.json').exists(),
                'automatic_retry': False, 'authority_logged': False,
                'receipt': str(output / 'browser-receipt.json')}
            descriptor = os.open(output / 'launcher-receipt.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(summary, stream, indent=2)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                if prior_subreaper is not None:
                    child_subreaper(prior_subreaper)
            finally:
                signal.signal(signal.SIGINT, prior_int)
                signal.signal(signal.SIGTERM, prior_term)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-m3', action='store_true')
    parser.add_argument('--node-executable', type=Path, required=True)
    parser.add_argument('--application', type=Path, required=True)
    for prefix in ('cutover', 'launch', 'machine'):
        parser.add_argument('--' + prefix + '-receipt', type=Path, required=True)
        parser.add_argument('--' + prefix + '-sha256', required=True)
    parser.add_argument('--current-project-provenance', type=Path, required=True)
    parser.add_argument('--provenance-sha256', required=True)
    parser.add_argument('--project-alias', default='current-lab')
    parser.add_argument('--import-reference', required=True)
    parser.add_argument('--allow-project-import', action='store_true')
    parser.add_argument('--existing-record')
    parser.add_argument('--owner-session-file', type=Path, default=OPS / 'migration-owner-session.private.txt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    node = inspect_node(args.node_executable)
    config = prepare(args)
    config['node_executable'] = node
    if not args.execute_after_m3:
        print(json.dumps({'status': 'locally_validated_network_not_executed', 'endpoint': ENDPOINT,
                          'project_id': config['provenance']['project_id'], 'original_service_accessed': False}))
        return
    verify_process(config['launch'])
    output = args.output.resolve()
    require(output.is_relative_to(MIGRATION) and not output.exists(), 'Choose a fresh task-owned output')
    output.mkdir(mode=0o700)
    config_path = output / 'config.private.json'
    with os.fdopen(os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(config, stream, indent=2)
    require(sha(args.node_executable) == node['sha256'], 'Selected Node changed before launch')
    command = [str(args.node_executable), str(OPS / 'live_readback.cjs'), str(config_path)]
    # Two measured slow Run-view loaders (90 s each), initial/final list reads
    # (45 s each), plus Project/evidence/field inspection remain bounded here.
    summary = run_browser(command, output, timeout=360, node_identity=node)
    print(json.dumps(summary))
    require(summary['status'] == 'passed', 'Live readback stopped; inspect its bounded failure receipt, do not replay imports')


if __name__ == '__main__':
    main()
