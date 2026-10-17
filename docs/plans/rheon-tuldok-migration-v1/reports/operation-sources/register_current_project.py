"""Root-triggered normal API registration of the exact current Project04.

Preparation reads public local artifacts only. Execution uses the existing owner
session and current registered HTTP contracts; no catalog module/SQL is opened.
There is one registration dispatch and no retry, approval, admission or worker.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MIGRATION = ROOT / 'runs/dependency-migration-v1'
OPS = MIGRATION / 'ops'
APPLICATION = MIGRATION / 'Tuldok-deployed/current-03'
ENDPOINT = 'http://127.0.0.1:8091'
CATALOG = 'a7da4a8de53541a2a0e2aad08005bb71'
PROJECT = '97eae52ac8074b4baccee2c12101ead6'
MANIFEST = 'runs/dependency-migration-v1/current-project-04/project.json'
MANIFEST_SHA = '97a8020cd6cfe39afa426921dc692dc948d0d6475f9a4dfd37dea49f4792b2a0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def checked(path, expected=None):
    """Bounded ordinary public artifact, checked before/after and against its pin."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 8*1024**2, 'bounded regular public artifact required')
        raw = bytearray()
        while len(raw) <= before.st_size:
            block = os.read(descriptor, min(65536, before.st_size + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                and len(raw) == before.st_size, 'public artifact changed during read')
        digest = hashlib.sha256(raw).hexdigest()
        require(expected is None or digest == expected, 'public artifact hash differs')
        return json.loads(raw), digest
    finally:
        os.close(descriptor)


def persist(path, value, *, exclusive=False):
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + '.pending')
        persist(temporary, value, exclusive=True)
        os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def validate(cutover, launch, deployment, deployment_sha, project):
    require(cutover.get('schema') == 'rheon-tuldok-migration-cutover-v1'
            and cutover.get('m3_service_switched') is True
            and cutover.get('m3_complete') is False and cutover.get('ordinary_admissions_open') is False
            and cutover.get('endpoint') == ENDPOINT and cutover.get('catalog_id') == CATALOG,
            'exact structural current-service switch with ordinary admission closed required')
    require(launch.get('status') == 'current_service_launched_smoke_pending'
            and launch.get('endpoint') == ENDPOINT and launch.get('catalog_id') == CATALOG
            and launch.get('endpoint_owned_by_selected_pid') is True
            and launch.get('structural_service_switch_only') is True
            and Path(launch['deployment']).resolve() == APPLICATION
            and launch.get('deployment_manifest_sha256') == deployment_sha,
            'exact clean03 current-service launch required')
    rows = {row['path']: row for row in deployment['files']}
    require(launch.get('app_sha256') == rows['app.py']['sha256'], 'launch application hash differs')
    require(all(Path(launch['project_roots'][alias]).resolve() == ROOT for alias in ('current-lab', 'pressure')),
            'both current-lab and pressure aliases must retain the primary artifact root')
    require(project['id'] == PROJECT and project['schema_version'] == 2,
            'exact current Project04 identity required')


def prepare(args):
    cutover, cutover_sha = checked(args.cutover_receipt, args.cutover_sha256)
    launch, launch_sha = checked(args.launch_receipt, args.launch_sha256)
    require(args.application.resolve(strict=True) == APPLICATION, 'select exact clean03 deployment')
    spec = importlib.util.spec_from_file_location('register_current_stage_owner',
        MIGRATION / 'lab/experiments/dependency-migration-v1/stage_tuldok.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    deployment, deployment_sha = builder.verify(APPLICATION)
    project, actual_sha = checked(ROOT / MANIFEST, MANIFEST_SHA)
    validate(cutover, launch, deployment, deployment_sha, project)
    # Use the verified deployed registry, never a direct Projects/Dataset owner.
    sys.path.insert(0, str(APPLICATION))
    import agent_api
    require(Path(agent_api.__file__).resolve() == APPLICATION / 'agent_api.py', 'loaded another API registry')
    arguments = {'root': 'current-lab', 'manifest_path': MANIFEST}
    agent_api.validate_arguments('project_snapshots.open', arguments, 1)
    method, path, body = agent_api.project_request('project_snapshots.open', arguments, 1)
    contract = agent_api.contract('project_snapshots.open', 1)
    return {'format': 'rheon_current_project_registration_v1', 'endpoint': ENDPOINT, 'catalog_id': CATALOG,
        'project_id': PROJECT, 'manifest_path': MANIFEST, 'manifest_sha256': actual_sha,
        'project_root_alias': 'current-lab', 'project_root': str(ROOT),
        'separate_pressure_root_preserved': True, 'application': str(APPLICATION),
        'deployment_manifest_sha256': deployment_sha,
        'cutover': {'path': str(args.cutover_receipt.resolve()), 'sha256': cutover_sha},
        'launch': {'path': str(args.launch_receipt.resolve()), 'sha256': launch_sha},
        'request': {'operation': 'project_snapshots.open', 'version': 1, 'sha256': contract['sha256'],
                    'method': method, 'path': path, 'body': body}}, launch, agent_api


def verify_process(launch):
    expected = launch['process']
    directory = Path('/proc') / str(expected['pid'])
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    command = [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
    require(int(fields[19]) == expected['start_ticks'] and fields[0] != 'Z' and command == expected['command'],
            'selected current launch process identity changed')


def prove_project(project):
    require(project.get('id') == PROJECT and project.get('schema_version') == 2
            and project.get('root') == 'current-lab' and project.get('manifest_path') == MANIFEST
            and project.get('manifest_sha256') == MANIFEST_SHA
            and project.get('manifest_status') == 'unchanged'
            and project.get('mode') == 'inspection' and project.get('execution') == 'unsupported',
            'normal registered snapshot proof differs from exact current Project04')
    return {key: project[key] for key in ('id', 'schema_version', 'root', 'manifest_path',
        'manifest_sha256', 'manifest_status', 'mode', 'execution')}


def dispatch(journal, record, invoke):
    """Persist one intent; uncertain outcomes stop and cannot replay this journal."""
    persist(journal, {**record, 'status': 'registration_dispatch_intent'}, exclusive=True)
    try:
        result = invoke()
        proof = prove_project(result)
    except BaseException as error:
        persist(journal, {**record, 'status': 'unresolved_inspect_durable_state',
            'code': getattr(error, 'code', type(error).__name__), 'http_status': getattr(error, 'status', None),
            'automatic_retry': False})
        raise
    persist(journal, {**record, 'status': 'registration_acknowledged', 'proof': proof, 'automatic_retry': False})
    return result


def execute(args, record, launch, api):
    checked(args.cutover_receipt, record['cutover']['sha256'])
    checked(args.launch_receipt, record['launch']['sha256'])
    verify_process(launch)
    output = args.output.resolve()
    require(output.is_relative_to(MIGRATION) and not output.exists(), 'fresh contained registration output required')
    journal = OPS / ('.current-project-04-registration-' + record['cutover']['sha256'] + '-' + record['launch']['sha256'] + '.json')
    require(not journal.exists(), 'registration journal exists; inspect durable state instead of replaying')
    # Authority is first read only after the explicit execution flag and all
    # local public gates, deployment bytes and actual launch process match.
    token = api._private_file(args.owner_session_file, 4096).decode('ascii').strip()
    transport = api.AgentClient(ENDPOINT, token, CATALOG, timeout=120)

    def checked_response(envelope):
        if envelope['status'] >= 400:
            value = envelope['result']
            raise api.AgentAPIError(value['error'], value['code'], envelope['status'])
        return envelope['result']

    identity = checked_response(transport._request('GET', '/api/access/session'))
    require(identity['kind'] == 'owner' and identity['catalog_id'] == CATALOG, 'existing current catalog owner required')
    require(checked_response(transport._request('GET', '/api/workspace/context'))['dataset_id'] == CATALOG,
            'actual endpoint catalog differs')
    versions = api.negotiate(checked_response(transport._request('GET', '/api/agent/capabilities')), CATALOG)

    def call(name, arguments, version=None):
        version = versions[name] if version is None else version
        require(versions.get(name, 0) >= version, 'current selected operation is unavailable')
        api.validate_arguments(name, arguments, version)
        method, path, body = api.project_request(name, arguments, version)
        envelope = transport._request(method, path, body, (name, version))
        api.validate_output(name, envelope, version)
        return checked_response(envelope)

    require({'current-lab', 'pressure'} <= set(call('project_snapshots.list', {})['roots']), 'live configured root aliases differ')
    before = sorted(row['id'] for row in call('runs.list', {}))
    checked(args.cutover_receipt, record['cutover']['sha256'])
    checked(args.launch_receipt, record['launch']['sha256'])
    output.mkdir(mode=0o700)
    started = time.monotonic()
    record = {**record, 'journal': str(journal), 'owner_actor_id': identity['actor_id'], 'started_ns': time.time_ns()}
    try:
        registered = dispatch(journal, record, lambda: call('project_snapshots.open', record['request']['body'], 1))
        readback = call('project_snapshots.get', {'project_id': PROJECT})
        proof = prove_project(readback)
        require(proof == prove_project(registered), 'normal current API readback differs from registration')
        after = sorted(row['id'] for row in call('runs.list', {}))
        require(before == after, 'Run identities changed during registration; inspect concurrent state')
        receipt = {**record, 'status': 'registered_and_read_back', 'proof': proof,
            'normal_current_api_only': True, 'readback_operation': 'project_snapshots.get',
            'run_ids_before': before, 'run_ids_after': after, 'new_runs': 0,
            'approval_or_admission': False, 'optimizer_updates_executed': 0,
            'scientific_acceptance': 'not_asserted', 'elapsed_s': time.monotonic() - started}
        persist(output / 'receipt.json', receipt, exclusive=True)
        return receipt
    except BaseException as error:
        persist(output / 'failure.json', {'status': 'stopped_inspect_journal_and_current_snapshot',
            'journal': str(journal), 'project_id': PROJECT, 'code': getattr(error, 'code', type(error).__name__),
            'automatic_retry': False, 'elapsed_s': time.monotonic() - started}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-cutover', action='store_true')
    parser.add_argument('--application', type=Path, default=APPLICATION)
    for prefix in ('cutover', 'launch'):
        parser.add_argument('--' + prefix + '-receipt', type=Path, required=True)
        parser.add_argument('--' + prefix + '-sha256', required=True)
    parser.add_argument('--owner-session-file', type=Path, default=OPS / 'migration-owner-session.private.txt')
    parser.add_argument('--output', type=Path, default=MIGRATION / 'current-project-04-registration-01')
    args = parser.parse_args()
    record, launch, api = prepare(args)
    if not args.execute_after_cutover:
        print(json.dumps({**record, 'status': 'locally_prepared_no_authority_or_network_access'}))
        return
    receipt = execute(args, record, launch, api)
    print(json.dumps({'status': receipt['status'], 'project_id': PROJECT, 'receipt': str(args.output / 'receipt.json')}))


if __name__ == '__main__':
    main()
