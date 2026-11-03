"""Pure-local root completion assembler; no app/catalog/network/process imports.

Observed receipts stay immutable. In particular, machine browser-pending and
cutover incomplete statuses are completed by separate actual browser/review proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/dependency-migration-v1'
OPS = BASE / 'ops'
CATALOG = 'a7da4a8de53541a2a0e2aad08005bb71'
ENDPOINT = 'http://127.0.0.1:8091'
EXISTING_RECORD = '77971da0b73d45a29c373a8461133761'
FIELD_SOURCE = 'd91c7e7136a5d1a4589db02f0ad86cca904571e500c9ab76ba99104e5d3a42b5'
RUNTIME = 'b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682'
FIXED = {
    'cutover': (OPS / 'm3-cutover.json', '7bd1c17f52b29469893711ca2d298d6e5b723317b94dab5e05fe864c0c82a33a'),
    'launch': (OPS / 'm3-app-launch.json', '7e25019a7075c0acf402380b90cecaa5fc6caf20cb5473f3a19535e540834bc7'),
    'gates': (OPS / 'm3-gates.json', '66624920793820c6646e7fd6c51f41837bfd01056b56208679f394d6363436f9'),
    'machine': (BASE / 'typed-pressure-live-01/receipt.json', '9595933635cbde2041acefe3fb076ed9ced673b7b423610f3612c2c7f2320832'),
    'project_provenance': (BASE / 'current-project-04/provenance.json', '9cca1d763eebb2c069780678c43a466d7a5baf8ebc7cce6c217418687de524e5'),
    'project_manifest': (BASE / 'current-project-04/project.json', '97a8020cd6cfe39afa426921dc692dc948d0d6475f9a4dfd37dea49f4792b2a0'),
    'runtime_manifest': (BASE / 'runtime/current-manifest-v2.json', RUNTIME),
}
FUTURE = {
    'runtime_registration': OPS / 'm3-runtime-registration.json',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def local(path):
    selected = Path(path)
    require(selected.is_absolute() and not selected.is_symlink(), 'Explicit absolute nonsymlink local proof required')
    resolved = selected.resolve(strict=True)
    require(resolved.is_relative_to(BASE), 'Proof is outside migration evidence; catalog access is forbidden')
    return resolved


def bound(row):
    require(type(row) is dict and set(row) >= {'path', 'sha256'}, 'Explicit proof path/hash required')
    path = local(row['path'])
    require(sha(path) == row['sha256'], 'Proof bytes changed')
    require(path.stat().st_size <= 64 * 1024**2, 'JSON proof exceeds bounded local read')
    return json.loads(path.read_bytes())


def same_ref(actual, expected):
    return actual == {'path': str(Path(expected['path']).resolve()), 'sha256': expected['sha256']}


def validate_browser(browser, launcher, machine, project, bindings):
    require(browser['schema'] == 'rheon-root-gated-live-browser-readback-v1' and browser['status'] == 'passed',
            'Actual root-selected browser PASS required')
    require(launcher['status'] == 'passed' and launcher['returncode'] == 0
            and launcher['owned_child_reaped'] is True and launcher['cleanup_error_kind'] is None
            and launcher['import_dispatch_may_have_committed'] is False,
            'Actual browser launcher must be successful, reaped and read-only')
    require(browser['catalog_id'] == machine['catalog_id'] == CATALOG
            and browser['endpoint'] == machine['endpoint'] == ENDPOINT, 'Browser target differs')
    require(browser['project_id'] == project['id']
            and browser['project_manifest_sha256'] == bindings['project_manifest']['sha256'], 'Browser current Project04 differs')
    require(browser['collection']['record_id'] == EXISTING_RECORD
            and browser['collection']['content_hash'] == FIELD_SOURCE
            and browser['collection']['read_only_reuse'] is True
            and browser['collection']['created_new_item'] is False
            and browser['collection']['existing_record_lookup']['collection_writes'] == 0
            and browser['mutations'] == [] and browser['extra_runs'] == 0
            and browser['approval_or_admission_or_promotion'] is False, 'Browser must reuse exact existing field without another import')
    for key in ('model', 'composition_sha256', 'training_run_id', 'restore_run_id', 'checkpoint_sha256'):
        require(browser[key] == machine[key], 'Browser/machine identity differs: ' + key)
    require(browser['trial']['id'] == machine['trial']['id']
            and browser['trial']['material'] == machine['trial']['material'], 'Browser Trial/composition differs')
    require(browser['field_view']['checkpoint_sha256'] == machine['checkpoint_document_sha256']
            and browser['field_view']['cpu_pressure_sha256'] == machine['probe']['pressure_sha256']
            and browser['field_view']['units'] == 'Pa' and browser['field_view']['dtype'] == '<f8', 'Browser physical field/CPU proof differs')
    for name in ('cutover', 'launch', 'machine', 'project_provenance'):
        require(any(same_ref(row, bindings[name]) for row in browser['gates']), 'Browser did not bind exact proof: ' + name)
    activity = browser['activity']
    require(activity['authority'] == 'external_observation' and activity['healthy'] is True
            and activity['state'] in ('running', 'completed') and activity['phase'] == 'Dependency migration observation'
            and activity['progress']['completed'] == activity['progress']['total'] == 74
            and activity['machine_status'] == machine['status']
            and activity['last_sequence'] > activity['first_sequence']
            and '74/74 current regression case receipts passed' in activity['message']
            and '0 failed, 0 unreadable' in activity['message']
            and 'typed-pressure-live-01: status=' + machine['status'] in activity['message']
            and 'Observation unavailable' not in activity['message']
            and hashlib.sha256(activity['message'].encode()).hexdigest() == activity['message_sha256'],
            'Actual healthy 74/74 and completed-machine activity proof required')
    require(browser['scientific_acceptance'] == 'not_asserted' and browser['screenshots'], 'Browser evidence/claim differs')


def validate_review(review, machine, bindings):
    require(review['schema'] == 'rheon-live-machine-browser-independent-review-v1'
            and review['status'] == 'passed' and review['accepted'] is True
            and review['scope'] == 'actual-original-catalog-machine-browser'
            and review['blocking_findings'] == [], 'Actual combined independent PASS required')
    require(review['machine_review']['status'] == 'passed_actual_live_machine_artifacts'
            and review['browser_review']['status'] == 'passed_read_only_existing_record', 'Independent machine/browser acceptance missing')
    require(review['original_target']['catalog_id'] == CATALOG and review['original_target']['endpoint'] == ENDPOINT
            and same_ref(review['original_target']['cutover'], bindings['cutover'])
            and same_ref(review['original_target']['launch'], bindings['launch']), 'Independent review original target differs')
    for key in ('model', 'training_run_id', 'restore_run_id', 'checkpoint_sha256', 'checkpoint_document_sha256'):
        require(review['identities'][key] == machine[key], 'Independent reviewed identity differs: ' + key)
    attempts = [value for value in review['browser_review'].values() if isinstance(value, dict)
                and value.get('receipt') == bindings['browser']['path']
                and value.get('sha256') == bindings['browser']['sha256']]
    require(len(attempts) == 1, 'Independent review must bind exactly the selected successful browser attempt')
    attempt = attempts[0]
    require(attempt['receipt'] == bindings['browser']['path'] and attempt['sha256'] == bindings['browser']['sha256']
            and attempt['launcher_receipt'] == bindings['browser_launcher']['path']
            and attempt['launcher_sha256'] == bindings['browser_launcher']['sha256']
            and attempt['import_dispatches'] == 0 and attempt['new_field_records'] == 0 and attempt['new_runs'] == 0
            and attempt['existing_record'] == EXISTING_RECORD and attempt['source_sha256'] == FIELD_SOURCE,
            'Independent browser proof/effects differ')


def assemble(selection):
    require(selection['format'] == 'rheon_m3_completion_selection_v1', 'Unknown completion selection')
    bindings = selection['proofs']
    require(set(bindings) == set(FIXED) | set(FUTURE) | {'browser', 'browser_launcher', 'independent_review'},
            'Every exact completion proof must be selected explicitly')
    for name, (path, expected) in FIXED.items():
        require(bindings[name] == {'path': str(path), 'sha256': expected}, 'Immutable proof selection differs: ' + name)
    for name, path in FUTURE.items():
        require(bindings[name]['path'] == str(path), 'Explicit runtime registration proof path differs')
    browser_path, launcher_path = (Path(bindings[name]['path']) for name in ('browser', 'browser_launcher'))
    require(browser_path.name == 'browser-receipt.json' and launcher_path.name == 'launcher-receipt.json'
            and browser_path.parent == launcher_path.parent, 'Select one explicit browser attempt and its own launcher')
    documents = {name: bound(row) for name, row in bindings.items()}
    cutover, launch, gates, machine = (documents[name] for name in ('cutover', 'launch', 'gates', 'machine'))
    require(cutover['schema'] == 'rheon-tuldok-migration-cutover-v1' and cutover['m3_service_switched'] is True
            and cutover['m3_complete'] is False and cutover['ordinary_admissions_open'] is False,
            'Consumed cutover must remain immutable/incomplete')
    require(machine['format'] == 'rheon_typed_pressure_qualification_v1'
            and machine['status'] == 'machine_qualified_browser_pending'
            and machine['optimizer_updates_executed'] == 1 and machine['restore_optimizer_updates'] == 0
            and machine['ordinary_restore_bound_profile'] is False and machine['model_promotion'] is False
            and machine['scientific_acceptance'] == 'not_asserted', 'Actual bounded machine proof differs')
    require(same_ref(machine['m3'], bindings['cutover']) and same_ref(machine['app_launch'], bindings['launch'])
            and cutover['root_gate_sha256'] == bindings['gates']['sha256']
            and launch['gates']['sha256'] == bindings['gates']['sha256'], 'Cutover/launch/gate/machine chain differs')
    require(launch['status'] == 'current_service_launched_smoke_pending'
            and launch['app_source'] == gates['tuldok_source'], 'Observed launch/source proof differs')
    for name in ('catalog_id', 'endpoint', 'app_sha256', 'app_source_sha256', 'deployment_manifest_sha256', 'rheon_binary_sha256'):
        require(cutover[name] == launch[name], 'Observed cutover/launch identity differs: ' + name)
    require(cutover['catalog_id'] == machine['catalog_id'] == CATALOG and cutover['endpoint'] == machine['endpoint'] == ENDPOINT, 'Original target differs')
    for row in cutover['proofs'].values():
        bound(row)
    for row in gates['artifacts'].values():
        require(sha(local(row['path'])) == row['sha256'], 'Bound deployed artifact changed')
    deployment_path = local(gates['artifacts']['deployment_manifest']['path'])
    deployment = json.loads(deployment_path.read_bytes())
    for row in deployment['files']:
        path = local(deployment_path.parent / row['path'])
        require(path.is_relative_to(deployment_path.parent) and path.stat().st_size == row['bytes']
                and path.stat().st_mode & 0o7777 == row['mode'] and sha(path) == row['sha256'], 'Deployed payload changed')
    for row in gates['receipts'].values():
        value = bound(row)
        for check in row['checks']:
            current = value
            for key in check['path']:
                current = current[key]
            require(canonical(current) == canonical(check['equals']) if 'equals' in check else len(current) == check['length'], 'Original actual gate no longer verifies')
    require(bound(gates['receipts']['q3_pipeline'])['observed_optimizer_updates']['pipeline'] == 1
            and bound(gates['receipts']['q8_lifecycle'])['observed_optimizer_updates']['lifecycle'] == 0, 'Precutover update accounting differs')
    provenance, project = documents['project_provenance'], documents['project_manifest']
    require(project['schema_version'] == 2 and project['id'] == provenance['project_id']
            and provenance['current_manifest_sha256'] == bindings['project_manifest']['sha256'], 'Project04 provenance differs')
    registration = documents['runtime_registration']
    require(registration['status'] == 'passed' and registration['catalog_id'] == CATALOG
            and registration['runtime_manifest_sha256'] == machine['source_verification']['runtime_manifest_sha256'] == RUNTIME
            and registration['app_launch_sha256'] == bindings['launch']['sha256']
            and registration['optimizer_updates'] == 0
            and machine['source_verification']['runtime_fully_verified'] is True, 'Actual registered runtime differs')
    require(documents['runtime_manifest']['profile'] == 'managed_torch_python_v1', 'Runtime profile differs')
    for row in machine['download_receipts']:
        path = local(row['output'])
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Actual downloaded Run artifact changed')
    live = BASE / 'typed-pressure-live-01'
    train, restore = (json.loads((live / label / 'receipt.json').read_bytes()) for label in ('train', 'restore'))
    require(train['optimizer_updates'] == 1 and train['execution']['device'] == 'cuda:0'
            and restore['optimizer_updates'] == 0 and restore['execution']['device'] == 'cpu'
            and restore['execution']['cuda_initialized'] is False
            and restore['originating_training_run']['run_id'] == machine['training_run_id']
            and train['prediction_inventory'] == restore['prediction_inventory'], 'Actual live Run restore/update identity differs')
    require(machine['probe']['status'] == 'exact' and machine['probe']['optimizer_updates'] == 0
            and machine['probe']['cuda_initialized'] is False and machine['probe']['field_api_cli_exact'] is True
            and machine['probe']['weights_sha256'] == machine['checkpoint_sha256'], 'Fresh CPU pressure evidence differs')
    browser, launcher = documents['browser'], documents['browser_launcher']
    validate_browser(browser, launcher, machine, project, bindings)
    require(browser['deployment_manifest_sha256'] == launch['deployment_manifest_sha256'], 'Browser deployment differs')
    validate_review(documents['independent_review'], machine, bindings)
    return {'schema': 'rheon-tuldok-migration-completion-v1', 'status': 'completed', 'm3_complete': True,
        'm3_service_switched': True, 'ordinary_admissions_open': True,
        'ordinary_admission_status_authority': 'operator_policy_not_transport_enforcement',
        'catalog_id': CATALOG, 'endpoint': ENDPOINT, 'proofs': bindings,
        'migration_optimizer_updates': {'cuda_total': 2, 'pre_cutover_q3': 1, 'post_cutover_smoke': 1, 'cpu_total': 0},
        'model': machine['model'], 'trial': machine['trial'], 'training_run_id': machine['training_run_id'],
        'restore_run_id': machine['restore_run_id'], 'checkpoint_sha256': machine['checkpoint_sha256'],
        'current_project_id': project['id'], 'existing_field_record_id': EXISTING_RECORD,
        'final_browser_read_only_reuse': True, 'final_browser_new_imports': 0,
        'scientific_acceptance': 'not_asserted', 'model_promotion': False,
        'historical_ancestry': 'unknown; prospective Trial',
        'observed_source_identity': gates['tuldok_source'], 'deployed_manifest_sha256': launch['deployment_manifest_sha256'],
        'later_source_retirement_or_unpinned_document_refresh_invalidates_observation': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--write-completion', action='store_true')
    args = parser.parse_args(argv)
    require(sha(args.selection) == args.selection_sha256, 'Root completion selection changed')
    result = assemble(json.loads(args.selection.read_bytes()))
    if args.write_completion:
        output = OPS / 'm3-complete.json'
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(result)); stream.flush(); os.fsync(stream.fileno())
        directory = os.open(OPS, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(json.dumps({'status': 'all_actual_completion_proofs_verified', 'completion_written': args.write_completion,
                      'completion_sha256': hashlib.sha256(canonical(result)).hexdigest()}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.stderr.write('M3 completion stopped (' + type(error).__name__ + '); no pending proof is inferred.\n')
        sys.exit(1)
