"""One gated prospective pressure Trial and ordinary CPU restore through Tuldok.

Source verification is inert. Execution is deliberately non-resumable: a failed
or uncertain mutation leaves a journal for explicit inspection, never a retry.
This is a qualification client, not a worker, scheduler or custody authority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from dependency_config import canonical, decode, digest, require, source_identity


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rooted(root, relative):
    candidate = Path(relative)
    require(not candidate.is_absolute() and '..' not in candidate.parts,
            'material path must be relative and contained')
    path = (root / candidate).resolve(strict=True)
    require(path.is_relative_to(root), 'material escapes selected root')
    return path


def verify_sources(root, assembly_path, tuldok_root, *, verify_runtime=True):
    """Measure declared bytes only; import no model, application or numerical code."""
    root, tuldok_root = root.resolve(strict=True), tuldok_root.resolve(strict=True)
    assembly = decode(assembly_path.read_bytes())
    require(assembly['format'] == 'rheon_pressure_project_assembly_v1'
            and assembly['execution_submitted'] is False
            and assembly['optimizer_updates_executed'] == 0,
            'prospective unexecuted pressure assembly required')
    selected = assembly['current_project']
    project_path = rooted(root, selected['path'])
    require(digest(project_path) == selected['sha256'], 'assembled Project changed')
    project = decode(project_path.read_bytes())
    require(project['id'] == selected['id'] and project['schema_version'] == 2,
            'exact current Project required')
    inventory = {}

    def measure(row):
        path = rooted(root, row['path'])
        actual = digest(path)
        require(actual == row['sha256'], 'changed declared material: ' + row['path'])
        require('bytes' not in row or row['bytes'] == path.stat().st_size, 'declared byte size differs')
        inventory[row['path']] = {'path': row['path'], 'sha256': actual, 'bytes': path.stat().st_size}
        return path

    recipes = {}
    for group in ('data', 'recipes', 'evidence', 'extensions'):
        for row in project.get(group, []):
            path = measure(row)
            if group == 'recipes':
                recipe = decode(path.read_bytes())
                recipes[row['id']] = recipe
                for source in recipe['sources']:
                    measure(source)
            elif group == 'extensions':
                provider = decode(path.read_bytes())
                for contribution in provider['contributions'] + provider['sources']:
                    measure(contribution)
    train, restore = recipes['train'], recipes['restore']
    require(train['schema_version'] == 3 and train['runtime_device'] == 'cuda'
            and restore['schema_version'] == 1 and restore['runtime_device'] == 'cpu',
            'bound CUDA train and ordinary CPU restore required')
    policy = train['model_evidence']
    require(policy['model_specification'] == assembly['model_specification']
            and policy['profile']['capture'] == {'clock': 'optimizer_update', 'interval': 1,
                'max_snapshots': 1, 'retain_selected': True}
            and len(policy['probe']['candidates']) == 1
            and policy['probe']['candidates'][0]['frozen_reference']['split'] == 'validation'
            and policy['probe']['candidates'][0]['frozen_reference']['sha256'] == assembly['probe']['sha256']
            and assembly['probe']['consumption'].startswith('already consumed'),
            'exact one-update consumed-validation pressure contract required')
    configuration = decode((project_path.parent / 'config.json').read_bytes())
    require(configuration['training']['updates'] == 1
            and configuration['training']['checkpoints'] == [0, 1]
            and assembly['limits']['attempt_slots'] == 2,
            'one update and finite two-attempt budget required')
    runtime_path = Path(assembly['runtime']['manifest'])
    require(digest(runtime_path) == assembly['runtime']['manifest_sha256'], 'runtime manifest changed')
    sys.path.insert(0, str(tuldok_root))
    import torch_runtime
    runtime = torch_runtime.load(runtime_path)
    if verify_runtime:
        torch_runtime.verify(runtime)
    revision = subprocess.check_output(['git', '-C', str(tuldok_root), 'rev-parse', 'HEAD']).decode().strip()
    identity = source_identity(tuldok_root, revision)
    return {'format': 'rheon_typed_pressure_source_verification_v1',
            'project_root': str(root), 'tuldok_root': str(tuldok_root),
            'assembly': str(assembly_path.resolve()), 'assembly_sha256': digest(assembly_path),
            'project': selected, 'inventory': sorted(inventory.values(), key=lambda row: row['path']),
            'inventory_sha256': sha(sorted(inventory.values(), key=lambda row: row['path'])),
            'runtime_manifest_sha256': assembly['runtime']['manifest_sha256'],
            'runtime_fully_verified': verify_runtime,
            'tuldok_source': {key: identity[key] for key in ('revision', 'tree', 'content_sha256')},
            'driver_sha256': digest(Path(__file__)), 'optimizer_updates_executed': 0,
            'scientific_acceptance': 'not_asserted'}, assembly, project, recipes, runtime


class Journal:
    """A persisted dispatch intent prevents command restart from replaying writes."""
    def __init__(self, directory):
        self.directory = directory
        self.index = 0

    def mutation(self, phase, operation, arguments, invoke, *, redact_result=False):
        self.index += 1
        path = self.directory / ('%02d-%s.json' % (self.index, phase))
        require(not path.exists(), 'mutation journal exists; inspect durable state, do not replay')
        record = {'phase': phase, 'operation': operation, 'arguments': arguments,
                  'arguments_sha256': sha(arguments), 'status': 'dispatch_intent', 'started_ns': time.time_ns()}
        write(path, record)
        started = time.monotonic()
        try:
            result = invoke()
        except BaseException as error:
            # A validation failure after HTTP success may also conceal a commit.
            record.update(status='unresolved', code=getattr(error, 'code', type(error).__name__),
                          http_status=getattr(error, 'status', None), elapsed_s=time.monotonic() - started,
                          recovery='Read durable state and journal; no automatic mutation retry.')
            write(path, record)
            raise
        record.update(status='acknowledged', elapsed_s=time.monotonic() - started,
                      result={'credential_saved_privately': True} if redact_result else result)
        write(path, record)
        return result


class OwnerHTTP:
    """Current registry/transport, explicitly bound to an existing owner session."""
    def __init__(self, api, endpoint, catalog_id, session_file):
        self.api = api
        token = api._private_file(session_file, 4096).decode('ascii').strip()
        self.transport = api.AgentClient(endpoint, token, catalog_id, timeout=300)
        self.catalog_id = catalog_id
        self.identity = self.raw('GET', '/api/access/session')
        require(self.identity['kind'] == 'owner' and self.identity['catalog_id'] == catalog_id,
                'existing owner session for exact catalog required')
        require(self.raw('GET', '/api/workspace/context')['dataset_id'] == catalog_id,
                'live endpoint catalog differs')
        self.versions = api.negotiate(self.raw('GET', '/api/agent/capabilities'), catalog_id)

    def raw(self, method, path, body=None):
        envelope = self.transport._request(method, path, body)
        return self.checked(envelope)

    def checked(self, envelope):
        if envelope['status'] >= 400:
            result = envelope['result']
            raise self.api.AgentAPIError(result['error'], result['code'], envelope['status'])
        return envelope['result']

    def call(self, operation, arguments):
        version = self.versions[operation]
        self.api.validate_arguments(operation, arguments, version)
        method, path, body = self.api.project_request(operation, arguments, version)
        envelope = self.transport._request(method, path, body, (operation, version))
        self.api.validate_output(operation, envelope, version)
        return self.checked(envelope)


def await_drained(client, run_id, timeout=660):
    deadline = time.monotonic() + timeout
    while True:  # Read-only polling, never admission/approval/submission retry.
        run = client.call('runs.get', {'run_id': run_id})
        if run['state'] in ('completed', 'failed', 'cancelled', 'interrupted', 'unknown'):
            require(run['state'] == 'completed' and run['custody_phase'] == 'drained',
                    'Run failed or cessation is unknown; inspect durable state')
            return run
        require(time.monotonic() < deadline, 'Run wait expired; inspect without resubmission')
        time.sleep(2)


def artifact(run, name):
    matches = [row for row in run['artifacts'] if row['name'] == name]
    require(len(matches) == 1, 'one exact published artifact required: ' + name)
    return matches[0]


def restore_bindings(definition, project, training_run):
    refs = {row['id']: row for row in project['data']}
    result = {}
    for slot in definition['inputs']:
        if slot.startswith('trained_'):
            names = [row['name'] for row in training_run['artifacts']
                     if 'trained_' + row['name'].replace('-', '_').replace('.', '_') == slot]
            if slot == 'trained_profile_json':
                names = ['training-profile.json']
            require(len(names) == 1, 'restore slot has no unique actual Run artifact: ' + slot)
            item = artifact(training_run, names[0])
            result[slot] = {'kind': 'run_artifact', 'run_id': training_run['id'],
                            'artifact_name': item['name'], 'expected_sha256': item['sha256']}
        else:
            ref = refs[slot]
            result[slot] = {'reference_id': slot, 'expected_sha256': ref['observed_sha256']}
    return result


def validate_receipts(training, restored, train_run_id):
    require(training['status'] == restored['status'] == 'complete'
            and training['mode'] == 'train' and training['optimizer_updates'] == 1
            and training['execution']['device'].startswith('cuda')
            and restored['mode'] == 'restore' and restored['optimizer_updates'] == 0
            and restored['execution']['device'] == 'cpu'
            and restored['execution']['cuda_initialized'] is False
            and training['closure'] == restored['closure']
            and training['prediction_inventory'] == restored['prediction_inventory']
            and restored['originating_training_run']['run_id'] == train_run_id,
            'actual one-update training and zero-update fresh CPU custody/parity required')


def check_gate(path, expected_sha256, catalog_id, endpoint):
    require(path is not None and expected_sha256 is not None
            and digest(path) == expected_sha256, 'exact post-M3 receipt required')
    receipt = decode(path.read_bytes())
    require(receipt.get('schema') == 'rheon-tuldok-migration-cutover-v1'
            and receipt.get('m3_service_switched') is True and receipt.get('m3_complete') is False
            and receipt.get('ordinary_admissions_open') is False
            and receipt.get('smoke_authorized') == {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1}
            and receipt.get('catalog_id') == catalog_id and receipt.get('endpoint') == endpoint,
            'structural cutover and exact single-smoke authority required')
    return {'path': str(path.resolve()), 'sha256': expected_sha256}


def check_launch(path, expected_sha256, args, verification):
    require(path is not None and expected_sha256 is not None and digest(path) == expected_sha256,
            'exact current application launch receipt required')
    receipt = decode(path.read_bytes())
    require(receipt['endpoint'] == args.endpoint and receipt['catalog_id'] == args.catalog_id
            and Path(receipt['project_roots'][args.project_alias]).resolve() == args.project_root.resolve()
            and receipt['app_source_sha256'] == verification['tuldok_source']['content_sha256']
            and receipt['app_sha256'] == digest(args.tuldok_root / 'app.py')
            and len(receipt['rheon_binary_sha256']) == 64,
            'live launch root, endpoint, catalog or frozen application source differs')
    return {'path': str(path.resolve()), 'sha256': expected_sha256}


def claim_smoke(gate_path, gate_sha256, output):
    """Consume this local command's single smoke authority before any API write.

    A partial failure needs root inspection/new authority, not another output
    directory silently creating a second prospective update.
    """
    path = gate_path.parent / ('.typed-pressure-smoke-' + gate_sha256 + '.json')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical({'gate_sha256': gate_sha256, 'output': str(output.resolve()),
                                'status': 'claimed_before_first_mutation', 'created_ns': time.time_ns()}))
        stream.flush()
        os.fsync(stream.fileno())
    return str(path.resolve())


def execute(args, verification, assembly, project_document, recipes, runtime):
    import agent_api
    import project_cli
    require(args.prepared_receipt and args.prepared_sha256, 'source-frozen preparation receipt required')
    require(digest(args.prepared_receipt) == args.prepared_sha256, 'preparation receipt changed')
    prior = decode(args.prepared_receipt.read_bytes())
    require(prior['source_verification'] == verification, 'source changed since inert preparation')
    require(prior['target'] == {'endpoint': args.endpoint, 'catalog_id': args.catalog_id, 'project_alias': args.project_alias},
            'execution target differs from inert preparation')
    gate = check_gate(args.m3_receipt, args.m3_sha256, args.catalog_id, args.endpoint)
    launch = check_launch(args.app_launch_receipt, args.app_launch_sha256, args, verification)
    require(args.session_file and args.catalog_root, 'explicit live catalog root and private owner session required')
    registered = args.catalog_root.resolve(strict=True) / 'torch-runtimes' / (verification['runtime_manifest_sha256'] + '.json')
    require(digest(registered) == verification['runtime_manifest_sha256'],
            'register exact managed runtime through current owner CLI before execution')
    owner = OwnerHTTP(agent_api, args.endpoint, args.catalog_id, args.session_file)
    aliases = owner.call('project_snapshots.list', {})['roots']
    require(args.project_alias in aliases, 'selected root alias is not configured')
    smoke_claim = claim_smoke(args.m3_receipt, args.m3_sha256, args.output)
    journal_dir = args.output / 'mutations'
    journal_dir.mkdir()
    journal = Journal(journal_dir)

    def mutate(phase, operation, arguments):
        return journal.mutation(phase, operation, arguments, lambda: owner.call(operation, arguments))

    project = mutate('open-project', 'project_snapshots.open',
        {'root': args.project_alias, 'manifest_path': assembly['current_project']['path']})
    require(project['id'] == assembly['current_project']['id']
            and project['manifest_sha256'] == assembly['current_project']['sha256']
            and project['manifest_status'] == 'unchanged', 'current remote Project differs')
    refs = {row['id']: row for row in project['data']}
    require(all(row['status'] == 'available' and row['integrity'] == 'verified' for row in refs.values()),
            'all exact Project data must be available and verified')
    recipe_ref = next(row for row in project['recipes'] if row['id'] == 'train')
    recipe_sha = recipe_ref.get('observed_sha256', recipe_ref.get('sha256'))
    inspected = owner.call('recipes.inspect', {'project_id': project['id'], 'reference_id': 'train', 'expected_sha256': recipe_sha})
    require(inspected['definition'] == recipes['train'], 'remote train definition differs')
    policy = recipes['train']['model_evidence']
    model = mutate('create-model', 'models.create', {'name': 'Prospective one-update pressure diagnostic',
        'task': inspected['model_task'], 'kind': 'captured_source', 'specification': assembly['model_specification'],
        'source_selector': {'project_id': project['id'], 'recipe_id': 'train', 'expected_sha256': recipe_sha},
        'preprocessing_policy': assembly['preprocessing_policy'], 'output_policy': assembly['output_policy']})
    published = mutate('publish-model', 'models.publish', {'model_id': model['id'], 'revision': model['revision']})
    graph = mutate('train-graph', 'graphs.create_external', {'name': 'Exact prospective pressure fit',
        'project_id': project['id'], 'recipe_id': 'train', 'expected_sha256': recipe_sha,
        'inputs': {slot: {'reference_id': slot, 'expected_sha256': refs[slot]['observed_sha256']}
                   for slot in recipes['train']['inputs']}})
    study = mutate('create-study', 'studies.create', {'name': 'Finite pressure diagnostic', 'limits': assembly['limits'],
        'policy': {'format': 'selection_policy_v1', 'metric_definition_sha256': sha(policy['metric_definitions'][0]),
            'checkpoint_rule': 'explicit', 'replicate_reduce': 'individual',
            'required_capabilities': policy['profile']['required_capabilities'],
            'numerical_acceptance': 'diagnostic', 'tie_break': 'trial_id'}})
    key = 'a11-' + sha({'catalog': args.catalog_id, 'output': str(args.output.resolve())})[:24]
    trial = mutate('prepare-trial', 'studies.prepare_trial', {'study_id': study['id'], 'request_key': key + '-prepare',
        'revision': study['revision'], 'graph_id': graph['id'], 'graph_revision': graph['revision'],
        'model': published['ref'], 'replicate': 0})
    closed = owner.call('studies.get', {'study_id': study['id']})
    require(not closed['study']['open'] and closed['study']['trial_ids'] == [trial['id']]
            and not closed['reservations'], 'one closed Trial and zero reservations required')
    material = owner.call('recipes.material', {'ref': trial['material']})
    require(material['material']['model'] == published['ref'], 'Trial must bind exact published Model')
    write(args.output / 'closed-trial.json', {'study': closed, 'trial': trial, 'composition': material, 'model': published})
    grant = mutate('approve-trial', 'studies.approve', {'study_id': study['id'], 'trial_id': trial['id'],
        'expected_material_sha256': trial['material']['sha256'], 'consumer_actor_id': owner.identity['actor_id'], 'ttl_seconds': 1800})
    opened = mutate('open-admission', 'studies.continue', {'study_id': study['id'], 'request_key': key + '-open',
        'revision': closed['study']['revision'], 'admission_epoch': closed['study']['admission_epoch']})
    admitted = mutate('admit-training', 'studies.admit', {'study_id': study['id'], 'trial_id': trial['id'],
        'request_key': key + '-fit', 'revision': opened['revision'], 'admission_epoch': opened['admission_epoch'], 'approval_id': grant['id']})
    training = await_drained(owner, admitted['run']['id'])
    current_study = owner.call('studies.get', {'study_id': study['id']})['study']
    stopped = mutate('close-study', 'studies.stop', {'study_id': study['id'], 'request_key': key + '-stop', 'revision': current_study['revision']})
    require(stopped['status']['study']['trial_ids'] == [trial['id']]
            and len(stopped['status']['reservations']) == 1
            and stopped['status']['reservations'][0]['state'] == 'settled', 'one settled protected membership required')
    evidence = owner.call('runs.model_evidence', {'run_id': training['id'], 'split': 'validation'})
    require(evidence['state'] == 'verified' and len(evidence['fields']) == 1
            and len(evidence['checkpoints']) == 1, 'one retained SDK field/checkpoint required')
    selected_field = evidence['fields'][0]
    require(evidence['manifest']['model_sha256'] == published['ref']['sha256']
            and evidence['manifest']['producer']['material_sha256'] == trial['material']['sha256']
            and evidence['checkpoints'][0]['weights']['sha256'] == artifact(training, 'checkpoint-001.npz')['sha256']
            and evidence['checkpoint_index']['entries'][0]['checkpoint']['sha256'] == selected_field['checkpoint_sha256'],
            'SDK Model/composition/checkpoint/actual weight mapping differs')
    field_args = {'run_id': training['id'], **{key: selected_field[key] for key in ('checkpoint_sha256', 'member_ordinal', 'field_id')},
                  'axis': 'z', 'index': 0, 'split': 'validation'}
    field = owner.call('runs.model_evidence', field_args)
    write(args.output / 'field-api.json', field)
    restore_ref = next(row for row in project['recipes'] if row['id'] == 'restore')
    restore_sha = restore_ref.get('observed_sha256', restore_ref.get('sha256'))
    inspected_restore = owner.call('recipes.inspect', {'project_id': project['id'], 'reference_id': 'restore', 'expected_sha256': restore_sha})
    require(inspected_restore['definition'] == recipes['restore'], 'remote ordinary CPU restore differs')
    bindings = restore_bindings(recipes['restore'], project, training)
    restore_graph = mutate('restore-graph', 'graphs.create_external', {'name': 'Ordinary CPU restore of actual pressure Run artifacts',
        'project_id': project['id'], 'recipe_id': 'restore', 'expected_sha256': restore_sha, 'inputs': bindings})
    preflight = owner.call('graphs.preflight_external', {'graph_id': restore_graph['id'], 'revision': restore_graph['revision']})
    approval_body = {'graph_id': restore_graph['id'], 'graph_revision': restore_graph['revision'],
                     'expected_digest': preflight['material_sha256'], 'consumer_actor_id': owner.identity['actor_id'], 'ttl_seconds': 1800}
    restore_grant = journal.mutation('approve-restore', 'runs.approve_external', approval_body,
        lambda: owner.raw('POST', '/api/workspace/runs/approve-external', approval_body))
    restored_admit = mutate('admit-restore', 'runs.submit_external', {'request_key': key + '-restore',
        'graph_id': restore_graph['id'], 'graph_revision': restore_graph['revision'],
        'expected_digest': preflight['material_sha256'], 'approval_id': restore_grant['id']})
    restored = await_drained(owner, restored_admit['id'])
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    if args.agent_config:
        config_path = args.agent_config
    else:
        credential = journal.mutation('issue-read-agent', 'access.issue', {'label': 'A11 pressure evidence readback',
            'scopes': ['inspect'], 'ttl_seconds': 3600}, lambda: owner.raw('POST', '/api/access/credentials',
                {'label': 'A11 pressure evidence readback', 'scopes': ['inspect'], 'ttl_seconds': 3600}), redact_result=True)
        token_path = private / 'read-agent.token'
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream: stream.write(credential['token'])
        config_path = private / 'read-agent.json'
        fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical({'endpoint': args.endpoint, 'catalog_id': args.catalog_id, 'token_file': str(token_path.resolve()), 'timeout': 300}))
    reader = agent_api.AgentClient.from_config(config_path)
    require((reader.host, reader.port, reader.catalog_id) ==
            (owner.transport.host, owner.transport.port, args.catalog_id), 'read agent target differs from exact live endpoint/catalog')
    readback_identity = reader.bootstrap()['identity']
    require(reader.selected_contracts().get('runs.model_evidence') == 2, 'current typed field contract required')
    require(reader.call('runs.model_evidence', field_args)['result'] == field, 'agent/API field readback differs')
    download_receipts = []
    for label, run in (('train', training), ('restore', restored)):
        directory = args.output / label
        directory.mkdir()
        names = ['receipt.json', 'weights.npz', 'predictions.npz', 'normalization.json', 'history.json', 'checkpoint-001.npz']
        if label == 'train': names += ['pressure-prediction.npy', 'pressure-reference.npy', 'pressure-diagnostic.json']
        for name in names:
            download_receipts.append(project_cli.download(reader, run['id'], name, artifact(run, name)['sha256'], directory / name))
    train_receipt = decode((args.output / 'train/receipt.json').read_bytes())
    restored_receipt = decode((args.output / 'restore/receipt.json').read_bytes())
    validate_receipts(train_receipt, restored_receipt, training['id'])
    for name in ('weights.npz', 'predictions.npz', 'normalization.json', 'history.json', 'checkpoint-001.npz'):
        require(digest(args.output / 'train' / name) == digest(args.output / 'restore' / name), 'exact restore artifact parity: ' + name)
    require(digest(args.output / 'train/weights.npz') == digest(args.output / 'train/checkpoint-001.npz'), 'final checkpoint/weights differ')
    cli_arguments = args.output / 'field-cli-arguments.json'
    write(cli_arguments, field_args)
    cli = subprocess.run([sys.executable, '-B', str(args.tuldok_root / 'project_cli.py'), '--config', str(config_path),
        'call', '--operation', 'runs.model_evidence', '--arguments', str(cli_arguments)], capture_output=True, timeout=330)
    require(cli.returncode == 0, 'current CLI field readback failed; inspect retained stderr privately')
    cli_value = decode(cli.stdout)
    require(cli_value == field, 'CLI/API exact field readback differs')
    write(args.output / 'field-cli.json', cli_value)
    payload = {'project_root': str(args.project_root.resolve()), 'assembly': str(args.assembly.resolve()),
               'tuldok_root': str(args.tuldok_root.resolve()), 'output': str(args.output.resolve()),
               'field': field['field'], 'observation': evidence['observations'][0]}
    write(args.output / 'probe-request.json', payload)
    bootstrap = 'import sys,json,pathlib; m=json.load(open(sys.argv[1])); sys.path[:0]=[str(pathlib.Path(sys.argv[2]).parent),m["stdlib"],m["stdlib"]+"/lib-dynload",m["site_packages"]]; import runpy; sys.argv=[sys.argv[2],"--probe-child",sys.argv[3]]; runpy.run_path(sys.argv[0],run_name="__main__")'
    env = {'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '', 'TULDOK_TORCH_DEVICE': 'cpu', 'PYTHONDONTWRITEBYTECODE': '1'}
    child = subprocess.run([runtime['python'], '-I', '-S', '-B', '-c', bootstrap, assembly['runtime']['manifest'],
        str(Path(__file__).resolve()), str(args.output / 'probe-request.json')], env=env, capture_output=True, timeout=180)
    require(child.returncode == 0 and not child.stderr, 'fresh hidden-CUDA pressure probe failed')
    probe = decode(child.stdout)
    write(args.output / 'fresh-probe.json', probe)
    result = {'format': 'rheon_typed_pressure_qualification_v1', 'status': 'machine_qualified_browser_pending',
        'catalog_id': args.catalog_id, 'endpoint': args.endpoint, 'source_verification': verification,
        'm3': gate, 'app_launch': launch, 'smoke_claim': smoke_claim,
        'project_root_alias': args.project_alias, 'project_id': project['id'], 'model': published['ref'],
        'trial': trial, 'study': stopped['status'], 'composition_sha256': trial['material']['sha256'],
        'training_run_id': training['id'], 'checkpoint_sha256': artifact(training, 'checkpoint-001.npz')['sha256'],
        'checkpoint_document_sha256': selected_field['checkpoint_sha256'], 'readback_identity': readback_identity,
        'restore_run_id': restored['id'], 'restore_inputs': bindings, 'download_receipts': download_receipts,
        'probe': probe, 'optimizer_updates_executed': 1, 'restore_optimizer_updates': 0,
        'browser': {'status': 'pending', 'url': args.endpoint + '/?workspace=models',
                    'run_id': training['id'], **selected_field},
        'scientific_acceptance': 'not_asserted', 'model_promotion': False,
        'historical_ancestry': 'unknown; new prospective Trial',
        'ordinary_restore_bound_profile': False}
    write(args.output / 'receipt.json', result)
    return result


def probe_child(path):
    """CPU diagnostic of actual restored bytes; never call train.run/main/optimizer."""
    started = time.monotonic()
    payload = decode(path.read_bytes())
    output = Path(payload['output'])
    sys.path.insert(0, str(Path(payload['assembly']).parent))
    import numpy as np
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('TULDOK_TORCH_DEVICE') == 'cpu', 'hidden CUDA required')
    torch.optim.Optimizer.__init__ = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('optimizer forbidden'))
    import train
    configuration = decode((Path(payload['assembly']).parent / 'config.json').read_bytes())
    train.model.select(configuration)
    train.model.configure('cpu')
    history = decode((output / 'restore/history.json').read_bytes())
    network = train.model.PressureCorrectionNet().cpu()
    train.data.helpers().load_weights(network, output / 'restore/weights.npz', history['checkpoints']['checkpoint-001.npz']['inventory'])
    network.eval()
    assembly = decode(Path(payload['assembly']).read_bytes())
    with np.load(assembly['probe']['path'], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    shape = arrays['initial_residual'].shape
    sample = {'base': train.torch.from_numpy(arrays['previous_pressure']).reshape(1, 1, *shape),
              'native_input_residual': train.torch.from_numpy(arrays['initial_residual']).reshape(1, 1, *shape),
              'rho': train.torch.tensor([1000.], dtype=train.torch.float64),
              'dt': train.torch.tensor([1/60], dtype=train.torch.float64),
              'cell_size_m': train.torch.tensor([assembly['probe']['geometry']['spacing_xyz'][0]], dtype=train.torch.float64)}
    with train.torch.no_grad():
        predicted = train.model.predict_pressure(network, sample, decode((output / 'restore/normalization.json').read_bytes())).reshape(*shape).numpy()
    retained = np.load(output / 'train/pressure-prediction.npy', allow_pickle=False)
    reference = np.load(output / 'train/pressure-reference.npy', allow_pickle=False)
    require(predicted.dtype.str == '<f8' and predicted.tobytes() == retained.tobytes()
            and reference.tobytes() == arrays['native_pressure'].tobytes(), 'fresh fixed-probe exact pressure bytes differ')
    require(payload['field']['prediction'] == predicted[0].tolist()
            and payload['field']['reference'] == reference[0].tolist(), 'API/CLI SDK field slice differs from exact pressure bytes')
    from typed_pressure_evidence import mean_free_relative_l2
    metric = mean_free_relative_l2(predicted, reference)
    diagnostic = decode((output / 'train/pressure-diagnostic.json').read_bytes())
    require(metric == diagnostic['mean_free_same_request_pressure_relative_l2'] == payload['observation']['value'],
            'fresh same-request mean-free pressure metric differs from SDK observation')
    require(train.torch.cuda.is_initialized() is False, 'CUDA initialized during CPU diagnostic')
    print(json.dumps({'status': 'exact', 'pressure_sha256': hashlib.sha256(predicted.tobytes()).hexdigest(),
        'weights_sha256': digest(output / 'restore/weights.npz'), 'cuda_initialized': False,
        'optimizer_updates': 0, 'fresh_process': True, 'field_api_cli_exact': True,
        'mean_free_same_request_pressure_relative_l2': metric, 'elapsed_s': time.monotonic() - started}))


def main(argv=None):
    if argv is None: argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == '--probe-child':
        probe_child(Path(argv[1]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'assembly', 'tuldok-root', 'catalog-root', 'session-file', 'agent-config',
                 'output', 'prepared-receipt', 'm3-receipt', 'app-launch-receipt'):
        parser.add_argument('--' + name, type=Path, required=name in ('project-root', 'assembly', 'tuldok-root', 'output'))
    for name in ('endpoint', 'catalog-id', 'prepared-sha256', 'm3-sha256', 'app-launch-sha256'):
        parser.add_argument('--' + name, required=name in ('endpoint', 'catalog-id'))
    parser.add_argument('--project-alias', default='pressure')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--execute-after-m3', action='store_true')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'fresh output required; inspect prior journal instead of mutation replay')
    args.output.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    try:
        verification, assembly, project, recipes, runtime = verify_sources(args.project_root, args.assembly, args.tuldok_root)
        if args.prepare_only:
            result = {'status': 'source_verified_no_execution', 'source_verification': verification,
                'target': {'endpoint': args.endpoint, 'catalog_id': args.catalog_id, 'project_alias': args.project_alias},
                'optimizer_updates_executed': 0, 'http_requests': 0, 'elapsed_s': time.monotonic() - started}
        else:
            result = execute(args, verification, assembly, project, recipes, runtime)
        result.update(command=[sys.executable, str(Path(__file__).resolve()), *argv],
                      elapsed_s=time.monotonic() - started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output / 'receipt.json', result)
        print(json.dumps({'status': result['status'], 'receipt': str(args.output / 'receipt.json'),
                          'elapsed_s': time.monotonic() - started}))
    except BaseException as error:
        write(args.output / 'failure.json', {'status': 'stopped', 'code': getattr(error, 'code', type(error).__name__),
            'elapsed_s': time.monotonic() - started, 'recovery': 'Inspect journal and durable state; do not replay mutations.'})
        raise


if __name__ == '__main__':
    main()
