"""Own one managed guided CUDA fit and four-checkpoint typed CPU restore."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/full-ai-next-priorities-v1'))
from managed_input_stream_v1 import streamed_identity


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def sha(p):
    return streamed_identity(p)['sha256']


def run(args):
    if args.output.exists():
        raise ValueError('Fresh pair custody required; inspect unknown mutations without replay')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Registered dualwidth source changed: ' + name)
    project_dir = args.project.resolve()
    assembly, document = read(project_dir / 'assembly.json'), read(project_dir / 'project.json')
    root = Path(assembly['project_root'])
    if sha(project_dir / 'project.json') != assembly['project_sha256']:
        raise ValueError('Captured temporary-root Project changed')
    for row in assembly['sources']:
        if sha(root / row['path']) != row['sha256']:
            raise ValueError('Captured managed entrypoint source changed')
    observed = read(ROOT / reg['actual_entrypoint_qualification'])
    if (observed['mode'] != 'qualify' or observed['optimizer_updates'] != 0
            or observed['parameter_fits'] != 0 or observed['cpu_model_forward_executed']
            or observed['execution']['ambient_repository_reads'] != 0
            or observed['closure']['config_sha256'] != sha(project_dir / 'config.json')):
        raise ValueError('Actual zero-update captured-entrypoint gate required')
    for slot, spec in assembly['static_inputs'].items():
        if streamed_identity(spec['path']) != {n: spec[n] for n in ('bytes', 'sha256')}:
            raise ValueError('Exact prepared static input differs')
        if observed['closure']['inputs'][slot] != {n: spec[n] for n in ('bytes', 'sha256')}:
            raise ValueError('Qualified actual entrypoint used different data')
    required = sum(r['bytes'] for r in assembly['static_inputs'].values()) + reg['output_and_metadata_budget_bytes'] + reg['reserve_bytes']
    if shutil.disk_usage(ROOT).free < required:
        raise ValueError('Registered single-copy managed disk admission failed')
    sys.path[:0] = [str(ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'),
                   str(ROOT / 'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api, project_cli
    from qualify_typed_pressure import OwnerHTTP, Journal
    owner = OwnerHTTP(agent_api, 'http://127.0.0.1:8091', 'a7da4a8de53541a2a0e2aad08005bb71',
        ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt')
    args.output.mkdir(parents=True)
    (args.output / 'mutations').mkdir()
    journal = Journal(args.output / 'mutations')
    started = time.monotonic()
    write(args.output / 'intent.json', dict(project=str(project_dir), project_sha256=assembly['project_sha256'],
        assembly_sha256=sha(project_dir / 'assembly.json'), protocol_sha256=sha(args.protocol),
        model_parameter_fits_planned=1, managed_training_Runs_planned=1, automatic_retry=False))

    def mutate(phase, operation, body):
        return journal.mutation(phase, operation, body, lambda: owner.call(operation, body))

    project = mutate('open-project', 'project_snapshots.open', dict(root=assembly['project_root_alias'], manifest_path=assembly['project']))
    if project['id'] != document['id'] or project['manifest_sha256'] != assembly['project_sha256']:
        raise ValueError('Remote Project identity differs')
    refs = {r['id']: r for r in project['data']}
    for slot, spec in assembly['static_inputs'].items():
        row = refs[slot]
        if row['status'] != 'available':
            raise ValueError('Prepared static input unavailable to configured Project root')
        if row['integrity'] == 'verified':
            if row['observed_sha256'] != spec['sha256']:
                raise ValueError('Remote input identity differs')
        elif row.get('reason') != 'Present; content exceeds the bounded hash inspection budget.':
            raise ValueError('Unexpected bounded Project preview status')
    write(args.output / 'input-verification.json', dict(status='full_local_static_hashes_and_remote_preview_verified',
        static_inputs=assembly['static_inputs'], server_material_preflight_still_required=True))
    key = 'guided-shared-' + hashlib.sha256(str(args.output).encode()).hexdigest()[:24]
    active, runs = {}, {}

    def await_run(run_id, limit):
        deadline, previous = time.monotonic() + limit, None
        while True:
            result = owner.call('runs.get', dict(run_id=run_id))
            status = (result['state'], result.get('custody_phase'), json.dumps(result.get('progress'), sort_keys=True))
            if status != previous:
                write(args.output / ('state-' + run_id + '.json'), result)
                print('Managed shared temporal Run', run_id, result['state'], result.get('custody_phase'), result.get('progress'), flush=True)
                previous = status
            if result['state'] in ('completed', 'failed', 'cancelled', 'interrupted', 'unknown'):
                write(args.output / ('terminal-' + run_id + '.json'), result)
                if result['state'] != 'completed' or result['custody_phase'] != 'drained':
                    raise RuntimeError('Managed dualwidth Run did not complete/drain; preserve it without automatic retry')
                return result
            if time.monotonic() > deadline:
                raise TimeoutError('Observation deadline; poll the same Run without resubmission')
            time.sleep(2)

    for phase in ('train', 'restore'):
        recipe = read(project_dir / (phase + '-recipe.json'))
        ref = next(r for r in project['recipes'] if r['id'] == phase)
        inspected = owner.call('recipes.inspect', dict(project_id=project['id'], reference_id=phase, expected_sha256=ref['observed_sha256']))
        if inspected['definition'] != recipe:
            raise ValueError('Remote recipe differs from captured source')
        bindings = {}
        for slot in recipe['inputs']:
            if slot in assembly['restore_artifact_map']:
                name = assembly['restore_artifact_map'][slot]
                artifact = next(r for r in runs['train']['artifacts'] if r['name'] == name)
                bindings[slot] = dict(kind='run_artifact', run_id=runs['train']['id'], artifact_name=name, expected_sha256=artifact['sha256'])
            else:
                bindings[slot] = dict(reference_id=slot, expected_sha256=assembly['static_inputs'][slot]['sha256'])
        graph = mutate(phase + '-graph', 'graphs.create_external', dict(name=document['name'] + ' ' + phase,
            project_id=project['id'], recipe_id=phase, expected_sha256=ref['observed_sha256'], inputs=bindings))
        preflight = owner.call('graphs.preflight_external', dict(graph_id=graph['id'], revision=graph['revision']))
        body = dict(graph_id=graph['id'], graph_revision=graph['revision'], expected_digest=preflight['material_sha256'],
            consumer_actor_id=owner.identity['actor_id'], ttl_seconds=3600)
        grant = journal.mutation(phase + '-approve', 'runs.approve_external', body,
            lambda: owner.raw('POST', '/api/workspace/runs/approve-external', body))
        admitted = mutate(phase + '-admit', 'runs.submit_external', dict(request_key=key + '-' + phase,
            graph_id=graph['id'], graph_revision=graph['revision'], expected_digest=preflight['material_sha256'], approval_id=grant['id']))
        active[phase] = admitted['id']
        write(args.output / 'active.json', active)
        runs[phase] = await_run(admitted['id'], 2150 if phase == 'train' else 600)
        class Reader:
            def call(self, operation, arguments):
                if operation != 'runs.artifact_chunk':
                    raise ValueError('Artifact-only reader')
                return dict(status=200, result=owner.call(operation, arguments))
        directory = args.output / phase
        directory.mkdir()
        for artifact in runs[phase]['artifacts']:
            project_cli.download(Reader(), runs[phase]['id'], artifact['name'], artifact['sha256'], directory / artifact['name'])
    trained, restored = read(args.output / 'train/receipt.json'), read(args.output / 'restore/receipt.json')
    if (trained['schema'] != 'rheon.guided-shared-managed-fit.v1' or trained['parameter_fits'] != 1
            or not 0 < trained['optimizer_updates'] <= 8 or not 0 <= trained['accepted_steps'] <= 4
            or restored['optimizer_updates'] or restored['cpu_model_forward_executed']
            or not restored['typed_cpu_weight_restore_verified'] or not restored['gpu_probe_copied_without_cpu_forward']
            or restored['restored_checkpoints'] != sorted(['weights.npz','best-mean-weights.npz','initial-weights.npz','last-evaluated-weights.npz'])):
        raise ValueError('Actual guided fitting or four-checkpoint CPUrestore semantics differ')
    parity = {name: sha(args.output / 'train' / name) == sha(args.output / 'restore' / name)
              for name in ('weights.npz', 'best-mean-weights.npz', 'initial-weights.npz', 'last-evaluated-weights.npz', 'history.json', 'probe.npy', 'normalization.json')}
    if not all(parity.values()) or restored['originating_training_run']['run_id'] != runs['train']['id']:
        raise ValueError('CPU artifact restoration provenance/parity failed')
    write(args.output / 'receipt.json', dict(status='completed_managed_guided_fit_and_four_checkpoint_CPUrestore',
        training_run_id=runs['train']['id'], restore_run_id=runs['restore']['id'], new_parameter_fits=1,
        optimizer_updates=trained['optimizer_updates'], accepted_steps=trained['accepted_steps'], artifact_parity=parity,
        source_assembly_sha256=sha(project_dir / 'assembly.json'), model_promotion=False,
        cpu_model_forward=False, elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
