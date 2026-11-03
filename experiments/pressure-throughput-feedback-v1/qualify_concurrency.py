"""Qualify actual CUDA pressure training beside real managed CPU computation.

Creates one isolated qualification catalog, one CUDA optimizer update, one CPU
tensor fixture and one fresh CPU checkpoint restore. No original catalog writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'tuldok', 'runtime', 'training-project', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    for name in ('root', 'tuldok', 'runtime', 'training_project'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.tuldok.resolve()))
    from app import Dataset
    from runs import process_status
    from torch_runtimes import TorchRuntimes
    import torch_runtime
    environment = {'TULDOK_CPU_RUN_SLOTS': '6', 'TULDOK_CPU_RUN_MEMORY_MIB': '4096',
        'TULDOK_CUDA_RUN_MEMORY_MIB': '8192', 'TULDOK_RUN_MEMORY_RESERVE_MIB': '8192'}
    os.environ.update(environment)
    report = {'schema': 'pressure-actual-device-overlap-v1', 'status': 'running',
        'source_sha256': sha(__file__), 'tuldok_source_sha256': {name: sha(args.tuldok / name)
            for name in ('runs.py', 'run_capacity.py', 'recipe_contract.py', 'recipe_process.py', 'torch_runtime.py')},
        'runtime_manifest_sha256': sha(args.runtime), 'configuration': environment,
        'optimizer_updates_authorized': 1, 'original_catalog_used': False, 'scientific_promotion': False,
        'runs': [], 'observations': []}
    started = time.perf_counter()
    clock_offset = time.time() - time.monotonic()
    project = args.output / 'project'
    project.mkdir()
    source = Path(__file__).with_name('cpu_overlap_fixture.py')
    (project / source.name).write_bytes(source.read_bytes())
    prefix = project.relative_to(args.root).as_posix() + '/'
    definition = {'schema_version': 1, 'format': 'external_recipe_v1', 'id': 'cpu-overlap-fixture',
        'name': 'Real CPU overlap qualification; no model quality claim', 'version': 1,
        'runtime': 'managed_torch_python_v1', 'runtime_device': 'cpu', 'runtime_snapshot_sha256': sha(args.runtime),
        'sources': [{'path': prefix + source.name, 'sha256': sha(source)}], 'entrypoint': prefix + source.name,
        'arguments': [{'literal': '--output-dir'}, {'output_dir': True}, {'literal': '--seconds'}, {'literal': '120'}],
        'parameters': {}, 'inputs': {}, 'environment': {},
        'outputs': [{'path': name, 'role': 'diagnostic', 'required': True, 'max_bytes': 4096}
                    for name in ('started.json', 'receipt.json')],
        'resources': {'wall_seconds': 180, 'max_output_bytes': 8192, 'cpu_cores': 1, 'memory_bytes': 1024 ** 3}}
    write(project / 'cpu-recipe.json', definition)
    manifest = {'schema_version': 2, 'id': uuid.uuid4().hex, 'name': 'CPU/CUDA scheduler qualification',
        'data': [], 'evidence': [], 'extensions': [], 'requirements': [],
        'recipes': [{'id': 'cpu', 'label': 'CPU arithmetic', 'path': prefix + 'cpu-recipe.json',
                     'sha256': sha(project / 'cpu-recipe.json')}]}
    write(project / 'project.json', manifest)
    dataset = Dataset(args.output / 'catalog', project_roots={'lab': args.root})
    session = dataset.access.exchange(dataset.access._local_owner_token())
    token = session['token']
    try:
        with dataset.access.bind(token):
            runtime = torch_runtime.load(args.runtime)
            TorchRuntimes(dataset.db, dataset.access, dataset.path).create(runtime, sha(args.runtime))
            manifests = {'cpu': manifest, 'train': read(args.training_project / 'project.json')}
            dataset.projects.register_snapshot({'root': 'lab', 'manifest_path': (project / 'project.json').relative_to(args.root).as_posix()})
            dataset.projects.register_snapshot({'root': 'lab', 'manifest_path': (args.training_project / 'project.json').relative_to(args.root).as_posix()})
            graphs = {}
            preflight = {}
            def graph_for(mode, inputs=None):
                doc = manifests['cpu' if mode == 'cpu' else 'train']
                recipe = next(row for row in doc['recipes'] if row['id'] == mode)
                if inputs is None:
                    required = read(args.root / recipe['path'])['inputs']
                    refs = {row['id']: row for row in doc['data']}
                    inputs = {slot: {'reference_id': slot, 'expected_sha256': refs[slot]['sha256']} for slot in required}
                return dataset.pipelines.create_external({'name': 'Scheduler qualification ' + mode,
                    'project_id': doc['id'], 'recipe_id': mode, 'expected_sha256': recipe['sha256'], 'inputs': inputs})
            for mode in ('cpu', 'train'):
                graphs[mode] = graph_for(mode)
                preflight[mode] = dataset.pipelines.preflight_external(graphs[mode]['id'], graphs[mode]['revision'])
            def submit(mode):
                graph, material = graphs[mode], preflight[mode]
                run = dataset.runs.submit_external({'request_key': 'device-overlap-' + mode + '-' + uuid.uuid4().hex,
                    'graph_id': graph['id'], 'graph_revision': graph['revision'], 'expected_digest': material['material_sha256']})
                report['runs'].append({'mode': mode, 'id': run['id'], 'material_sha256': run['material_sha256']})
                return run['id']
            cpu_id = submit('cpu')
            train_id = submit('train')
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                current = [dataset.runs.get(run_id) for run_id in (cpu_id, train_id)]
                report['observations'].append({'elapsed_seconds': time.perf_counter() - started,
                    'states': [run['state'] for run in current],
                    'custody_phases': [run['custody_phase'] for run in current]})
                if all(run['state'] == 'completed' for run in current):
                    break
                if any(run['state'] in ('failed', 'cancelled', 'unknown', 'interrupted') for run in current):
                    raise RuntimeError('Device qualification Run failed; inspect retained catalog')
                time.sleep(.25)
            else:
                raise TimeoutError('Actual device overlap qualification timed out')
            report['execution_resources'] = [run['execution_resources'] for run in current]
            cpu = json.loads(dataset.runs.artifact(cpu_id, 'receipt.json')[0])
            trained = json.loads(dataset.runs.artifact(train_id, 'receipt.json')[0])
            history = json.loads(dataset.runs.artifact(train_id, 'history.json')[0])
            assert cpu['device'] == 'cpu' and not cpu['cuda_initialized'] and cpu['optimizer_updates'] == 0
            assert trained['mode'] == 'train' and trained['optimizer_updates'] == 1 and trained['execution']['device'] == 'cuda:0'
            intersection = min(cpu['compute_finished_unix'], history['training_finished_unix']) - max(cpu['compute_started_unix'], history['training_started_unix'])
            assert intersection > 0, 'real CUDA optimizer work did not overlap actual CPU computation'
            report.update(cpu_compute=cpu, cuda_execution=trained['execution'], cuda_training_interval={
                key: history[key] for key in ('training_started_unix', 'training_finished_unix')},
                actual_optimizer_cpu_overlap_seconds=intersection)
            refs = {row['id']: row for row in manifests['train']['data']}
            restore = read(args.training_project / 'restore-recipe.json')
            artifacts = {row['name']: row for row in current[1]['artifacts']}
            inputs = {}
            for slot in restore['inputs']:
                if slot in refs:
                    inputs[slot] = {'reference_id': slot, 'expected_sha256': refs[slot]['sha256']}
                else:
                    name = next(name for name in artifacts if 'trained_' + name.replace('-', '_').replace('.', '_') == slot)
                    inputs[slot] = {'kind': 'run_artifact', 'run_id': train_id, 'artifact_name': name, 'expected_sha256': artifacts[name]['sha256']}
            graphs['restore'] = graph_for('restore', inputs)
            preflight['restore'] = dataset.pipelines.preflight_external(graphs['restore']['id'], graphs['restore']['revision'])
            restore_id = submit('restore')
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                restored_run = dataset.runs.get(restore_id)
                if restored_run['state'] == 'completed':
                    break
                if restored_run['state'] in ('failed', 'cancelled', 'unknown', 'interrupted'):
                    raise RuntimeError('Fresh CPU restore failed')
                time.sleep(.25)
            else:
                raise TimeoutError('Fresh CPU restore timed out')
            restored = json.loads(dataset.runs.artifact(restore_id, 'receipt.json')[0])
            assert restored['mode'] == 'restore' and restored['optimizer_updates'] == 0 and not restored['execution']['cuda_initialized']
            for name in ('weights.npz', 'normalization.json', 'history.json'):
                assert hashlib.sha256(dataset.runs.artifact(restore_id, name)[0]).hexdigest() == artifacts[name]['sha256']
            report.update(cpu_restore=restored['execution'], restored_artifacts_exact=True)
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        dataset.close()
        report['all_executor_workers_joined'] = not dataset.runs._thread or not dataset.runs._thread.is_alive()
        report['elapsed_seconds'] = time.perf_counter() - started
        report['wall_monotonic_offset_change_seconds'] = time.time() - time.monotonic() - clock_offset
        write(args.output / 'receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('status', 'elapsed_seconds', 'actual_optimizer_cpu_overlap_seconds')}))


if __name__ == '__main__':
    main()
