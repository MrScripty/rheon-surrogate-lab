"""Assemble one guided full-population CUDA fit and four-checkpoint CPU restore."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

from managed_input_stream_v1 import streamed_identity

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(p):
    return streamed_identity(p)['sha256']


def run(args):
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Registered Project preparation source changed: ' + name)
    staging = Path(reg['project_root'])
    project = staging / reg['project_subdirectory']
    if project.exists():
        raise ValueError('Fresh temporary Project directory required')
    project.mkdir()
    config = read(ROOT / reg['config'])
    source_dir = ROOT / reg['model_source_capture']
    model_manifest = read(source_dir / 'manifest.json')
    additions = [HERE / name for name in ('guided_pressure_fit_core_v1.py',
        'guided_full_population_optimizer_v1.py', 'train_guided_pressure_v1.py')]
    with zipfile.ZipFile(source_dir / 'model-source.zip') as source, zipfile.ZipFile(project / 'model-source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        for path in additions:
            member = str(path.relative_to(ROOT))
            if member in model_manifest['members']:
                if model_manifest['members'][member] != sha(path):
                    raise ValueError('Existing captured fitting helper differs')
            else:
                target.write(path, member)
                model_manifest['members'][member] = sha(path)
    bank = read(ROOT / reg['transport'])
    if Path(bank['archive']['path']) != staging / 'fields.zip':
        raise ValueError('Single existing prepared archive required')
    bank_manifest = read(ROOT / reg['bank_manifest'])
    manifest = dict(schema='rheon.guided-pressure-managed-manifest.v1',
        reference_eligibility=read(ROOT / reg['eligibility']), qualified_index=read(ROOT / reg['qualified_index']),
        objective_qualification=read(ROOT / reg['objective_qualification']),
        source_manifest=model_manifest, model_source_sha256=sha(project / 'model-source.zip'),
        normalization_sha256=sha(ROOT / reg['normalization']),
        archive={n: bank['archive'][n] for n in ('bytes', 'sha256')},
        bank_semantic_sha256=hashlib.sha256(json.dumps(bank_manifest, sort_keys=True).encode()).hexdigest(),
        forbidden_repository_root=str(ROOT))
    write(project / 'manifest.json', manifest)
    shutil.copyfile(ROOT / reg['config'], project / 'config.json')
    shutil.copyfile(ROOT / reg['normalization'], project / 'normalization.json')
    shutil.copyfile(ROOT / reg['runtime_manifest'], project / 'runtime-manifest.json')
    for name in ('train_guided_pressure_v1.py', 'portable_shared_runtime_v1.py', 'managed_input_stream_v1.py'):
        shutil.copyfile(HERE / name, project / name)
    shutil.copyfile(ROOT / 'experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py', project / 'managed_fit_artifacts.py')
    shutil.copyfile(ROOT / 'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py', project / 'custody_validator.py')
    prefix = reg['project_subdirectory'] + '/'
    source_names = ['train_guided_pressure_v1.py', 'portable_shared_runtime_v1.py', 'managed_input_stream_v1.py',
                    'managed_fit_artifacts.py', 'custody_validator.py', 'config.json']
    sources = [dict(path=prefix + name, sha256=sha(project / name)) for name in source_names]
    static = dict(manifest=project / 'manifest.json', normalization=project / 'normalization.json',
        runtime_manifest=project / 'runtime-manifest.json', model_source=project / 'model-source.zip', part_00=staging / 'fields.zip')
    outputs = [(name, 'checkpoint', 128*1024) for name in
        ('weights.npz', 'best-mean-weights.npz', 'initial-weights.npz', 'last-evaluated-weights.npz')]
    outputs += [('history.json','training_diagnostics',8*1024**2), ('probe.npy','predictions',24*1024**2),
        ('normalization.json','preprocessing',1024**2), ('receipt.json','training_result',4*1024**2)]
    artifact_map = dict(trained_weights=('weights.npz',128*1024), trained_mean=('best-mean-weights.npz',128*1024),
        trained_initial=('initial-weights.npz',128*1024), trained_last=('last-evaluated-weights.npz',128*1024),
        trained_history=('history.json',8*1024**2), trained_probe=('probe.npy',24*1024**2), trained_receipt=('receipt.json',4*1024**2))
    recipes = []
    for mode in ('train', 'restore'):
        inputs = {n: dict(max_bytes=3*1024**3 if n == 'part_00' else 16*1024**2)
                  for n in static if mode == 'train' or n != 'part_00'}
        arguments = [dict(literal=mode), dict(literal='--config'), dict(source=prefix + 'config.json')]
        for name in ('manifest', 'normalization', 'runtime_manifest', 'model_source'):
            arguments.extend([dict(literal='--' + name.replace('_', '-')), dict(input=name)])
        arguments.extend([dict(literal='--input-evidence'), dict(input_evidence=True), dict(literal='--output-dir'), dict(output_dir=True)])
        if mode == 'train':
            arguments.extend([dict(literal='--shards'), dict(input='part_00')])
        else:
            for name, (_, cap) in artifact_map.items():
                inputs[name] = dict(max_bytes=cap)
                arguments.extend([dict(literal='--' + name.replace('_', '-')), dict(input=name)])
        declared = [dict(path=name, role='evaluation_result' if mode == 'restore' and name == 'receipt.json' else role,
            max_bytes=cap, required=True) for name, role, cap in outputs]
        definition = dict(schema_version=1, format='external_recipe_v1', id=config['identifier'] + '-' + mode,
            name=config['identifier'] + ' ' + mode, version=1, runtime='managed_torch_python_v1',
            runtime_snapshot_sha256=sha(static['runtime_manifest']), runtime_device='cuda' if mode == 'train' else 'cpu',
            sources=sources, entrypoint=prefix + 'train_guided_pressure_v1.py', arguments=arguments,
            parameters={}, inputs=inputs, outputs=declared, environment={},
            resources=dict(wall_seconds=config['wall_seconds'] if mode == 'train' else 300,
                           max_output_bytes=config['maximum_output_bytes'], cpu_cores=1,
                           memory_bytes=4*1024**3 if mode == 'train' else 1024**3))
        if mode == 'train':
            definition['capture_policy'] = dict(format='streamed_project_inputs_v1', max_input_file_bytes=3*1024**3,
                max_input_bytes=4*1024**3, max_capture_bytes=4*1024**3 + 16*1024**2)
        write(project / (mode + '-recipe.json'), definition)
        recipes.append(dict(id=mode, label=mode, path=prefix + mode + '-recipe.json', sha256=sha(project / (mode + '-recipe.json'))))
    document = dict(schema_version=2, id=uuid.uuid4().hex, name='Complete24scene independent pressure residual width4 full-population CUDA fit',
        data=[dict(id=n, label=n, path=str(p.relative_to(staging)), sha256=sha(p)) for n, p in static.items()],
        recipes=recipes, evidence=[], requirements=[], extensions=[])
    write(project / 'project.json', document)
    sys.path.insert(0, str(ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'))
    import projects, recipe_contract
    projects.manifest(document)
    for mode in ('train', 'restore'):
        recipe_contract.decode_definition((project / (mode + '-recipe.json')).read_bytes())
    write(project / 'assembly.json', dict(status='prepared_complete24scene_guided_managed_Project',
        project_root_alias='long-training', project_root=str(staging), project=prefix + 'project.json',
        project_sha256=sha(project / 'project.json'), sources=sources, static_inputs={n: dict(path=str(p), **streamed_identity(p)) for n, p in static.items()},
        config=config, registration=str(args.protocol.resolve().relative_to(ROOT)), registration_sha256=sha(args.protocol),
        restore_artifact_map={n: pair[0] for n, pair in artifact_map.items()}, new_parameter_fits=0, cpu_model_forward_allowed=False))
    print('Prepared guided width4 Project and all4checkpoint CPU restore; recipe syntax validated; no fits submitted.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    run(parser.parse_args())
