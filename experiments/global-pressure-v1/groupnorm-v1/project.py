#!/usr/bin/env python3
"""Assemble the distinct GN experiment into existing generic Tuldok contracts.

This owner never grants material, submits a Run, or executes training.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import uuid

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
ORIGINAL = HERE.parent / 'training'
MIB = 1024**2
TULDOK_PIN = 'f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6'
DATA_NAMES = {'train0': 'train-0.npz', 'train1': 'train-1.npz', 'validation': 'validation.npz',
              'splits': 'splits.json', 'origin': 'origin.json'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contracts():
    # The candidate owns the scientific schemas and inventory. Generic Tuldok
    # remains the sole owner of execution/capture contracts.
    package_name = 'rheon_groupnorm_project_sources'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(HERE)]
        sys.modules[package_name] = package
    name = package_name+'.train'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE/'train.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def definition(command, sources, snapshot):
    train = contracts()
    inputs = {slot: {'max_bytes': 256*1024 if slot == 'splits' else 32*MIB} for slot in DATA_NAMES}
    arguments = [{'literal': command}]
    for slot in DATA_NAMES:
        arguments += [{'literal': '--'+slot}, {'input': slot}]
    for name in ('config', 'runtime-manifest', 'source-witness'):
        arguments += [{'literal': '--'+name}, {'source': name+'.json'}]
    arguments += [{'literal': '--input-evidence'}, {'input_evidence': True},
                  {'literal': '--output-dir'}, {'output_dir': True}]
    if command == 'restore':
        for slot, (artifact, _role) in train.TRAINED_INPUTS.items():
            inputs[slot] = {'max_bytes': train.OUTPUTS[artifact][1]}
            arguments += [{'literal': '--'+slot.replace('_', '-')}, {'input': slot}]
    return {'schema_version': 1, 'format': 'external_recipe_v1',
            'id': 'global-pressure-groupnorm-'+command, 'name': 'Fixed GN CPU '+command,
            'version': 1, 'runtime': 'captured_cpu_python_v1', 'runtime_snapshot_sha256': snapshot,
            'sources': [{'path': name, 'sha256': sha(raw)} for name, raw in sorted(sources.items())],
            'entrypoint': 'train.py', 'arguments': arguments, 'parameters': {}, 'inputs': inputs,
            'outputs': [{'path': name, 'role': 'evaluation_result' if command == 'restore' and name == 'receipt.json' else role,
                         'max_bytes': cap, 'required': True} for name, (role, cap) in train.OUTPUTS.items()],
            'environment': {}, 'resources': {'wall_seconds': 600, 'max_output_bytes': 32*MIB}}


def assemble(args):
    train = contracts()
    numerics = train.model
    reader = numerics.load_original('data', numerics.ORIGINAL_DATA_SHA256)
    read = train.regular_bytes
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=LAB, text=True),
            'commit qualified source before immutable assembly')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LAB, text=True).strip()
    tuldok = args.tuldok_source.resolve(strict=True)
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tuldok, text=True).strip() == TULDOK_PIN
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=tuldok, text=True),
            'unchanged exact generic Tuldok pin required')
    sys.path.insert(0, str(tuldok))
    from recipe_contract import decode_definition
    from projects import manifest as generic_manifest
    packet = args.package.resolve(strict=True)
    receipt_raw = read(packet/'dataset-receipt.json', 2*MIB)
    receipt = reader.strict_json(receipt_raw)
    require(receipt['status'] == 'verified_lossless_development_packet'
            and receipt['packager_source_sha256'] == numerics.ORIGINAL_DATA_SHA256,
            'original qualified packet reader identity')
    inputs = {slot: read(packet/name, 256*1024 if slot == 'splits' else 32*MIB) for slot, name in DATA_NAMES.items()}
    for slot, raw in inputs.items():
        require(receipt['files'][DATA_NAMES[slot]] == {'sha256': sha(raw), 'bytes': len(raw)}, 'packet identity: '+slot)
    dataset = reader.load_data(*(packet/DATA_NAMES[slot] for slot in DATA_NAMES),
                               expected_native_residual_origin_sha256=receipt['native_residual_origin_sha256'])
    require(len(dataset.train_rows) == 288 and len(dataset.validation_rows) == 96 and sum(map(len, inputs.values())) <= 256*MIB,
            'complete unchanged development population/input budget')
    runtime_raw = read(args.runtime_manifest, 8*MIB)
    runtime = reader.strict_json(runtime_raw)
    require(runtime['profile'] == 'captured_cpu_python_v1'
            and runtime['packages']['torch'] == '2.8.0+cpu' and runtime['packages']['numpy'] == '2.3.5',
            'qualified captured CPU packages')
    snapshot = sha(runtime_raw)
    config_raw = read(HERE/'config.json', 65536)
    require(canonical(reader.strict_json(config_raw)) == canonical(train.FIXED_CONFIG), 'exact fixed GN configuration')
    qualification_raw = read(args.qualification_report, MIB)
    qualification = reader.strict_json(qualification_raw)
    budget_raw = read(args.output_budget, MIB)
    require(sha(budget_raw) == qualification['evidence']['output_budget_sha256'], 'qualified full output budget binding')
    sources = {name: read(HERE/name, MIB) for name in ('model.py', 'train.py')}
    sources.update({'original_'+name+'.py': read(ORIGINAL/(name+'.py'), MIB) for name in ('model', 'train', 'data')})
    hashes = {name: sha(raw) for name, raw in sources.items()}
    require(qualification['status'] == 'passed' and qualification['executable_sources'] == hashes,
            'exact qualified executable closure')
    budget = reader.strict_json(budget_raw)
    require(budget['status'] == 'passed_pretraining_conservative_projection'
            and budget['executable_sources'] == hashes and budget['optimizer_updates'] == 0
            and budget['optimizer_constructed'] is False and budget['artifact_count'] == len(train.OUTPUTS)
            and budget['aggregate']['aggregate_passed'] is True
            and budget['aggregate']['conservative_projection_bytes'] <= 32*MIB,
            'complete conservative zero-update output budget')
    native = reader.strict_json(read(args.native_build_receipt, MIB))
    require(native['status'] == 'qualified_build', 'qualified native input residual source')
    nr = native['identity']['native_receipt']
    sources.update({'config.json': config_raw, 'runtime-manifest.json': runtime_raw,
                    'model-qualification.json': qualification_raw, 'output-budget.json': budget_raw})
    witness = {'format': train.SOURCE_SCHEMA, 'source_commit': commit, 'tuldok_revision': TULDOK_PIN,
               'fixedmodel_format': numerics.ARCHITECTURE['schema'], 'runtime_snapshot_sha256': snapshot,
               'executable_sources': hashes, 'config_sha256': sha(config_raw),
               'package_inputs': {slot: {'sha256': sha(raw), 'bytes': len(raw)} for slot, raw in inputs.items()},
               'native_qualification': {'base_commit': nr['base_commit'], 'patch_sha256': nr['patch_sha256'],
                                        'prepared_tree_sha256': nr['prepared_tree_sha256'], 'operator_binary_sha256': native['binary_sha256']},
               'qualification_report_sha256': sha(qualification_raw),
               'native_residual_origin_sha256': receipt['native_residual_origin_sha256']}
    train.validate_qualification(qualification, witness)
    sources['source-witness.json'] = canonical(witness)
    require(len(sources) <= 64 and all(len(raw) <= 8*MIB for raw in sources.values())
            and sum(map(len, sources.values()))+sum(map(len, inputs.values()))+MIB <= 320*MIB,
            'source and complete capture closure budget')
    output = args.output.absolute()
    namespace = LAB/'runs/global-pressure-v1-project'
    require(output == output.resolve() and output.parent == namespace and not output.exists(),
            'fresh direct owned project child required')
    files = dict(sources)
    files.update({'inputs/'+DATA_NAMES[slot]: raw for slot, raw in inputs.items()})
    files['evidence/dataset-receipt.json'] = receipt_raw
    files['evidence/model-qualification.json'] = qualification_raw
    recipes = []
    for command in ('train', 'restore'):
        raw = canonical(definition(command, sources, snapshot)); decode_definition(raw)
        name = 'recipes/'+command+'.json'; files[name] = raw
        recipes.append({'id': command, 'label': 'Fixed GN '+command, 'path': name, 'sha256': sha(raw)})
    manifest = {'schema_version': 1, 'id': uuid.uuid4().hex, 'name': 'Global pressure fixed GroupNorm smoke',
                'data': [{'id': slot, 'label': 'Immutable development '+slot, 'path': 'inputs/'+DATA_NAMES[slot], 'sha256': sha(raw)} for slot, raw in inputs.items()],
                'recipes': recipes,
                'evidence': [{'id': 'dataset', 'label': 'Original development packet', 'path': 'evidence/dataset-receipt.json', 'sha256': sha(receipt_raw)},
                             {'id': 'qualification', 'label': 'GN pretraining qualification', 'path': 'evidence/model-qualification.json', 'sha256': sha(qualification_raw)}],
                'requirements': [{'id': 'parent-admission', 'label': 'No grant or training before independent review and parent admission'},
                                 {'id': 'final-reservations', 'label': 'Development only; final suites remain sealed'}]}
    generic_manifest(manifest)
    files['project.json'] = canonical(manifest)
    namespace.mkdir(exist_ok=True); output.mkdir()
    for name, raw in files.items():
        path = output/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        path.chmod(0o400)
    result = {'status': 'assembled_not_admitted', 'source_commit': commit, 'project_id': manifest['id'],
              'path': str(output), 'manifest_sha256': sha(files['project.json']), 'runtime_snapshot_sha256': snapshot,
              'source_bytes': sum(map(len, sources.values())), 'input_bytes': sum(map(len, inputs.values())),
              'files': {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in files.items()}}
    print(json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'runtime-manifest', 'native-build-receipt', 'qualification-report', 'output-budget', 'tuldok-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    try:
        assemble(parser.parse_args())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print('groupnorm_project_assembly_failure: '+str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
