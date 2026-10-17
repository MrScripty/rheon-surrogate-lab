"""Offline, explicitly invoked conversion of the two frozen pressure packages.

Never imported by the adapter. Originals are copied byte-for-byte to a separate
archive, then mapped to a fresh current descriptor. No inference or training runs.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil

from model_adapter import STOPPING, STAGE, measure, require, strict

HERE = Path(__file__).resolve().parent
CONTROL_WEIGHTS = '26b66232928ce81d428dbdb1c8cc750a0cf5c0bc5f678977efc111a0a56d2d9a'
CONTROL_NORMALIZATION = 'f73b6cf32108f63cbd8d5edef8b08afce2e3a05d47234886458ddfbc99e1d1c1'
AA82_WEIGHTS = 'aa82bd122fde9406d9e6b92114b3e6ba77281f5ee8b9c726aa168dacbd93a62d'
AA82_NORMALIZATION = 'bee9be00a94f0241bc8baf6df2499314918b0a9e708b3f995bea5046c40a3296'
AA82_DESCRIPTOR = 'be4eef1f051206205598e6876c83edaf5246138e17613f1aa787f3df183a761f'
CONTROL_WITNESS = '2a2653c68fb6ce1f5ef15794e4441d559203d05fc46e341e8f5c78545488fbef'
ITERATION_SOURCE = 'a14273b0a07e9885f01182695be6d3a82fab84583963b1119681cefc454e4cd5'


def decode(path):
    return strict(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def archive_files(sources, archive):
    """Archive only explicitly selected originals; refuse any existing output."""
    archive.mkdir(parents=True, exist_ok=False)
    identities = {}
    for relative, source in sorted(sources.items()):
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(source.is_file() and not source.is_symlink(), 'Regular original file required')
        shutil.copyfile(source, destination)
        identity = measure(source)
        require(identity == measure(destination), 'Archive verification failed')
        identities[relative] = {'source': str(source), **identity}
    write(archive / 'archive-inventory.json', {'schema': 'rheon-model-original-archive-v1', 'files': identities})
    return identities


def iteration_functions(path):
    """Retain exact policy functions, omitting the obsolete dataset/CLI decoder."""
    source = path.read_text()
    tree = ast.parse(source)
    wanted = {'validate_budget', 'stopping_policy', 'equation_quality', 'proposal_width',
              'predict_proposal', 'correct'}
    parts = {node.name: ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in wanted}
    require(set(parts) == wanted, 'Complete original iterative policy functions required')
    return ('"""Exact frozen CPU correction/stopping functions; no dataset CLI."""\n'
            'import math\nMAX_NETWORK_EVALUATIONS=32\n\n'+'\n\n\n'.join(parts.values())+'\n')


def package(kind, sources, artifacts, inventory, runtime, output, archive, originals, provenance, iteration=None, runtime_owner=None):
    require(not output.exists() and not archive.exists(), 'Fresh package and archive destinations required')
    require(runtime_owner is not None and runtime_owner.is_file(), 'Explicit current runtime owner required')
    owner_spec = importlib.util.spec_from_file_location('rheon_packaging_torch_runtime', runtime_owner)
    current_owner = importlib.util.module_from_spec(owner_spec)
    owner_spec.loader.exec_module(current_owner)
    runtime_value = current_owner.load(runtime)
    originals['runtime/current-manifest.json'] = runtime
    originals['runtime/torch_runtime.py'] = runtime_owner
    archived = archive_files(originals, archive)
    output.mkdir(parents=True, exist_ok=False)
    source_mapping = {}
    numerical_sources = ['portable_model.py', 'frozen_gn.py', 'original_model.py']
    if kind == 'physical_residual':
        numerical_sources.append('physical_residual.py')
    for name in numerical_sources:
        shutil.copyfile(sources / name, output / name)
        source_mapping[name] = {'source': str(sources / name), **measure(output / name)}
    for name in ('weights.npz', 'normalization.json'):
        shutil.copyfile(artifacts / name, output / name)
    shutil.copyfile(runtime, output / 'installed-runtime.json')
    shutil.copyfile(runtime_owner, output / 'torch_runtime.py')
    for name in ('model_adapter.py', 'model_adapter_numerics.py'):
        shutil.copyfile(HERE / name, output / name)
    shutil.copyfile(HERE / 'model_adapter_policy.py', output / 'policy.py')
    if iteration is not None:
        (output / 'learned_iteration.py').write_text(iteration_functions(iteration))
        source_mapping['learned_iteration.py'] = {'source': str(iteration),
            'source_identity': measure(iteration), 'conversion': 'exact six policy function source segments',
            **measure(output / 'learned_iteration.py')}
    numerical = {'schema': 'rheon-pressure-numerics-v1', 'kind': kind,
                 'correction': {'activation': 'linear', 'steps': 2} if kind == 'physical_residual' else None,
                 'weights': inventory,
                 'inference': {'max_network_calls': 32, 'single_evaluation': True,
                               'minimum_network_evaluations': 0, 'stopping': STOPPING}
                              if kind == 'physical_residual' else {'max_network_calls': 1}}
    write(output / 'numerical.json', numerical)
    write(output / 'provenance.json', {'schema': 'rheon-model-package-provenance-v1',
        'role': 'unchanged migration comparator; no training or promotion', 'original': provenance,
        'archive_inventory': str(archive / 'archive-inventory.json'),
        'archive_inventory_identity': measure(archive / 'archive-inventory.json'),
        'numerical_sources': source_mapping, 'weights': measure(output / 'weights.npz'),
        'normalization': measure(output / 'normalization.json'), 'runtime_manifest': measure(runtime),
        'runtime_owner': {'source': str(runtime_owner), **measure(runtime_owner)}})
    python = Path(runtime_value['python'])
    manifest = {'schema': 'rheon-pressure-adapter-v1',
        'id': 'rheon-water-control-26b' if kind == 'physical_residual' else 'rheon-water-aa82',
        'version': 'migration-20261016', 'stage': STAGE,
        'python': {'executable': str(python), **measure(python)}, 'entrypoint': 'model_adapter.py',
        'files': {p.name: measure(p) for p in sorted(output.iterdir())},
        'support': {'density_kg_m3': 1000., 'dynamic_viscosity_pa_s': .001, 'dt_s': 1/60,
            'counts_min': [5, 5, 5], 'counts_max': [128, 128, 128], 'max_cells': 65536,
            'spacing_min_m': 1/128, 'spacing_max_m': 1., 'origin': 'zero', 'cells': 'cubic'},
        'context': {'minimum_native_steps': 15, 'prior_pressure': 'required'},
        'inputs': {'base': 'previous_pressure', 'residual': 'initial_residual', 'rho': 'density', 'dt': 'dt'},
        'output': {'role': 'full_pressure', 'dtype': 'f64', 'layout': 'x_fast', 'gauge': 'cell_zero'}}
    write(output / 'manifest.json', manifest)
    receipt = {'schema': 'rheon-model-package-conversion-v1', 'status': 'packaged_not_numerically_qualified',
               'manifest': str(output / 'manifest.json'), 'manifest_identity': measure(output / 'manifest.json'),
               'weights': measure(output / 'weights.npz'), 'normalization': measure(output / 'normalization.json'),
               'archive': str(archive), 'original_file_count': len(archived), 'source_mapping': source_mapping}
    write(archive / 'package-conversion.json', receipt)
    return receipt


def convert_aa82(bundle, runtime, output, archive, runtime_owner=None):
    descriptor_path = bundle / 'deployment.json'
    descriptor = decode(descriptor_path)
    require(measure(descriptor_path)['sha256'] == AA82_DESCRIPTOR, 'Frozen original deployment descriptor required')
    require(descriptor['schema'] == 'rheon-pressure-cpu-deployment-v2', 'Offline source must be original v2 bundle')
    require(descriptor['checkpoint_sha256'] == AA82_WEIGHTS, 'Fixed original aa82 checkpoint required')
    require(descriptor['context_policy']['minimum_native_steps'] == 15, 'Original native warm-up required')
    for relative, identity in descriptor['files'].items():
        require((bundle / relative).resolve().is_relative_to(bundle.resolve())
                and measure(bundle / relative) == identity, 'Original bundle inventory drift')
    require(measure(bundle / 'weights.npz')['sha256'] == AA82_WEIGHTS
            and measure(bundle / 'normalization.json')['sha256'] == AA82_NORMALIZATION
            and descriptor['normalization_sha256'] == AA82_NORMALIZATION,
            'Original aa82 numerical identity drift')
    history = decode(bundle / 'history.json')
    inventory = history['checkpoints'][f'checkpoint-{descriptor["training_updates"]:03d}.npz']['inventory']
    originals = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    return package('seven_channel', bundle / 'sources', bundle, inventory, runtime, output, archive,
                   originals, {'deployment': measure(descriptor_path), 'training_run_id': descriptor['training_run_id']},
                   runtime_owner=runtime_owner)


def convert_control(project, artifacts, runtime, iteration, output, archive, runtime_owner=None):
    require(measure(artifacts / 'weights.npz')['sha256'] == CONTROL_WEIGHTS
            and measure(artifacts / 'normalization.json')['sha256'] == CONTROL_NORMALIZATION,
            'Frozen 26b control numerical identities required')
    require(measure(project / 'source-witness.json')['sha256'] == CONTROL_WITNESS
            and measure(iteration)['sha256'] == ITERATION_SOURCE, 'Frozen control sources/policy required')
    witness = decode(project / 'source-witness.json')
    for name, digest in witness['sources'].items():
        require(measure(project / name)['sha256'] == digest, 'Original model source witness drift')
    config = decode(project / 'config.json')
    require(config['conditioning'] == 'physical-residual' and config['normalization'] == 'none'
            and config['correction'] == {'activation': 'linear', 'steps': 2}, 'Frozen control mapping required')
    receipt = decode(artifacts / 'receipt.json')
    require(receipt['mode'] == 'restore' and receipt['optimizer_updates'] == 0
            and receipt['execution']['cuda_initialized'] is False, 'Original CPU restore required')
    require(receipt['closure']['source_witness_sha256'] == measure(project / 'source-witness.json')['sha256']
            and receipt['closure']['experiment_sha256'] == measure(project / 'config.json')['sha256'],
            'Original CPU restore source/config custody drift')
    for name in ('weights.npz', 'normalization.json', 'history.json'):
        require(measure(artifacts / name)['sha256'] == receipt['outputs'][name]['sha256'], 'Original restore artifact drift')
    history = decode(artifacts / 'history.json')
    inventory = history['checkpoints'][f'checkpoint-{config["training"]["updates"]:03d}.npz']['inventory']
    originals = {'sources/' + name: project / name for name in witness['sources']}
    originals.update({'sources/' + name: project / name for name in ('config.json', 'source-witness.json', 'runtime-manifest.json')})
    originals.update({'artifacts/' + name: artifacts / name for name in ('weights.npz', 'normalization.json', 'history.json', 'receipt.json')})
    originals['policy/learned_iteration.py'] = iteration
    return package('physical_residual', project, artifacts, inventory, runtime, output, archive,
                   originals, {'source_witness': measure(project / 'source-witness.json'),
                               'cpu_restore_receipt': measure(artifacts / 'receipt.json')}, iteration, runtime_owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    for kind in ('aa82', 'control'):
        command = sub.add_parser(kind)
        for name in ('runtime', 'runtime-owner', 'output', 'archive'):
            command.add_argument('--' + name, type=Path, required=True)
        if kind == 'aa82':
            command.add_argument('--bundle', type=Path, required=True)
        else:
            for name in ('project', 'artifacts', 'iteration'):
                command.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'aa82':
        receipt = convert_aa82(args.bundle.absolute(), args.runtime.absolute(), args.output.absolute(), args.archive.absolute(), args.runtime_owner.absolute())
    else:
        receipt = convert_control(args.project.absolute(), args.artifacts.absolute(), args.runtime.absolute(),
                                  args.iteration.absolute(), args.output.absolute(), args.archive.absolute(), args.runtime_owner.absolute())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
