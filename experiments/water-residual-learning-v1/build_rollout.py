"""Build the public diagnostic probe against configuration-selected current Rheon."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / 'experiments/local-pressure-v1'))
import dependency


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    selection = dependency.selection(args.dependency_config)
    source = selection.source_root('rheon')
    native = dependency.verify(source, selection)
    cargo = selection.path_for('cargo')
    args.output.mkdir(parents=True, exist_ok=False)
    crate = LAB / 'probes/residual-rollout-v1'
    inputs = [crate / name for name in ('Cargo.toml', 'Cargo.lock', 'src/main.rs')]
    inputs += [Path(__file__).resolve(), LAB / 'experiments/local-pressure-v1/dependency.py', LAB / 'experiments/local-pressure-v1/contract.py',
               LAB / 'experiments/dependency-migration-v1/dependency_config.py']
    hashes = {str(path): digest(path) for path in inputs}
    environment = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CARGO_', 'RHEON_', 'RUSTFLAGS', 'RUSTC', 'RUSTDOC', 'RUSTUP_TOOLCHAIN'))}
    environment.update(CARGO_HOME=str(selection.path_for('cargo_home')),
        CARGO_TARGET_DIR=str(selection.path_for('cargo_target')), CARGO_BUILD_JOBS='2',
        RUSTUP_TOOLCHAIN='1.92.0', RUSTC=str(cargo.parent/'rustc'), RUSTDOC=str(cargo.parent/'rustdoc'),
        RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='', RAYON_NUM_THREADS='1')
    common = ['--manifest-path', str(crate/'Cargo.toml'), '--config',
        f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(source))}', '--offline', '--locked']
    metadata = json.loads(subprocess.check_output([str(cargo), 'metadata', *common, '--format-version', '1'], env=environment))
    selected = [p for p in metadata['packages'] if p['name'] == 'rheon']
    if not (len(selected) == 1 and selected[0]['source'] is None
            and Path(selected[0]['manifest_path']) == source/'Cargo.toml'):
        raise ValueError('Cargo graph did not select configured current Rheon')
    commands = []
    for operation in ('build', 'test', 'clippy'):
        command = [str(cargo), operation, *common, '--release', *(['--', '-D', 'warnings'] if operation == 'clippy' else [])]
        started = time.monotonic()
        with (args.output/(operation+'.log')).open('wb') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        commands.append({'command': command, 'elapsed_s': time.monotonic()-started, 'exit_code': 0})
    if hashes != {str(path): digest(path) for path in inputs} or native != dependency.verify(source, selection):
        raise ValueError('Source changed during probe qualification')
    binary = args.output/'rheon-residual-rollout'
    shutil.copyfile(Path(environment['CARGO_TARGET_DIR'])/'release/rheon-residual-rollout', binary)
    binary.chmod(0o555)
    receipt = {'schema': 'rheon-current-residual-rollout-build-v1', 'status': 'qualified_build',
        'source_sha256': hashes, 'native_receipt': native, 'commands': commands,
        'dependency_config': str(selection.path), 'environment': {k: environment[k] for k in
            ('CARGO_HOME', 'CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'RUSTUP_TOOLCHAIN', 'RUSTC', 'RUSTDOC', 'RAYON_NUM_THREADS')},
        'cargo_version': subprocess.check_output([str(cargo), '--version'], env=environment, text=True).strip(),
        'toolchain': {'rustc': subprocess.check_output([str(cargo.parent/'rustc'), '-vV'], env=environment, text=True).strip(),
            'executables': {str(path): digest(path) for path in (cargo, cargo.parent/'rustc', cargo.parent/'rustdoc')}},
        'binary_sha256': digest(binary), 'metadata_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        'execution_policy': 'independent_stop_on_reject', 'native_fallback': False, 'native_pressure_refinement': False}
    (args.output/'build-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.absolute()
    print(json.dumps(build(args)), flush=True)


if __name__ == '__main__':
    main()
