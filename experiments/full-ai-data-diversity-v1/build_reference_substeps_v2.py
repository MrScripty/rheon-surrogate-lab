"""Build reference substeps with unchanged numerical methods and acceptance gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Fresh isolated diversity reference build required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered native source changed: ' + name)
    prior = ROOT / reg['predecessor']
    old = read(prior / 'build-receipt.json')
    if old['status'] != 'qualified_build' or digest(prior / 'rheon-temporal-probe') != old['binary_sha256']:
        raise ValueError('Qualified predecessor required')
    output.mkdir(parents=True)
    for directory in ('rheon', 'probe'):
        shutil.copytree(prior / directory, output / directory, ignore=shutil.ignore_patterns('target'))
    shutil.copyfile(HERE / 'scheduled_forcing_substeps.rs', output / 'rheon/src/diagnostic/scheduled_forcing.rs')
    engine = output / 'rheon/src/diagnostic/engine.rs'
    source = engine.read_text()
    original = 'use super::scene::{DT, MU, RHO, Scene};'
    if source.count(original) != 1:
        raise ValueError('Exact timestep import seam required')
    source = source.replace(original, 'use super::scene::{MU, RHO, Scene};')
    import re
    source, count = re.subn(r'\bDT\b', 'super::scheduled_forcing::dt()', source)
    if count != 20:
        raise ValueError('Expected exactly20 qualified physical timestep uses, got '+str(count))
    engine.write_text(source)
    driver = output / 'probe/src/main.rs'
    source = driver.read_text()
    original = """    let controls=object.remove("forcing_schedule");
    object.remove("forcing_program");
    let scene:Scene=serde_json::from_value(raw_scene)?;
    if let Some(value)=controls {
        rheon::diagnostic::scheduled_forcing::configure(serde_json::from_value(value)?,scene.steps)
            .map_err(anyhow::Error::msg)?;
    }"""
    replacement = """    let controls=object.remove("forcing_schedule");
    object.remove("forcing_program");
    let factor=match object.remove("reference_substeps") {
        Some(value)=>value.as_u64().ok_or_else(||anyhow::anyhow!("reference substeps integer"))? as usize,
        None=>1,
    };
    let mut scene:Scene=serde_json::from_value(raw_scene)?;
    let frames=match controls {Some(value)=>Some(serde_json::from_value(value)?),None=>None};
    rheon::diagnostic::scheduled_forcing::configure(frames,scene.steps,factor)
        .map_err(anyhow::Error::msg)?;
    scene.steps=scene.steps.checked_mul(factor).ok_or_else(||anyhow::anyhow!("fine-step overflow"))?;"""
    if source.count(original) != 1:
        raise ValueError('Exact scheduled-clock protocol seam required')
    source = source.replace(original, replacement)
    source = source.replace('ensure!(scene.steps<=600,"bounded temporal scene");',
                            'ensure!(scene.steps<=3600,"bounded temporal fine-step count");')
    source = source.replace('session.state_step()as f64/60.',
        'session.state_step()as f64/(60.*rheon::diagnostic::scheduled_forcing::substeps()as f64)')
    source = source.replace('"dt":1./60.', '"dt":rheon::diagnostic::scheduled_forcing::dt()')
    driver.write_text(source)
    (output / 'target').symlink_to(prior / 'target', target_is_directory=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('CARGO_', 'RHEON_', 'RUSTFLAGS', 'RUSTC', 'RUSTDOC', 'RUSTUP_TOOLCHAIN'))}
    env.update(read(ROOT / 'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')['environment'])
    env.update(CARGO_TARGET_DIR=str(output / 'target'), CARGO_BUILD_JOBS='2',
               RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='')
    cargo = str(Path(env['RUSTC']).with_name('cargo'))
    common = ['--manifest-path', str(output / 'probe/Cargo.toml'), '--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path=' + json.dumps(str(output / 'rheon')),
        '--offline', '--locked', '--release']
    commands = [('build', [cargo, 'build', *common]), ('probe-tests', [cargo, 'test', *common]),
        ('native-tests', [cargo, 'test', '--manifest-path', str(output / 'rheon/Cargo.toml'),
            '--offline', '--locked', '--release', '--no-default-features', '--features', 'diagnostic', '--lib']),
        ('clippy', [cargo, 'clippy', *common, '--', '-D', 'warnings'])]
    files = [p for directory in ('rheon', 'probe') for p in (output / directory).rglob('*')
             if p.is_file() and 'target' not in p.relative_to(output).parts]
    record = dict(status='building', source_sha256=reg['source_sha256'],
        generated_sources={str(p.relative_to(ROOT)): digest(p) for p in files},
        predecessor_receipt_sha256=digest(prior / 'build-receipt.json'), commands=[],
        pressure_solver_and_acceptance_gates_unchanged=True, protocol_sha256=digest(args.protocol),
        original_source_and_binary_unchanged=True, optimizer_updates=0)
    def save():
        (output / 'build-receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        for name, command in commands:
            tick = time.perf_counter()
            with (output / (name+'.log')).open('xb') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            record['commands'].append(dict(name=name, returncode=result.returncode,
                seconds=time.perf_counter()-tick, log_sha256=digest(output / (name+'.log'))))
            save()
            if result.returncode:
                raise RuntimeError('Native qualification failed: '+name)
        binary = output / 'rheon-temporal-probe'
        shutil.copyfile(output / 'target/release/rheon-residual-rollout', binary)
        binary.chmod(0o555)
        record.update(status='qualified_build', binary_sha256=digest(binary))
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        save()
    print('Built isolated reference substep clock', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
