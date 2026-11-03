"""Build a separate scheduled-force native reference and retain original binaries."""
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
    shutil.copyfile(HERE / 'scheduled_forcing.rs', output / 'rheon/src/diagnostic/scheduled_forcing.rs')
    module = output / 'rheon/src/diagnostic/mod.rs'
    module.write_text(module.read_text() + '\npub mod scheduled_forcing;\n')
    engine = output / 'rheon/src/diagnostic/engine.rs'
    source = engine.read_text()
    original = '''    let center = std::array::from_fn::<_, 3, _>(|d| {
        extents[d] * (scene.center[d] + 0.05 * (omega * time + phase + d as f64).sin())
    });
    let pulse = 0.65 + 0.35 * (omega * time + phase).sin();'''
    replacement = '''    let control = super::scheduled_forcing::sample(time)?;
    let center = match control {
        Some(frame) => std::array::from_fn::<_, 3, _>(|d| extents[d] * frame.center[d]),
        None => std::array::from_fn::<_, 3, _>(|d| {
            extents[d] * (scene.center[d] + 0.05 * (omega * time + phase + d as f64).sin())
        }),
    };
    let pulse = match control {
        Some(frame) => frame.pulse,
        None => 0.65 + 0.35 * (omega * time + phase).sin(),
    };'''
    if source.count(original) != 1:
        raise ValueError('Exact qualified force extension seam required')
    engine.write_text(source.replace(original, replacement) + '\n' + (HERE / 'forcing_preview_v2.rs').read_text())
    driver = output / 'probe/src/main.rs'
    source = driver.read_text()
    original = '    let scene:Scene=serde_json::from_value(line(&mut input)?)?;'
    replacement = '''    let mut raw_scene=line(&mut input)?;
    let object=raw_scene.as_object_mut().ok_or_else(||anyhow::anyhow!("scene object"))?;
    let controls=object.remove("forcing_schedule");
    object.remove("forcing_program");
    let scene:Scene=serde_json::from_value(raw_scene)?;
    if let Some(value)=controls {
        rheon::diagnostic::scheduled_forcing::configure(serde_json::from_value(value)?,scene.steps)
            .map_err(anyhow::Error::msg)?;
    }'''
    if source.count(original) != 1:
        raise ValueError('Exact scene protocol extension seam required')
    source = source.replace(original, replacement)
    original = '            Some("advance") => {'
    replacement = '''            Some("forcing") => {
                let step=command["step"].as_u64().ok_or_else(||anyhow::anyhow!("force step"))? as usize;
                let mut velocity=[Vec::new(),Vec::new(),Vec::new()];
                for (axis,value)in velocity.iter_mut().enumerate() {
                    let size=(scene.counts[axis]+1)*scene.counts.iter().enumerate()
                        .filter(|(a,_)|*a!=axis).map(|(_,n)|n).product::<usize>();
                    *value=read_f32(&mut input,size)?;
                }
                let values=session.preview_prescribed_force(step,velocity).map_err(anyhow::Error::msg)?;
                send(&mut out,&json!({"kind":"forcing","step":step,"state_unchanged":true}))?;
                for value in values { f32_field(&mut out,&value)?; }
                out.flush()?;
            }
            Some("advance") => {'''
    if source.count(original) != 1:
        raise ValueError('Exact forcing-preview protocol seam required')
    driver.write_text(source.replace(original, replacement))
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
    print('Built isolated scheduled-force native reference', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
