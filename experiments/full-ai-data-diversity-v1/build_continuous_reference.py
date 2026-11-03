"""Build fine-time external-control transport on the qualified reference clock."""
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


def replace_once(path, original, replacement):
    text = path.read_text()
    if text.count(original) != 1:
        raise ValueError('Exact isolated build seam required: ' + str(path))
    path.write_text(text.replace(original, replacement))


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Fresh isolated continuous reference build required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    prior = ROOT / reg['predecessor']
    old = read(prior / 'build-receipt.json')
    if old['status'] != 'qualified_build' or digest(prior / 'rheon-temporal-probe') != old['binary_sha256']:
        raise ValueError('Qualified predecessor required')
    output.mkdir(parents=True)
    for directory in ('rheon', 'probe'):
        shutil.copytree(prior / directory, output / directory, ignore=shutil.ignore_patterns('target'))
    shutil.copyfile(HERE / 'scheduled_forcing_fine_controls.rs',
                    output / 'rheon/src/diagnostic/scheduled_forcing.rs')
    driver = output / 'probe/src/main.rs'
    replace_once(driver, '    object.remove("forcing_program");', '''    object.remove("forcing_program");
    let control_factor=match object.remove("forcing_control_substeps") {
        Some(value)=>value.as_u64().ok_or_else(||anyhow::anyhow!("control cadence integer"))? as usize,
        None=>1,
    };
    let binary_count=match object.remove("forcing_binary_controls") {
        Some(value)=>Some(value.as_u64().ok_or_else(||anyhow::anyhow!("control count integer"))? as usize),
        None=>None,
    };''')
    replace_once(driver,
        '    let frames=match controls {Some(value)=>Some(serde_json::from_value(value)?),None=>None};\n'
        '    rheon::diagnostic::scheduled_forcing::configure(frames,scene.steps,factor)',
        '''    let frames=if let Some(count)=binary_count {
        ensure!(controls.is_none(),"only one external-control encoding permitted");
        ensure!(count<=38400 && count>0,"bounded external-control count");
        let raw=read_f64(&mut input,count*4)?;
        Some(raw.chunks_exact(4).map(|v|rheon::diagnostic::scheduled_forcing::ForceFrame {
            center:[v[0],v[1],v[2]],pulse:v[3],
        }).collect())
    } else {match controls {Some(value)=>Some(serde_json::from_value(value)?),None=>None}};
    rheon::diagnostic::scheduled_forcing::configure(frames,scene.steps,factor,control_factor)''')
    replace_once(driver, 'ensure!(scene.steps<=3600,"bounded temporal fine-step count");',
                         'ensure!(scene.steps<=38400,"bounded temporal fine-step count");')
    engine = output / 'rheon/src/diagnostic/engine.rs'
    replace_once(engine, 'if !(1..=3600).contains(&scene.steps) || scene.warmup_steps >= scene.steps {',
                         'if !(1..=38400).contains(&scene.steps) || scene.warmup_steps >= scene.steps {')
    replace_once(engine, 'Choose 1–3600 streamed steps and a shorter warmup',
                         'Choose 1–38400 streamed steps and a shorter warmup')
    replace_once(engine, 'Scene { steps: 3601, ..Default::default() },',
                         'Scene { steps: 38401, ..Default::default() },')
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
        reference_only=True, physical_time_per_output_s=1/60,
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
    print('Built fine-time external-control reference', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
