"""Build an isolated instrumented native engine, retaining its numerical order."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha

def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Profile seam changed: '+before[:100])
    return text.replace(before, after)

def build(output):
    if output.exists():
        raise ValueError('Fresh profiling build required')
    prior = ROOT/'runs/pressure-methods-v2/large-native-build-01'
    receipt = read(prior/'build-receipt.json')
    if receipt['status'] != 'qualified_build' or sha(prior/'rheon-large-single-track') != receipt['binary_sha256']:
        raise ValueError('Qualified predecessor required')
    for path, expected in receipt['sources'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Predecessor source changed')
    output.mkdir(parents=True)
    for directory in ['rheon', 'probe']:
        shutil.copytree(prior/directory, output/directory, ignore=shutil.ignore_patterns('target'))
    engine = output/'rheon/src/diagnostic/engine.rs'
    original = engine.read_text()
    start = original.index('fn prepare(')
    end = original.index('\nfn finish(', start)
    plain = original[start:end]
    profiled = plain
    profiled = replace(profiled, '    let mut pre =', '    let tick = std::time::Instant::now();\n    let mut pre =')
    profiled = replace(profiled, '    advect_velocity(', '    let allocation = tick.elapsed().as_secs_f64();\n    let tick = std::time::Instant::now();\n    advect_velocity(')
    profiled = replace(profiled, '    advect_tracer(', '    let advect_velocity = tick.elapsed().as_secs_f64();\n    let tick = std::time::Instant::now();\n    advect_tracer(')
    profiled = replace(profiled, '    force(scene, grid, &mut pre, time, cancel)?;', '    let advect_tracer = tick.elapsed().as_secs_f64();\n    let tick = std::time::Instant::now();\n    force(scene, grid, &mut pre, time, cancel)?;\n    let forcing = tick.elapsed().as_secs_f64();\n    let tick = std::time::Instant::now();')
    profiled = replace(profiled, '    let operator =', '    let viscosity = tick.elapsed().as_secs_f64();\n    let tick = std::time::Instant::now();\n    let operator =')
    profiled = replace(profiled, '    Ok(Prepared {', '    let rhs = tick.elapsed().as_secs_f64();\n    LAST_PREPARATION.with(|profile| profile.set(PreparationProfile { allocation, advect_velocity, advect_tracer, forcing, viscosity, rhs }));\n    Ok(Prepared {')
    # Avoid shadowing the RHS field with its timing value.
    profiled = profiled.replace('let rhs = tick.elapsed()', 'let rhs_seconds = tick.elapsed()').replace('viscosity, rhs }', 'viscosity, rhs: rhs_seconds }')
    declarations = '''
/// Seconds spent in the last successful preparation on this thread.
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct PreparationProfile {
    pub allocation: f64,
    pub advect_velocity: f64,
    pub advect_tracer: f64,
    pub forcing: f64,
    pub viscosity: f64,
    pub rhs: f64,
}
thread_local! {
    static LAST_PREPARATION: std::cell::Cell<PreparationProfile> = std::cell::Cell::new(PreparationProfile::default());
}
pub fn last_preparation_profile() -> PreparationProfile {
    LAST_PREPARATION.with(std::cell::Cell::get)
}
'''
    tests = '''
#[cfg(test)]
mod profile_parity_tests {
    use super::*;
    #[test]
    fn timed_preparation_preserves_all_fields_bitwise() {
        for counts in [[8, 7, 5], [16, 16, 16]] {
            let scene = Scene { counts, steps: 2, warmup_steps: 0, ..Default::default() };
            let session = SingleTrackSession::new(scene.clone()).unwrap();
            let mut v1 = ViscosityWorkspace::new(session.grid.clone(), 128*1024*1024).unwrap();
            let mut v2 = ViscosityWorkspace::new(session.grid.clone(), 128*1024*1024).unwrap();
            for time in [0., 0.5] {
                let a = prepare(&scene, &session.grid, &session.track, &mut v1, time, &AtomicBool::new(false)).unwrap();
                let b = prepare_unprofiled(&scene, &session.grid, &session.track, &mut v2, time, &AtomicBool::new(false)).unwrap();
                assert_eq!(a.velocity, b.velocity);
                assert_eq!(a.tracer, b.tracer);
                assert_eq!(a.rhs, b.rhs);
            }
        }
    }
}
'''
    engine.write_text(original[:start]+declarations+profiled+'\n#[cfg(test)]\n'+plain.replace('fn prepare(', 'fn prepare_unprofiled(', 1)+original[end:]+tests)
    scene = output/'rheon/src/diagnostic/scene.rs'
    scene.write_text(scene.read_text().replace('768 * 1024 * 1024', '1024 * 1024 * 1024').replace('768 MiB', '1024 MiB').replace('s.steps = 1024;', 's.steps = 2048;'))
    driver = output/'probe/src/main.rs'
    driver.write_text((ROOT/'experiments/pressure-methods-v2/large_driver.rs').read_text()
        .replace('scene.steps <= 4', 'scene.steps <= 32')
        .replace('bounded large-grid coupling probe, not a long rollout', 'bounded 32-step streamed capture')
        .replace('"frame":step}', '"frame":step, "preparation_profile":rheon::diagnostic::engine::last_preparation_profile()}'))
    (output/'target').symlink_to(prior/'target', target_is_directory=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('CARGO_', 'RHEON_', 'RUSTFLAGS', 'RUSTC', 'RUSTDOC', 'RUSTUP_TOOLCHAIN'))}
    env.update(read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')['environment'])
    env.update(CARGO_TARGET_DIR=str(output/'target'), CARGO_BUILD_JOBS='2', RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='')
    cargo = str(Path(env['RUSTC']).with_name('cargo'))
    common = ['--manifest-path', str(output/'probe/Cargo.toml'), '--config', 'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')), '--offline', '--locked', '--release']
    commands = [('build', [cargo, 'build', *common]), ('wire-tests', [cargo, 'test', *common]),
        ('diagnostic-tests', [cargo, 'test', '--manifest-path', str(output/'rheon/Cargo.toml'), '--offline', '--locked', '--release', '--no-default-features', '--features', 'diagnostic', '--lib', 'diagnostic::']),
        ('clippy', [cargo, 'clippy', *common, '--', '-D', 'warnings'])]
    sources = {str(p.relative_to(ROOT)):sha(p) for p in [engine, scene, driver, Path(__file__)]}
    result = dict(status='building', predecessor_receipt_sha256=sha(prior/'build-receipt.json'), sources=sources, commands=[], physical_gates_unchanged=True, original_selected_rheon_modified=False)
    write(output/'build-receipt.json', result)
    for name, command in commands:
        tick = time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        result['commands'].append(dict(name=name, exit_code=0, seconds=time.perf_counter()-tick, log_sha256=sha(output/(name+'.log'))))
    binary = output/'rheon-large-single-track'
    shutil.copyfile(output/'target/release/rheon-residual-rollout', binary)
    binary.chmod(0o555)
    result.update(status='qualified_build', binary_sha256=sha(binary))
    write(output/'build-receipt.json', result)
    print('Qualified profiled single-track build', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
