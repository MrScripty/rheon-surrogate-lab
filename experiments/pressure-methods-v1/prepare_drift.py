"""Build isolated Rheon observations for velocity/tracer drift; preserve live sources."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha

METHOD = '''    /// Observations of committed independent tracks; no simulation mutation.
    pub fn trajectory_drift(&self) -> (f64, f64, f64, f64) {
        fn measure<'a>(pairs: impl Iterator<Item = (&'a f32, &'a f32)>) -> (f64, f64) {
            let mut error = 0.0_f64;
            let mut signal = 0.0_f64;
            let mut maximum = 0.0_f64;
            for (candidate, reference) in pairs {
                let reference = f64::from(*reference);
                let difference = f64::from(*candidate) - reference;
                error += difference * difference;
                signal += reference * reference;
                maximum = maximum.max(difference.abs());
            }
            (error.sqrt() / signal.sqrt().max(1e-30), maximum)
        }
        let velocity = measure(self.ai.velocity.iter().flatten().zip(self.reference.velocity.iter().flatten()));
        let tracer = measure(self.ai.tracer.iter().zip(&self.reference.tracer));
        (velocity.0, tracer.0, velocity.1, tracer.1)
    }
'''

TESTS = '''
#[cfg(test)]
mod trajectory_drift_observation_tests {
    use super::*;
    #[test]
    fn equal_tracks_have_zero_drift_and_observation_is_read_only() {
        let engine = DiagnosticSession::new(Scene::default()).unwrap();
        let before = engine.ai.tracer.clone();
        assert_eq!(engine.trajectory_drift(), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(engine.trajectory_drift(), (0.0, 0.0, 0.0, 0.0));
        assert_eq!(engine.ai.tracer, before);
    }
    #[test]
    fn observed_velocity_and_tracer_norms_match_known_perturbations() {
        let mut engine = DiagnosticSession::new(Scene::default()).unwrap();
        engine.reference.velocity[0][1] = 1.0;
        engine.ai.velocity[0][1] = 2.0;
        engine.reference.tracer.fill(0.0);
        engine.ai.tracer.fill(0.0);
        engine.reference.tracer[0] = 1.0;
        engine.ai.tracer[0] = 0.5;
        assert_eq!(engine.trajectory_drift(), (1.0, 0.5, 1.0, 0.5));
    }
}
'''


def replace(text,before,after):
    if text.count(before) != 1:
        raise ValueError('Drift observation source seam changed')
    return text.replace(before,after)


def prepare(output):
    if output.exists():
        raise ValueError('Fresh isolated observation build required')
    prior_path = ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json'
    prior = read(prior_path)
    source = ROOT/'runs/dependency-migration-v1/Rheon'
    sources = {str(source/row['path']):row['sha256'] for row in prior['native_receipt']['build_inputs']}
    for path,expected in sources.items():
        if sha(path) != expected:
            raise ValueError('Qualified original Rheon source changed')
    output.mkdir()
    isolated = output/'rheon'
    isolated.mkdir()
    for name in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:
        shutil.copyfile(source/name,isolated/name)
    shutil.copytree(source/'src',isolated/'src')
    shutil.copytree(source/'examples',isolated/'examples')
    engine = isolated/'src/diagnostic/engine.rs'
    original = engine.read_text()
    (output/'engine-original.rs').write_text(original)
    engine.write_text(replace(original,'    pub fn initial(&self) -> Frame {',METHOD+'    pub fn initial(&self) -> Frame {')+TESTS)
    scene = isolated/'src/diagnostic/scene.rs'
    original_scene = scene.read_text()
    scene.write_text(replace(replace(replace(replace(replace(original_scene,'pub const HISTORY_BYTES: usize = 96 * 1024 * 1024;',
        'pub const HISTORY_BYTES: usize = 384 * 1024 * 1024;'),'!(1..=512).contains(&self.steps)',
        '!(1..=3600).contains(&self.steps)'),'and 1–512 steps;','and 1–3600 steps;'),
        '> 65_536','> 262_144'),'65,536 cells total','262,144 cells total').replace('96 MiB;','384 MiB;'))
    crate = output/'probe'
    original_crate = ROOT/'runs/dependency-migration-v1/lab/probes/residual-rollout-v1'
    shutil.copytree(original_crate,crate,ignore=shutil.ignore_patterns('target','__pycache__'))
    driver = crate/'src/main.rs'
    text = driver.read_text()
    text = replace(text,'        if let Some(frame) = result.frame {',
        '        if let Some(frame) = result.frame {\n            let drift = session.trajectory_drift();')
    text = replace(text,'"pressure_comparison_scope":"independent_trajectories",',
        '"pressure_comparison_scope":"independent_trajectories",\n'
        '                "velocity_trajectory_relative_l2":drift.0,"tracer_trajectory_relative_l2":drift.1,\n'
        '                "velocity_trajectory_max_abs_m_s":drift.2,"tracer_trajectory_max_abs":drift.3,')
    driver.write_text(text)
    copied = [*isolated.rglob('*.rs'),isolated/'Cargo.toml',isolated/'Cargo.lock',isolated/'rust-toolchain.toml',
              crate/'Cargo.toml',crate/'Cargo.lock',driver]
    write(output/'binding.json',dict(status='prepared',original_sources=sources,
        captured_sources={str(p.relative_to(ROOT)):sha(p) for p in copied},source_sha256=sha(__file__),
        prior_build_receipt_sha256=sha(prior_path),changes=[
            'Read-only committed-track velocity/tracer norms and matching wire fields',
            'Versioned scene bounds: 3600 steps, 262144 cells and paired-history bound 384 MiB; physical gates and numerical steps unchanged'],
        original_selected_rheon_modified=False))
    print('Prepared isolated drift observation crate',flush=True)


def build(output):
    binding = read(output/'binding.json')
    for path,expected in {**binding['original_sources'],**binding['captured_sources']}.items():
        if sha(ROOT/path if not Path(path).is_absolute() else path) != expected:
            raise ValueError('Bound drift source changed')
    prior = read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    environment.update(prior['environment'])
    environment.update(CARGO_TARGET_DIR=str(output/'target'),CARGO_BUILD_JOBS='2',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    cargo = str(Path(environment['RUSTC']).with_name('cargo'))
    common = ['--manifest-path',str(output/'probe/Cargo.toml'),'--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')),'--offline','--locked','--release']
    commands = []
    for name,command in [
        ('build',[cargo,'build',*common]),
        ('diagnostic-tests',[cargo,'test','--manifest-path',str(output/'rheon/Cargo.toml'),'--offline','--locked',
                             '--release','--no-default-features','--features','diagnostic','--lib','diagnostic::']),
        ('clippy',[cargo,'clippy',*common,'--','-D','warnings'])]:
        tick = time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        commands.append(dict(name=name,command=command,elapsed_s=time.perf_counter()-tick,exit_code=0))
    for path,expected in binding['captured_sources'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Source changed during build')
    binary = output/'rheon-drift-rollout'
    shutil.copyfile(output/'target/release/rheon-residual-rollout',binary)
    binary.chmod(0o555)
    write(output/'build-receipt.json',dict(status='qualified_build',binding_sha256=sha(output/'binding.json'),
        binary_sha256=sha(binary),commands=commands,original_selected_rheon_modified=False))
    print('Qualified isolated drift observation binary',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','build'])
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    (prepare if args.action=='prepare' else build)(args.output.resolve())
