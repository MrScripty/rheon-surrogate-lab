"""Capture a versioned large-grid/substep native producer without changing history.

Derive the producer from the qualified local source with exact, audited edits.
Build inputs, source edits and dependency bytes remain bound in the output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAB = ROOT/'runs/dependency-migration-v1/lab'
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


BUILD_RS = '''use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};
fn main() {
    let binding_path = env::var("RHEON_BUILD_BINDING").expect("explicit binding");
    let binding: serde_json::Value = serde_json::from_slice(&fs::read(&binding_path).unwrap()).unwrap();
    for (path, expected) in binding["source_sha256"].as_object().unwrap() {
        assert_eq!(format!("{:x}", Sha256::digest(fs::read(path).unwrap())), expected.as_str().unwrap());
        println!("cargo:rerun-if-changed={path}");
    }
    for key in ["RHEON_BUILD_BINDING", "RHEON_CAPTURE_RESERVATION"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let reservation = env::var("RHEON_CAPTURE_RESERVATION").unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(fs::read(&reservation).unwrap())), binding["reservation_sha256"].as_str().unwrap());
    println!("cargo:rerun-if-changed={reservation}");
    let identity = serde_json::json!({"binding":binding,"native_threads":1});
    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("identity.json"), serde_json::to_vec(&identity).unwrap()).unwrap();
}
'''


def replacement(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Audited native source edit no longer unique: '+old)
    return text.replace(old, new)


def prepare(output):
    if output.exists():
        raise ValueError('Fresh native build directory required')
    original = LAB/'probes/sealed-water-v1'
    source = (original/'src/main.rs').read_text()
    edits = [
        ('<= 65_536', '<= 262_144'),
        ('let steps = if request.fixture {', 'let substeps = entry["teacher_substeps"].as_u64().unwrap_or(1) as usize;\n    ensure!([1, 2, 4].contains(&substeps), "declared substeps");\n    let steps = if request.fixture {'),
        ('config["steps"].as_u64().context("steps")? as usize', '(config["steps"].as_u64().context("steps")? as usize) * substeps'),
        ('serde_json::from_value(config["snapshot_steps"].clone())?', 'serde_json::from_value::<Vec<usize>>(config["snapshot_steps"].clone())?.into_iter().map(|n| (n + 1) * substeps - 1).collect()'),
        ('steps <= 64 &&', 'steps <= 256 &&'),
        ('let dt = config["dt_s"].as_f64().context("time step")?;', 'let base_dt = config["dt_s"].as_f64().context("time step")?;\n    ensure!(base_dt == 1.0 / 60.0, "qualified model interval");\n    let dt = base_dt / substeps as f64;'),
        ('ensure!(dt == 1.0 / 60.0, "qualified time step");', 'ensure!(dt > 0.0 && dt.is_finite(), "positive finite teacher step");'),
    ]
    for old, new in edits:
        source = replacement(source, old, new)
    # Tests verify the changed boundaries, without changing the native physics gates.
    source += '''
#[cfg(test)]
mod campaign_boundaries {
    use super::*;
    #[test]
    fn accepts_64_cube_and_rejects_next_cube() {
        assert!(cubic_grid([4.0; 3], 16).is_ok());
        assert!(cubic_grid([65.0/16.0; 3], 16).is_err());
    }
    #[test]
    fn substeps_match_physical_snapshot_times() {
        for substeps in [1usize, 2, 4] {
            for coarse_index in [15usize, 63] {
                let fine_index = (coarse_index + 1) * substeps - 1;
                let actual = (fine_index + 1) as f64 / (60.0 * substeps as f64);
                assert_eq!(actual, (coarse_index + 1) as f64 / 60.0);
            }
        }
    }
}
'''
    original_manifest = json.loads((ROOT/'runs/dependency-migration-v1/training-project-6/inputs/splits.json').read_text())
    parent = next(p for p in original_manifest['parents'] if p['entry']['split'] == 'train')
    config = dict(forcing_profile='compact-gaussian-taper-v1', steps=64,
        snapshot_steps=[15, 63], dt_s=1/60, material=dict(density_kg_m3=1000., dynamic_viscosity_pa_s=.001,
        boundary='sealed_impermeable_free_slip'))
    rows = []
    for size in [32, 64]:
        for spacing in [8, 16]:
            rows.append(dict(trajectory_id=f'large-train-{size}-cpm{spacing}', split='train',
                counts_xyz=[size]*3, cells_per_metre=spacing, teacher_substeps=1,
                force=parent['scene']['force'], seed=2026101800+size+spacing))
    for substeps in [1, 2, 4]:
        rows.append(dict(trajectory_id=f'temporal-pilot-substeps{substeps}', split='train',
            counts_xyz=[16]*3, cells_per_metre=16, teacher_substeps=substeps,
            force=parent['scene']['force'], seed=2026101801))
    for spacing in [4, 32]:
        rows.append(dict(trajectory_id=f'spacing-pilot-cpm{spacing}', split='train',
            counts_xyz=[16]*3, cells_per_metre=spacing, teacher_substeps=1,
            force=parent['scene']['force'], seed=2026101801))
    reservation = dict(schema='rheon-pressure-methods-reservation-v1', configuration=config, scenes=rows,
        scope='Training-only qualification pilots; no validation or reserved-test scenes; temporal pressures are not coarse-step labels.')
    output.mkdir(parents=True)
    crate = output/'crate'
    (crate/'src').mkdir(parents=True)
    (crate/'src/main.rs').write_text(source)
    (crate/'build.rs').write_text(BUILD_RS)
    for name in ['Cargo.toml', 'Cargo.lock']:
        shutil.copyfile(original/name, crate/name)
    write(output/'reservation.json', reservation)
    current_build = read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')
    selection_path = Path(current_build['dependency_config'])
    selection = read(selection_path)
    rheon_source = Path(selection['sources']['rheon']['root'])
    source_paths = [crate/'src/main.rs', crate/'build.rs', crate/'Cargo.toml', crate/'Cargo.lock', Path(__file__)]
    # Bind all actual current dependency sources independently of historical labels.
    dependency_paths = [p for p in rheon_source.rglob('*') if p.is_file() and
                        (p.suffix == '.rs' or p.name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'))
                        and 'target' not in p.relative_to(rheon_source).parts and '.git' not in p.relative_to(rheon_source).parts]
    binding = dict(schema='rheon-pressure-methods-native-v1',
        reservation_sha256=sha(output/'reservation.json'), source_sha256={str(p):sha(p) for p in source_paths+dependency_paths},
        original_source=str(original/'src/main.rs'), original_source_sha256=sha(original/'src/main.rs'),
        edits=[dict(original=old, replacement=new) for old, new in edits],
        rheon_source=str(rheon_source), selection_sha256=sha(selection_path),
        physical_gates='Unchanged native residual, predicted/actual divergence, wall flux, net flux, Courant and viscosity gates.')
    write(output/'binding.json', binding)
    print('Prepared native producer and', len(rows), 'pilot reservations', flush=True)
    return output


def build(output):
    binding = read(output/'binding.json')
    for path, expected in binding['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Bound native source changed: '+path)
    original = read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')
    env = {k:v for k, v in os.environ.items() if not k.startswith(
        ('CARGO_', 'RHEON_', 'RUSTFLAGS', 'RUSTC', 'RUSTDOC', 'RUSTUP_TOOLCHAIN'))}
    env.update(original['environment'])
    env.update(RHEON_BUILD_BINDING=str(output/'binding.json'),
               RHEON_CAPTURE_RESERVATION=str(output/'reservation.json'), CARGO_BUILD_JOBS='2')
    cargo = original['commands'][0]['command'][0]
    common = ['--manifest-path', str(output/'crate/Cargo.toml'), '--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(binding['rheon_source']),
        '--offline', '--locked', '--release']
    records = []
    for operation in ['build', 'test', 'clippy']:
        command = [cargo, operation, *common, *(['--', '-D', 'warnings'] if operation == 'clippy' else [])]
        tick = time.perf_counter()
        with (output/(operation+'.log')).open('xb') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        records.append(dict(command=command, elapsed_s=time.perf_counter()-tick, exit_code=0))
    binary = output/'rheon-pressure-methods-capture'
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/rheon-sealed-water-capture', binary)
    binary.chmod(0o555)
    identity = json.loads(subprocess.check_output([str(binary), '--identity'], timeout=10))
    if identity != dict(binding=binding, native_threads=1):
        raise ValueError('Native embedded identity mismatch')
    write(output/'build-receipt.json', dict(status='qualified_build', binary_sha256=sha(binary),
        identity=identity, commands=records, source_sha256=sha(__file__)))
    print('Qualified extended native producer', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'build'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output.resolve())
    else:
        build(args.output.resolve())
