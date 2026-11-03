"""Build an isolated single-track path without changing selected Rheon sources."""
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

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(output):
    if output.exists():
        raise ValueError('Fresh build directory required')
    prior = ROOT/'runs/pressure-methods-v1/drift-observation-02'
    binding = json.loads((prior/'binding.json').read_text())
    for path, expected in binding['captured_sources'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Qualified observation source changed: '+path)
    output.mkdir(parents=True)
    shutil.copytree(prior/'rheon', output/'rheon', ignore=shutil.ignore_patterns('target'))
    shutil.copytree(prior/'probe', output/'probe', ignore=shutil.ignore_patterns('target'))
    engine = output/'rheon/src/diagnostic/engine.rs'
    engine.write_text(engine.read_text()+'\n'+(HERE/'single_track.rs').read_text())
    module = output/'rheon/src/diagnostic/mod.rs'
    text = module.read_text()
    if text.count('mod engine;') != 1:
        raise ValueError('Engine module seam changed')
    module.write_text(text.replace('mod engine;', 'pub mod engine;'))
    shutil.copyfile(HERE/'single_driver.rs', output/'probe/src/main.rs')
    prior_build = json.loads((ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json').read_text())
    env = {k:v for k,v in os.environ.items() if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(prior_build['environment'])
    env.update(CARGO_TARGET_DIR=str(output/'target'), CARGO_BUILD_JOBS='2', RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='')
    cargo = str(Path(env['RUSTC']).with_name('cargo'))
    common = ['--manifest-path',str(output/'probe/Cargo.toml'),'--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')),
        '--offline','--locked','--release']
    commands = [('build',[cargo,'build',*common]),
        ('tests',[cargo,'test','--manifest-path',str(output/'rheon/Cargo.toml'),'--offline','--locked',
            '--release','--no-default-features','--features','diagnostic','--lib','diagnostic::']),
        ('clippy',[cargo,'clippy',*common,'--','-D','warnings'])]
    receipt = dict(status='building', prior_binding_sha256=sha(prior/'binding.json'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.rs'),Path(__file__)]}, commands=[])
    for name,command in commands:
        tick = time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        receipt['commands'].append(dict(name=name,elapsed_s=time.perf_counter()-tick,exit_code=0,log_sha256=sha(output/(name+'.log'))))
    binary = output/'rheon-single-track'
    shutil.copyfile(output/'target/release/rheon-residual-rollout',binary)
    binary.chmod(0o555)
    receipt.update(status='qualified_build',binary_sha256=sha(binary),
        changed_original_sources=False, reference_solves_during_candidate_steps=0)
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','binary_sha256']}),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output.resolve())
