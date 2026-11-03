"""Extend isolated single-track resource bounds; preserve physical gates and steps."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def replace(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError('Large-grid source seam changed: '+before)
    return text.replace(before,after)

def prepare(output):
    if output.exists():raise ValueError('Fresh isolated large-grid build required')
    prior=ROOT/'runs/pressure-methods-v2/single-build-01'
    qualified=read(prior/'build-receipt.json')
    if qualified['status']!='qualified_build' or sha(prior/'rheon-single-track')!=qualified['binary_sha256']:
        raise ValueError('Qualified single-track predecessor required')
    output.mkdir(parents=True)
    shutil.copytree(prior/'rheon',output/'rheon',ignore=shutil.ignore_patterns('target'))
    shutil.copytree(prior/'probe',output/'probe',ignore=shutil.ignore_patterns('target'))
    scene=output/'rheon/src/diagnostic/scene.rs'
    text=scene.read_text()
    for before,after in [('384 * 1024 * 1024','768 * 1024 * 1024'),
                         ('(5..=128)','(5..=256)'),('> 262_144','> 16_777_216'),
                         ('5–128 cells per axis, at most 262,144 cells total','5–256 cells per axis, at most 16,777,216 cells total'),
                         ('384 MiB','768 MiB'),('s.steps = 512;','s.steps = 1024;')]:
        text=replace(text,before,after)
    scene.write_text(text)
    engine=output/'rheon/src/diagnostic/engine.rs'
    engine.write_text(replace(engine.read_text(),'128 * 1024 * 1024','2 * 1024 * 1024 * 1024',count=2))
    shutil.copyfile(HERE/'large_driver.rs',output/'probe/src/main.rs')
    # Compiler cache is not a scientific artifact; retained qualified binaries are copies.
    (output/'target').symlink_to(prior/'target',target_is_directory=True)
    original=read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')
    env={k:v for k,v in os.environ.items() if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(original['environment']);env.update(CARGO_TARGET_DIR=str(output/'target'),CARGO_BUILD_JOBS='2',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    cargo=str(Path(env['RUSTC']).with_name('cargo'))
    common=['--manifest-path',str(output/'probe/Cargo.toml'),'--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')),'--offline','--locked','--release']
    commands=[('build',[cargo,'build',*common]),('wire-tests',[cargo,'test',*common]),
        ('diagnostic-tests',[cargo,'test','--manifest-path',str(output/'rheon/Cargo.toml'),'--offline','--locked','--release',
            '--no-default-features','--features','diagnostic','--lib','diagnostic::']),
        ('clippy',[cargo,'clippy',*common,'--','-D','warnings'])]
    sources=[engine,scene,output/'rheon/src/diagnostic/mod.rs',output/'probe/src/main.rs',Path(__file__)]
    binding={str(p.relative_to(ROOT)):sha(p) for p in sources}
    receipt=dict(status='building',predecessor_receipt_sha256=sha(prior/'build-receipt.json'),sources=binding,commands=[],
        changes=['Isolated grid ceiling 256^3, streamed paired-history declaration 768 MiB, pressure/viscosity allocation ceilings 2 GiB each',
                 'Bounded 1–4 fluid steps; little-endian f64 binary request/reply transport',
                 'No physical tolerance, timestep, advection, forcing, viscosity or pressure operator changes'])
    for name,command in commands:
        tick=time.perf_counter()
        with (output/(name+'.log')).open('xb') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        receipt['commands'].append(dict(name=name,exit_code=0,elapsed_s=time.perf_counter()-tick,log_sha256=sha(output/(name+'.log'))))
    if any(sha(ROOT/p)!=expected for p,expected in binding.items()):raise ValueError('Build sources changed')
    binary=output/'rheon-large-single-track'
    shutil.copyfile(output/'target/release/rheon-residual-rollout',binary);binary.chmod(0o555)
    receipt.update(status='qualified_build',binary_sha256=sha(binary),original_selected_rheon_modified=False)
    write(output/'build-receipt.json',receipt)
    print('Qualified isolated 256^3 single-track binary and physical/wire tests',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
