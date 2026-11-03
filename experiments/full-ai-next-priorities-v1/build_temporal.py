"""Build an isolated native state/restart probe and exercise real integrator restart."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from study_support import ROOT,HERE,BASE,read,write,sha

def build(output):
    output=output.resolve()
    if output.exists():raise ValueError('Fresh native build required')
    prior=ROOT/'runs/pressure-methods-v4/streamed-build-01'
    old=read(prior/'build-receipt.json')
    for name,digest in old['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Prior native source changed')
    if sha(prior/'rheon-large-single-track')!=old['binary_sha256']:raise ValueError('Prior binary changed')
    output.mkdir(parents=True)
    for directory in ('rheon','probe'):
        shutil.copytree(prior/directory,output/directory,ignore=shutil.ignore_patterns('target'))
    engine=output/'rheon/src/diagnostic/engine.rs'
    engine.write_text(engine.read_text()+'\n'+(HERE/'restart_methods.rs').read_text())
    shutil.copyfile(HERE/'temporal_driver.rs',output/'probe/src/main.rs')
    (output/'target').symlink_to(prior/'target',target_is_directory=True)
    env={k:v for k,v in os.environ.items()if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')['environment'])
    env.update(CARGO_TARGET_DIR=str(output/'target'),CARGO_BUILD_JOBS='2',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    cargo=str(Path(env['RUSTC']).with_name('cargo'))
    common=['--manifest-path',str(output/'probe/Cargo.toml'),'--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')),
        '--offline','--locked','--release']
    commands=[('build',[cargo,'build',*common]),('probe-tests',[cargo,'test',*common]),
        ('native-tests',[cargo,'test','--manifest-path',str(output/'rheon/Cargo.toml'),'--offline','--locked',
                         '--release','--no-default-features','--features','diagnostic','--lib']),
        ('clippy',[cargo,'clippy',*common,'--','-D','warnings'])]
    files=[engine,output/'probe/src/main.rs',HERE/'restart_methods.rs',HERE/'temporal_driver.rs',Path(__file__)]
    record=dict(status='building',commands=[],sources={str(p.relative_to(ROOT)):sha(p)for p in files},
        predecessor_receipt_sha256=sha(prior/'build-receipt.json'),original_pressure_gates_unchanged=True,
        original_source_and_binary_unchanged=True)
    write(output/'build-receipt.json',record)
    try:
        for name,command in commands:
            tick=time.perf_counter()
            with (output/(name+'.log')).open('xb')as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            record['commands'].append(dict(name=name,returncode=result.returncode,seconds=time.perf_counter()-tick,log_sha256=sha(output/(name+'.log'))))
            write(output/'build-receipt.json',record)
            if result.returncode:raise RuntimeError('Native qualification failed: '+name)
        binary=output/'rheon-temporal-probe';shutil.copyfile(output/'target/release/rheon-residual-rollout',binary);binary.chmod(0o555)
        record.update(status='qualified_build',binary_sha256=sha(binary),restart_tolerance='bitwise equality for eight future steps at each of two native grid shapes')
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:write(output/'build-receipt.json',record)
    print('Qualified native temporal capture and restart build',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
