"""Build an isolated native momentum-readout teacher probe against the qualified Rheon copy."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from study_support import ROOT,HERE,BASE,read,write,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise ValueError('Fresh native teacher build required')
    prior=BASE/'temporal-build-02';probe=output/'probe';(probe/'src').mkdir(parents=True)
    old_name='rheon-residual-rollout';name='rheon-momentum-teacher-probe'
    for filename in ('Cargo.toml','Cargo.lock'):
        text=(prior/'probe'/filename).read_text().replace(old_name,name)
        (probe/filename).write_text(text)
    shutil.copyfile(HERE/'momentum_teacher_probe.rs',probe/'src/main.rs')
    env={k:v for k,v in os.environ.items()if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')['environment'])
    env.update(CARGO_TARGET_DIR=str(prior/'target'),CARGO_BUILD_JOBS='2',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    cargo=str(Path(env['RUSTC']).with_name('cargo'))
    command=[cargo,'build','--manifest-path',str(probe/'Cargo.toml'),'--config',
        'patch."https://github.com/MrScripty/Rheon.git".rheon.path='+__import__('json').dumps(str(prior/'rheon')),
        '--offline','--locked','--release']
    with (output/'build.log').open('xb')as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError('Native momentum teacher build failed; retained build log')
    binary=output/name;shutil.copyfile(prior/'target/release'/name,binary);binary.chmod(0o555)
    write(output/'build-receipt.json',dict(status='built',binary_sha256=sha(binary),
        source_sha256=sha(HERE/'momentum_teacher_probe.rs'),controller_sha256=sha(__file__),
        native_operator_sha256=sha(prior/'rheon/src/operator.rs'),prior_build_receipt_sha256=sha(prior/'build-receipt.json'),
        log_sha256=sha(output/'build.log'),native_steps_executed=0,pressure_solves=0))
    print('Built isolated native momentum-readout teacher probe; no pressure solve',flush=True)
