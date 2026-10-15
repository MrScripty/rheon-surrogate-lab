"""Build the laboratory trajectory probe against explicitly selected Rheon sources."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

LAB=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    sys.path.insert(0,str(LAB/'experiments/local-pressure-v1'))
    import dependency
    native=dependency.verify(args.prepared)
    args.output.mkdir(parents=True,exist_ok=False)
    crate=LAB/'probes/residual-rollout-v1'
    inputs=[crate/name for name in ('Cargo.toml','Cargo.lock','build.rs','src/main.rs')]
    inputs += [args.rheon_source/'src/bin/desktop/ai_comparison'/(name+'.rs') for name in ('engine','scene')]
    inputs += [Path(__file__).resolve()]
    hashes={str(path):digest(path) for path in inputs}
    environment={k:v for k,v in os.environ.items() if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    environment.update(CARGO_HOME=str(LAB/'.cargo-local'),CARGO_TARGET_DIR=str(LAB/'runs/local-pressure-native-target'),
        RUSTUP_TOOLCHAIN='1.92.0',RUSTC=str(args.cargo.parent/'rustc'),RUSTDOC=str(args.cargo.parent/'rustdoc'),
        RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='',RAYON_NUM_THREADS='1',RHEON_COMPARISON_SOURCE=str(args.rheon_source))
    common=['--manifest-path',str(crate/'Cargo.toml'),'--config',f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(args.prepared))}','--offline','--locked']
    metadata=json.loads(subprocess.check_output([str(args.cargo),'metadata',*common,'--format-version','1'],env=environment))
    selected=[p for p in metadata['packages'] if p['name']=='rheon']
    assert len(selected)==1 and selected[0]['source'] is None and Path(selected[0]['manifest_path'])==args.prepared/'Cargo.toml'
    commands=[]
    for operation in ('build','test','clippy'):
        command=[str(args.cargo),operation,*common,'--release',*(['--','-D','warnings'] if operation=='clippy' else [])]
        with (args.output/(operation+'.log')).open('wb') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        commands.append(command)
    assert hashes=={str(path):digest(path) for path in inputs} and native==dependency.verify(args.prepared)
    binary=args.output/'rheon-residual-rollout';shutil.copyfile(Path(environment['CARGO_TARGET_DIR'])/'release/rheon-residual-rollout',binary);binary.chmod(0o555)
    receipt={'schema':'rheon-residual-rollout-build-v1','status':'qualified_build','source_sha256':hashes,
        'native_receipt':native,'commands':commands,'environment':{k:environment[k] for k in ('CARGO_HOME','CARGO_TARGET_DIR','RUSTUP_TOOLCHAIN','RUSTC','RUSTDOC','RAYON_NUM_THREADS','RHEON_COMPARISON_SOURCE')},
        'cargo_version':subprocess.check_output([str(args.cargo),'--version'],env=environment,text=True).strip(),
        'binary_sha256':digest(binary),'metadata_sha256':hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()}
    (args.output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','rheon-source','cargo','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for key,value in vars(args).items():setattr(args,key,value.absolute())
    print(json.dumps(build(args)),flush=True)


if __name__=='__main__':main()
