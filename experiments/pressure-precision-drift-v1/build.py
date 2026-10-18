"""Specialize measured Rheon storage kernels without changing production source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(source,signature):
    """Extract one balanced Rust function, ignoring comments and string braces."""
    if source.count(signature)!=1:raise ValueError('Ambiguous source seam '+signature)
    start=source.index(signature);opening=source.index('{',start);depth=0
    tokens=re.finditer(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/|[{}]',source[opening:])
    for token in tokens:
        if token[0]=='{':depth+=1
        elif token[0]=='}':
            depth-=1
            if depth==0:return source[start:opening+token.end()]
    raise ValueError('Unclosed function '+signature)


def specialize(source):
    # Preserve original numerical policy constants, including f32::MAX gates.
    return re.sub(r'\bf32\b(?!::)','f64',source).replace('_f32','_f64')


HELPER='''
    /// Research-only native observation; original preparation/solve stay unchanged.
    pub fn precision_advance_reference(&mut self,cancel:&AtomicBool)->Result<crate::precision_diagnostic::Snapshot32,String>{
        if self.halted||self.step>=self.scene.steps{return Err("Precision reference complete".into())}
        let time=self.step as f64*DT;
        let prepared=prepare(&self.scene,&self.grid,&self.reference,&mut self.viscosity,time,cancel)?;
        let(next,quality)=native_solve(&self.grid,&prepared,&mut self.solver,cancel)?;
        if !quality.qualified{return Err("Native precision reference rejected".into())}
        let casts=crate::precision_diagnostic::observe_casts(&self.scene,&self.grid,&self.reference.velocity,
            &self.reference.tracer,&prepared.velocity,&prepared.tracer,&next.pressure,&next.velocity,time,cancel,force)?;
        self.reference=next;self.step+=1;self.clock=self.step as f64*DT;
        Ok(crate::precision_diagnostic::Snapshot32{velocity:self.reference.velocity.clone(),tracer:self.reference.tracer.clone(),
            pressure:self.reference.pressure.clone(),quality,casts})
    }
    /// Immutable full fields for bitwise canonical-path qualification.
    pub fn precision_reference_velocity(&self)->[&[f32];3]{refs(&self.reference.velocity)}
    /// Legacy raw sessions keep only the current state; extend this research stream.
    /// Geometry/physics admission still occurs through the original constructor.
    pub fn precision_set_horizon(&mut self,steps:usize)->Result<(),String>{
        if self.step!=0||steps<self.scene.steps||steps>3600{return Err("Invalid research streaming horizon".into())}
        self.scene.steps=steps;Ok(())
    }
'''


def prepare(root,config_path,run):
    config=read(config_path);source=root/config['source'];baseline=read(root/config['baseline_build'])
    run.mkdir(parents=True,exist_ok=False);copy=run/'source';copy.mkdir()
    original={}
    for item in baseline['native_receipt']['build_inputs']:
        path=source/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Baseline build source changed: '+item['path'])
        target=copy/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if sha(target)!=item['sha256']:raise ValueError('Source changed during copy: '+item['path'])
        original[item['path']]=item['sha256']
    shutil.copyfile(source/'LICENSE',copy/'LICENSE')
    engine_path=copy/'src/diagnostic/engine.rs';engine=engine_path.read_text()
    marker='impl DiagnosticSession {'
    if engine.count(marker)!=1:raise ValueError('Native engine owner changed')
    engine_path.write_text(engine.replace(marker,marker+HELPER))
    operator_path=copy/'src/operator.rs';operator=operator_path.read_text();methods=[]
    for name,visibility in [('build_rhs','pub'),('divergence','pub'),('correct_velocity','pub'),('validate_velocity','pub(crate)')]:
        method=specialize(function(operator,visibility+' fn '+name+'('))
        method=method.replace('fn '+name+'(','fn '+name+'_f64(').replace('self.validate_velocity(','self.validate_velocity_f64(')
        methods.append(method)
    operator_path.write_text(operator+"\n#[allow(clippy::useless_conversion)]\nimpl<'a> PressureOperator<'a>{\n"+'\n'.join(methods)+'\n}\n')
    here=Path(__file__).parent;precision_dir=copy/'src/precision_diagnostic';precision_dir.mkdir()
    for name in ['sampling','advection','viscosity']:
        data=specialize((copy/'src'/f'{name}.rs').read_text())
        if name=='advection':
            old='use crate::{Axis, GridGeometry, SamplingError, ScalarSampler, VelocitySampler};'
            if data.count(old)!=1:raise ValueError('Advection imports changed')
            data=data.replace(old,'use crate::{Axis, GridGeometry};\nuse super::{SamplingError, ScalarSampler, VelocitySampler};')
        (precision_dir/f'{name}64.rs').write_text('#![allow(clippy::useless_conversion)]\n'+data)
    core=[]
    for name in ['native_solve','prepare','finish','force']:
        data=specialize(function(engine,'fn '+name+'('))
        for method in ['build_rhs','correct_velocity','divergence']:
            data=data.replace('.'+method+'(','.'+method+'_f64(')
        core.append(data)
    (copy/'src/precision_diagnostic.rs').write_text((here/'precision.rs').read_text()+'\n'+'\n'.join(core)+'\n')
    lib=copy/'src/lib.rs';lib.write_text(lib.read_text()+'\n#[cfg(feature="diagnostic")]\npub mod precision_diagnostic;\n')
    crate=run/'probe';(crate/'src').mkdir(parents=True)
    (crate/'Cargo.toml').write_text('''[package]
name = "rheon-storage-precision-probe"
version = "0.1.0"
edition = "2024"
rust-version = "1.92"
publish = false
[dependencies]
anyhow = "=1.0.100"
serde_json = { version = "=1.0.154", features = ["float_roundtrip"] }
rheon = { path = "../source", default-features = false, features = ["diagnostic"] }
[lints.rust]
unsafe_code = "forbid"
''')
    shutil.copyfile(here/'main.rs',crate/'src/main.rs')
    material=read(root/config['retained_scene']);scenes=[]
    for row in config['scenes']:
        scene={**material['config']['scene'],**row['overrides']}
        path=run/(row['id']+'-scene.json');path.write_text(json.dumps(scene,indent=2)+'\n')
        scenes.append(dict(id=row['id'],label=row['label'],scene=scene,path=str(path.relative_to(root))))
    fixture={**material['config']['scene'],'steps':16}
    (run/'fixture-scene.json').write_text(json.dumps(fixture,indent=2)+'\n')
    page=root/'docs/plans/cohort-progress.html';shutil.copyfile(page,run/'selection-chart.html')
    receipt=dict(schema=config['schema'],root=str(root),config=config,config_sha256=sha(config_path),scenes=scenes,
        original_build_sha256=sha(root/config['baseline_build']),original_sources=original,
        transformations='Original f32 functions unchanged; generate f64 storage specializations with the same pressure solver/settings. Added research observation methods and an initialization-only streaming horizon up to 3600 steps; original geometry/physics validation retained, no full-frame history is accumulated.',
        generated_sources={str(p.relative_to(copy)):sha(p)for p in copy.rglob('*')if p.is_file()},
        experiment_sources={p.name:sha(p)for p in here.iterdir()if p.is_file()},
        selection_chart_sha256=sha(page))
    (run/'material.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def build(root,config_path,run):
    receipt=prepare(root,config_path,run);selection=read(root/receipt['config']['dependency_config'])
    config_base=(root/receipt['config']['dependency_config']).parent
    def resolve(value):
        path=Path(value);return path if path.is_absolute()else(config_base/path).resolve()
    cargo=resolve(selection['paths']['cargo']);env={k:v for k,v in os.environ.items()if not k.startswith(('CARGO_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(CARGO_HOME=str(resolve(selection['paths']['cargo_home'])),CARGO_TARGET_DIR=str(resolve(selection['paths']['cargo_target'])),
        CARGO_BUILD_JOBS='2',RUSTUP_TOOLCHAIN='1.92.0',RUSTC=str(cargo.parent/'rustc'),RUSTDOC=str(cargo.parent/'rustdoc'),
        RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='',RAYON_NUM_THREADS='1')
    commands=[]
    for operation in ['build','clippy']:
        command=[str(cargo),operation,'--manifest-path',str(run/'probe/Cargo.toml'),'--release','--offline']
        if operation=='clippy':command+=['--','-D','warnings']
        start=time.perf_counter()
        with(run/(operation+'.log')).open('wb')as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        commands.append(dict(command=command,returncode=result.returncode,elapsed_s=time.perf_counter()-start))
        if result.returncode:raise RuntimeError('Precision probe '+operation+' failed; inspect '+str(run/(operation+'.log')))
    binary=run/'rheon-storage-precision-probe';shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/rheon-storage-precision-probe',binary);binary.chmod(0o555)
    for path,expected in receipt['generated_sources'].items():
        if sha(run/'source'/path)!=expected:raise ValueError('Research source changed during build: '+path)
    result=dict(status='built',commands=commands,binary=str(binary.relative_to(root)),sha256=sha(binary),
        environment={k:env[k]for k in ['CARGO_HOME','CARGO_TARGET_DIR','CARGO_BUILD_JOBS','RUSTUP_TOOLCHAIN','RUSTC','RUSTDOC','RAYON_NUM_THREADS']},
        material_sha256=sha(run/'material.json'),probe_lock_sha256=sha(run/'probe/Cargo.lock'))
    (run/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(Path(__file__).resolve().parents[2],args.config.resolve(),args.output.resolve())
