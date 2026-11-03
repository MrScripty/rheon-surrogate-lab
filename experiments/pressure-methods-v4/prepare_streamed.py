"""Qualify isolated full-state streaming, without allocating paired frame history."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from v4_support import ROOT,read,write,sha

def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Native source seam changed: '+before[:100])
    return text.replace(before,after)

def build(output):
    if output.exists():raise ValueError('Fresh streamed native build required')
    prior=ROOT/'runs/pressure-methods-v3/aligned-build-01';receipt=read(prior/'build-receipt.json')
    if receipt['status']!='qualified_build'or sha(prior/'rheon-large-single-track')!=receipt['binary_sha256']:
        raise ValueError('Qualified predecessor required')
    for name,expected in receipt['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Native predecessor changed')
    output.mkdir(parents=True)
    for directory in ['rheon','probe']:shutil.copytree(prior/directory,output/directory,ignore=shutil.ignore_patterns('target'))
    engine=output/'rheon/src/diagnostic/engine.rs';text=engine.read_text()
    text=replace(text,'        let session = DiagnosticSession::new(scene)?;\n        Ok(Self { scene: session.scene, grid: session.grid, track: session.ai,',
        '''        if !(1..=3600).contains(&scene.steps) || scene.warmup_steps >= scene.steps {
            return Err("Choose 1–3600 streamed steps and a shorter warmup".into());
        }
        // Validate and allocate only the initializer's actual one-frame needs.
        // The original scene still owns all forcing, step and warmup semantics.
        let setup = Scene { steps: 1, warmup_steps: 0, ..scene.clone() };
        let session = DiagnosticSession::new(setup)?;
        Ok(Self { scene, grid: session.grid, track: session.ai,''')
    seam='    pub fn advance(\n        &mut self,\n        native: bool,'
    text=replace(text,seam,'''    pub fn state_fields(&self) -> (&[f64], [&[f32]; 3], &[f32]) {
        (&self.track.pressure, refs(&self.track.velocity), &self.track.tracer)
    }
    pub fn advance(
        &mut self,
        native: bool,''')
    text+='''
#[cfg(test)]
mod streamed_state_tests {
    use super::*;
    #[test]
    fn streaming_does_not_require_historical_frame_storage() {
        let scene = Scene { counts: [32; 3], steps: 2000, warmup_steps: 0, ..Default::default() };
        assert!(scene.validate().is_err()); // Paired history would exceed its guard.
        let single = SingleTrackSession::new(scene).unwrap();
        let (pressure, velocity, tracer) = single.state_fields();
        assert_eq!(pressure.len(), 32usize.pow(3));
        assert_eq!(tracer.len(), pressure.len());
        assert!(velocity.iter().all(|v| v.len() == 33*32*32));
        assert_eq!(single.scene.steps, 2000);
    }
    #[test]
    fn streaming_preserves_geometry_and_step_bounds() {
        for scene in [
            Scene { steps: 0, warmup_steps: 0, ..Default::default() },
            Scene { steps: 3601, ..Default::default() },
            Scene { counts: [257; 3], ..Default::default() },
            Scene { steps: 2, warmup_steps: 2, ..Default::default() },
        ] { assert!(SingleTrackSession::new(scene).is_err()); }
    }
}
'''
    engine.write_text(text)
    driver=output/'probe/src/main.rs';text=driver.read_text()
    text=replace(text,'    ensure!(scene.steps <= 32, "bounded 32-step streamed capture");','    ensure!(scene.steps <= 600, "bounded 600-step streamed capture");')
    text=replace(text,'    let native = std::env::args().any(|arg| arg == "--native");','    let native = std::env::args().any(|arg| arg == "--native");\n    let states = std::env::args().any(|arg| arg == "--state-fields");')
    before='        if !advanced { break; }'
    after='''        if states {
            let (p, v, t) = session.state_fields();
            send(&mut out, &json!({"kind":"state", "pressure_cells":p.len(),
                "velocity_lengths":v.map(|axis| axis.len()), "tracer_cells":t.len(),
                "field_order":["pressure_f64","vx_f32","vy_f32","vz_f32","tracer_f32"]}))?;
            field(&mut out, p)?;
            for axis in v { field_f32(&mut out, axis)?; }
            field_f32(&mut out, t)?;
            out.flush()?;
        }
        if !advanced { break; }'''
    text=replace(text,before,after)
    text=replace(text,'fn pressure(input: &mut impl Read, count: usize) -> Result<Vec<f64>> {','''fn field_f32(out: &mut impl Write, values: &[f32]) -> Result<()> {
    let mut block = Vec::with_capacity(65536);
    for chunk in values.chunks(16384) {
        block.clear();
        for value in chunk { block.extend_from_slice(&value.to_le_bytes()); }
        out.write_all(&block)?;
    }
    Ok(())
}
fn pressure(input: &mut impl Read, count: usize) -> Result<Vec<f64>> {''')
    text+='''
#[cfg(test)]
mod state_wire_tests {
    use super::*;
    #[test]
    fn full_state_f32_wire_preserves_all_bits_across_block_boundary() {
        let mut values: Vec<f32> = (0..18001).map(|i| (i as f32 - 9000.)/17.).collect();
        values.extend([f32::MAX, f32::MIN, f32::from_bits(1), -0.0]);
        let mut wire = Vec::new(); field_f32(&mut wire, &values).unwrap();
        let bits: Vec<u32> = wire.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
        assert_eq!(bits, values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}
'''
    driver.write_text(text)
    (output/'target').symlink_to(prior/'target',target_is_directory=True)
    env={k:v for k,v in os.environ.items()if not k.startswith(('CARGO_','RHEON_','RUSTFLAGS','RUSTC','RUSTDOC','RUSTUP_TOOLCHAIN'))}
    env.update(read(ROOT/'runs/dependency-migration-v1/probe-build-current-03/build-receipt.json')['environment'])
    env.update(CARGO_TARGET_DIR=str(output/'target'),CARGO_BUILD_JOBS='2',RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='')
    cargo=str(Path(env['RUSTC']).with_name('cargo'))
    common=['--manifest-path',str(output/'probe/Cargo.toml'),'--config','patch."https://github.com/MrScripty/Rheon.git".rheon.path='+json.dumps(str(output/'rheon')),'--offline','--locked','--release']
    commands=[('build',[cargo,'build',*common]),('wire-tests',[cargo,'test',*common]),
        ('native-tests',[cargo,'test','--manifest-path',str(output/'rheon/Cargo.toml'),'--offline','--locked','--release','--no-default-features','--features','diagnostic','--lib']),
        ('clippy',[cargo,'clippy',*common,'--','-D','warnings'])]
    sources={str(p.relative_to(ROOT)):sha(p)for p in [engine,driver,output/'rheon/src/diagnostic/scene.rs',output/'rheon/src/sampling.rs',Path(__file__)]}
    result=dict(status='building',predecessor_receipt_sha256=sha(prior/'build-receipt.json'),sources=sources,commands=[],
        physical_gates_unchanged=True,original_selected_rheon_modified=False,streamed_history=True,max_streamed_steps=600)
    write(output/'build-receipt.json',result)
    for name,command in commands:
        tick=time.perf_counter()
        with (output/(name+'.log')).open('xb')as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        result['commands'].append(dict(name=name,exit_code=0,seconds=time.perf_counter()-tick,log_sha256=sha(output/(name+'.log'))))
    binary=output/'rheon-large-single-track';shutil.copyfile(output/'target/release/rheon-residual-rollout',binary);binary.chmod(0o555)
    result.update(status='qualified_build',binary_sha256=sha(binary));write(output/'build-receipt.json',result)
    print('Qualified streamed full-state native build',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output.resolve())
