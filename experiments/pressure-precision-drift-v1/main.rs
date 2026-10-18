//! Matched field-storage precision, canonical native qualification, and local casts.
use anyhow::{Result, ensure};
use rheon::{GridGeometry, PressureImplementation, PressureOperator, PressureSettings, PressureWorkspace};
use rheon::diagnostic::{DiagnosticSession, Scene, DT, RHO};
use rheon::precision_diagnostic::Track64;
use serde_json::json;
use std::{fs,io::{BufWriter,Write},sync::atomic::AtomicBool,time::Instant};

fn metric(a:impl Iterator<Item=f64>,b:impl Iterator<Item=f64>)->serde_json::Value{
    let mut squares=0.0;let mut reference=0.0;let mut maximum=0.0_f64;let mut count=0;
    for(a,b)in a.zip(b){let d=a-b;squares+=d*d;reference+=b*b;maximum=maximum.max(d.abs());count+=1;}
    json!({"rms":(squares/count as f64).sqrt(),"maximum":maximum,
        "relative_l2":if reference>0.0{Some((squares/reference).sqrt())}else if squares==0.0{Some(0.0)}else{None},
        "reference_rms":(reference/count as f64).sqrt()})
}
fn equal64(a:impl Iterator<Item=f64>,b:impl Iterator<Item=f64>)->bool{
    a.map(f64::to_bits).eq(b.map(f64::to_bits))
}
fn main()->Result<()>{
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(args.len()==2,"Pass scene JSON and a new NDJSON output path");
    let mut scene:Scene=serde_json::from_slice(&fs::read(&args[0])?)?;
    let horizon=scene.steps;
    ensure!((1..=3600).contains(&horizon),"Streaming horizon must be 1–3600 steps");
    // Admit the same bounded geometry/physics through the original constructor.
    // The raw DiagnosticSession and this probe retain current fields, not history.
    scene.steps=horizon.min(512);scene.validate().map_err(anyhow::Error::msg)?;
    let mut output=BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(&args[1])?);
    let start=Instant::now();let cancel=AtomicBool::new(false);
    let grid=GridGeometry::new(scene.counts.map(|n|n as u64),[scene.h();3],[0.0;3])?;
    let mut original=DiagnosticSession::new(scene.clone()).map_err(anyhow::Error::msg)?;
    let mut rounded=DiagnosticSession::new(scene.clone()).map_err(anyhow::Error::msg)?;
    let initial=original.initial();
    let mut precise=Track64::new(scene.clone(),&initial.reference_tracer).map_err(anyhow::Error::msg)?;
    let mut twin=Track64::new(scene.clone(),&initial.reference_tracer).map_err(anyhow::Error::msg)?;
    original.precision_set_horizon(horizon).map_err(anyhow::Error::msg)?;
    rounded.precision_set_horizon(horizon).map_err(anyhow::Error::msg)?;
    precise.set_horizon(horizon).map_err(anyhow::Error::msg)?;
    twin.set_horizon(horizon).map_err(anyhow::Error::msg)?;
    let method=if scene.sgs{PressureImplementation::SymmetricGaussSeidelPcgV1}else{PressureImplementation::JacobiPcgV1};
    let mut legacy_provider=PressureWorkspace::with_implementation(&grid,128*1024*1024,method)?;
    let operator=PressureOperator::new(&grid,RHO)?;
    let mut verified_casts=0_usize;
    for step in 1..=horizon {
        let frame=original.advance(&cancel,|request|{
            let mut rhs=vec![0.0;grid.cell_len()];operator.apply_full(request.initial_guess,&mut rhs).map_err(|e|e.to_string())?;
            for(a,b)in rhs.iter_mut().zip(request.residual){*a+=b;}
            legacy_provider.solve_rhs(&operator,&rhs,DT,PressureSettings{relative_residual:1e-8,absolute_residual:1e-12,
                divergence_limit:1e-7,max_iterations:3000},||false).map_err(|e|e.to_string())?;
            Ok(legacy_provider.pressure().to_vec())
        }).map_err(anyhow::Error::msg)?;
        let f32=rounded.precision_advance_reference(&cancel).map_err(anyhow::Error::msg)?;
        ensure!(equal64(f32.pressure.iter().copied(),frame.reference_pressure.iter().copied()),"Canonical pressure mismatch at step {step}");
        for d in 0..3 {
            ensure!(f32.velocity[d].iter().map(|v|v.to_bits()).eq(original.precision_reference_velocity()[d].iter().map(|v|v.to_bits())),"Canonical velocity mismatch at step {step}");
        }
        ensure!(f32.tracer.iter().map(|v|v.to_bits()).eq(frame.reference_tracer.iter().map(|v|v.to_bits())),"Canonical tracer mismatch at step {step}");
        let quality64=precise.advance(&cancel).map_err(anyhow::Error::msg)?;
        twin.advance(&cancel).map_err(anyhow::Error::msg)?;
        ensure!(equal64(precise.state.pressure.iter().copied(),twin.state.pressure.iter().copied())
            &&equal64(precise.state.velocity.iter().flatten().copied(),twin.state.velocity.iter().flatten().copied())
            &&equal64(precise.state.tracer.iter().copied(),twin.state.tracer.iter().copied()),"Float64 repeat control differs at step {step}");
        for c in [&f32.casts.advection,&f32.casts.tracer_advection,&f32.casts.force,&f32.casts.viscosity,&f32.casts.projection]{
            verified_casts+=c.values;ensure!(c.cast_mismatches==0,"Local cast did not reproduce native storage");
        }
        let row=json!({"kind":"frame","step":step,"time_s":step as f64*DT,
            "pressure":metric(f32.pressure.iter().copied(),precise.state.pressure.iter().copied()),
            "velocity":metric(f32.velocity.iter().flatten().copied().map(f64::from),precise.state.velocity.iter().flatten().copied()),
            "tracer":metric(f32.tracer.iter().copied().map(f64::from),precise.state.tracer.iter().copied()),
            "casts":f32.casts,"f32_quality":f32.quality,"f64_quality":quality64,
            "canonical_f32_exact":true,"f64_repeat_exact":true,"f64_repeat_drift":0.0});
        serde_json::to_writer(&mut output,&row)?;output.write_all(b"\n")?;
        if step%16==0{output.flush()?;}
    }
    serde_json::to_writer(&mut output,&json!({"kind":"complete","steps":horizon,"elapsed_s":start.elapsed().as_secs_f64(),
        "canonical_f32_exact_steps":horizon,"f64_repeat_exact_steps":horizon,"verified_cast_values":verified_casts,
        "neural_calls":0,"same_pressure_settings":true,"pressure_dtype":"f64","field_storage":["f32","f64"],
        "max_rss_kib":std::fs::read_to_string("/proc/self/status")?.lines().find(|l|l.starts_with("VmHWM:")).unwrap_or("")}))?;
    output.write_all(b"\n")?;output.flush()?;Ok(())
}
