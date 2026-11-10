//! Interactive research capture and explicit state restart, with original pressure gates.
use anyhow::{Result, ensure};
use rheon::diagnostic::Scene;
use rheon::diagnostic::engine::SingleTrackSession;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

fn send(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
fn line(input: &mut impl BufRead) -> Result<Value> {
    let mut raw=String::new();
    input.take(65537).read_line(&mut raw)?;
    ensure!(raw.len()<=65536 && raw.ends_with('\n'), "bounded complete command");
    Ok(serde_json::from_str(&raw)?)
}
fn f64_field(out: &mut impl Write, values: &[f64]) -> Result<()> {
    let mut block=Vec::with_capacity(65536);
    for chunk in values.chunks(8192) {
        block.clear();
        for v in chunk { block.extend_from_slice(&v.to_le_bytes()); }
        out.write_all(&block)?;
    }
    Ok(())
}
fn f32_field(out: &mut impl Write, values: &[f32]) -> Result<()> {
    let mut block=Vec::with_capacity(65536);
    for chunk in values.chunks(16384) {
        block.clear();
        for v in chunk { block.extend_from_slice(&v.to_le_bytes()); }
        out.write_all(&block)?;
    }
    Ok(())
}
fn read_f64(input: &mut impl Read, count: usize) -> Result<Vec<f64>> {
    ensure!(count<=256usize.pow(3),"bounded pressure input");
    let mut bytes=vec![0;count*8];input.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect())
}
fn read_f32(input: &mut impl Read, count: usize) -> Result<Vec<f32>> {
    ensure!(count<=257*256*256,"bounded staggered field input");
    let mut bytes=vec![0;count*4];input.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())
}
fn state(out: &mut impl Write, session: &SingleTrackSession) -> Result<()> {
    let (p,v,t)=session.state_fields();
    send(out,&json!({"kind":"state","step":session.state_step(),"time_s":session.state_step()as f64/(60.*rheon::diagnostic::scheduled_forcing::substeps()as f64),
        "pressure_cells":p.len(),"velocity_lengths":v.map(|a|a.len()),"tracer_cells":t.len(),
        "fields":["pressure_f64","vx_f32","vy_f32","vz_f32","tracer_f32"]}))?;
    f64_field(out,p)?;for axis in v { f32_field(out,axis)?; }f32_field(out,t)?;
    out.flush()?;
    Ok(())
}
fn main() -> Result<()> {
    let mut input=BufReader::with_capacity(1024*1024,io::stdin());
    let mut out=BufWriter::with_capacity(1024*1024,io::stdout());
    let mut raw_scene=line(&mut input)?;
    let object=raw_scene.as_object_mut().ok_or_else(||anyhow::anyhow!("scene object"))?;
    let controls=object.remove("forcing_schedule");
    object.remove("forcing_program");
    let control_factor=match object.remove("forcing_control_substeps") {
        Some(value)=>value.as_u64().ok_or_else(||anyhow::anyhow!("control cadence integer"))? as usize,
        None=>1,
    };
    let binary_count=match object.remove("forcing_binary_controls") {
        Some(value)=>Some(value.as_u64().ok_or_else(||anyhow::anyhow!("control count integer"))? as usize),
        None=>None,
    };
    let factor=match object.remove("reference_substeps") {
        Some(value)=>value.as_u64().ok_or_else(||anyhow::anyhow!("reference substeps integer"))? as usize,
        None=>1,
    };
    let mut scene:Scene=serde_json::from_value(raw_scene)?;
    let frames=if let Some(count)=binary_count {
        ensure!(controls.is_none(),"only one external-control encoding permitted");
        ensure!(count<=38400 && count>0,"bounded external-control count");
        let raw=read_f64(&mut input,count*4)?;
        Some(raw.chunks_exact(4).map(|v|rheon::diagnostic::scheduled_forcing::ForceFrame {
            center:[v[0],v[1],v[2]],pulse:v[3],
        }).collect())
    } else {match controls {Some(value)=>Some(serde_json::from_value(value)?),None=>None}};
    rheon::diagnostic::scheduled_forcing::configure(frames,scene.steps,factor,control_factor)
        .map_err(anyhow::Error::msg)?;
    scene.steps=scene.steps.checked_mul(factor).ok_or_else(||anyhow::anyhow!("fine-step overflow"))?;
    ensure!(scene.steps<=38400,"bounded temporal fine-step count");
    let tick=std::time::Instant::now();
    let mut session=SingleTrackSession::new(scene.clone()).map_err(anyhow::Error::msg)?;
    send(&mut out,&json!({"kind":"ready","setup_seconds":tick.elapsed().as_secs_f64()}))?;
    loop {
        let command=line(&mut input)?;
        match command["kind"].as_str() {
            Some("inspect") => state(&mut out,&session)?,
            Some("restore") => {
                let step=command["step"].as_u64().ok_or_else(||anyhow::anyhow!("restart step"))? as usize;
                let cells=scene.counts.iter().product();
                let pressure=read_f64(&mut input,cells)?;
                let mut velocity=[Vec::new(),Vec::new(),Vec::new()];
                for (axis,value)in velocity.iter_mut().enumerate() {
                    let size=(scene.counts[axis]+1)*scene.counts.iter().enumerate().filter(|(a,_)|*a!=axis).map(|(_,n)|n).product::<usize>();
                    *value=read_f32(&mut input,size)?;
                }
                let tracer=read_f32(&mut input,cells)?;
                session.restore_state_fields(step,pressure,velocity,tracer).map_err(anyhow::Error::msg)?;
                send(&mut out,&json!({"kind":"restored","step":session.state_step(),"numerical_steps_executed":0}))?;
            }
            Some("forcing") => {
                let step=command["step"].as_u64().ok_or_else(||anyhow::anyhow!("force step"))? as usize;
                let mut velocity=[Vec::new(),Vec::new(),Vec::new()];
                for (axis,value)in velocity.iter_mut().enumerate() {
                    let size=(scene.counts[axis]+1)*scene.counts.iter().enumerate()
                        .filter(|(a,_)|*a!=axis).map(|(_,n)|n).product::<usize>();
                    *value=read_f32(&mut input,size)?;
                }
                let values=session.preview_prescribed_force(step,velocity).map_err(anyhow::Error::msg)?;
                send(&mut out,&json!({"kind":"forcing","step":step,"state_unchanged":true}))?;
                for value in values { f32_field(&mut out,&value)?; }
                out.flush()?;
            }
            Some("advance") => {
                let native=command["native"].as_bool().unwrap_or(false);
                let result=session.advance(native,1e-8,|base,residual| {
                    let exchange=(|| -> Result<Vec<f64>> {
                        send(&mut out,&json!({"kind":"predict","cells":base.len(),"counts_xyz":scene.counts,
                            "cell_size_m":scene.h(),"rho":1000.,"dt":rheon::diagnostic::scheduled_forcing::dt()}))?;
                        f64_field(&mut out,base)?;f64_field(&mut out,residual)?;out.flush()?;
                        read_f64(&mut input,base.len())
                    })();
                    exchange.map_err(|e|e.to_string())
                }).map_err(anyhow::Error::msg)?;
                send(&mut out,&json!({"kind":"frame","frame":result,
                    "preparation_profile":rheon::diagnostic::engine::last_preparation_profile()}))?;
            }
            Some("stop") => { send(&mut out,&json!({"kind":"complete"}))?;break; }
            _ => anyhow::bail!("Unknown temporal command"),
        }
    }
    Ok(())
}
