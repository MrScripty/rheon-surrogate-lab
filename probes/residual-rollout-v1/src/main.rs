//! CPU experiment wire around Rheon's unchanged paired water engine.
include!(concat!(env!("OUT_DIR"), "/selected.rs"));
use anyhow::{Context, Result, ensure};
use rheon::{GridGeometry, PressureOperator, PressureSettings, PressureWorkspace};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, Read, Write};
use std::sync::atomic::AtomicBool;
const LIMIT: u64 = 8 * 1024 * 1024;
fn receive(input: &mut impl BufRead) -> Result<String> {
    let mut line = String::new();
    input.take(LIMIT + 1).read_line(&mut line)?;
    ensure!(
        !line.is_empty() && line.len() as u64 <= LIMIT && line.ends_with('\n'),
        "bounded complete input"
    );
    Ok(line)
}
fn send(value: &Value) -> Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    id: usize,
    pressure: Vec<f64>,
}
fn main() -> Result<()> {
    let control = std::env::args().any(|arg| arg == "--reference-control");
    let mut input = io::stdin().lock();
    let scene: scene::Scene = serde_json::from_str(&receive(&mut input)?)?;
    scene.validate().map_err(anyhow::Error::msg)?;
    let grid = GridGeometry::new(scene.counts.map(|n| n as u64), [scene.h(); 3], [0.0; 3])?;
    let operator = PressureOperator::new(&grid, scene::RHO)?;
    let mut solver = PressureWorkspace::new(&grid, 128 * 1024 * 1024)?;
    let mut engine = engine::Engine::new(scene.clone()).map_err(anyhow::Error::msg)?;
    let cancel = AtomicBool::new(false);
    let mut ai_steps = 0;
    let mut qualified = 0;
    let mut stopped = false;
    for _ in 0..scene.steps {
        let frame = engine.advance(&cancel, |request| {
            let result = (|| -> Result<Vec<f64>> {
                if control {
                    let mut rhs = vec![0.0;grid.cell_len()];
                    operator.apply_full(request.base, &mut rhs)?;
                    for (value, residual) in rhs.iter_mut().zip(request.residual) { *value += residual; }
                    solver.solve_rhs(&operator, &rhs, scene::DT, PressureSettings {
                        relative_residual:1e-8, absolute_residual:1e-12, divergence_limit:1e-7, max_iterations:3000,
                    }, || false)?;
                    return Ok(solver.pressure().to_vec());
                }
                send(&json!({"kind":"predict", "id":request.step, "counts_xyz":request.scene.counts,
                    "cell_size_m":request.scene.h(), "density_kg_m3":scene::RHO, "dt_s":scene::DT,
                    "base_pressure":request.base, "native_input_residual":request.residual}))?;
                let reply: Reply = serde_json::from_str(&receive(&mut input)?)?;
                ensure!(reply.id == request.step, "prediction identity");
                Ok(reply.pressure)
            })();
            result.map_err(|e| format!("prediction: {e:#}"))
        }).map_err(anyhow::Error::msg).context("paired native advance")?;
        let pressure_error = frame
            .ai_pressure
            .iter()
            .zip(&frame.reference_pressure)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let reference_norm = frame
            .reference_pressure
            .iter()
            .map(|p| p * p)
            .sum::<f64>()
            .sqrt();
        if let Some(quality) = &frame.ai_quality {
            ai_steps += 1;
            qualified += usize::from(quality.qualified);
            stopped = quality.hard_stop;
        }
        send(
            &json!({"kind":"frame", "phase":frame.phase,"step":frame.step,"time_s":frame.time_s,
            "reference_quality":frame.reference_quality,"ai_quality":frame.ai_quality,
            "relative_pressure_difference":pressure_error/reference_norm.max(1e-30),
            "reference_seconds":frame.reference_seconds,"ai_seconds":frame.ai_seconds}),
        )?;
        if stopped {
            break;
        }
    }
    send(
        &json!({"kind":"complete","ai_steps":ai_steps,"qualified_ai_steps":qualified,"hard_stop":stopped,
        "reference_control":control,"warmup_steps":scene.warmup_steps}),
    )
}
