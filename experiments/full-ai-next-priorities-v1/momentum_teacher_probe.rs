//! Offline native momentum readout teacher; no timestep or pressure solve.
use anyhow::{Context, Result, ensure};
use rheon::{Axis, GridGeometry, PressureOperator};
use serde_json::{Value, json};
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut text = String::new();
    io::stdin().take(64 * 1024 * 1024 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= 64 * 1024 * 1024, "bounded offline teacher input");
    let packet: Value = serde_json::from_str(&text)?;
    let cases = packet["cases"].as_array().context("cases required")?;
    ensure!(!cases.is_empty() && cases.len() <= 16, "bounded case count");
    let mut result = Vec::new();
    for case in cases {
        let mut counts = [0_u64; 3];
        for (d, n) in counts.iter_mut().enumerate() {
            *n = case["counts"][d].as_u64().context("counts required")?;
        }
        let h = case["cell_size_m"].as_f64().context("spacing required")?;
        let dt = case["dt_s"].as_f64().context("timestep required")?;
        let rho = case["rho"].as_f64().context("density required")?;
        let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
        ensure!(grid.cell_len() <= 64_usize.pow(3), "bounded cells");
        let pressure = case["pressure"].as_array().context("pressure required")?
            .iter().map(|v| v.as_f64().context("finite pressure required"))
            .collect::<Result<Vec<_>>>()?;
        let axes = [Axis::X, Axis::Y, Axis::Z];
        let old: [Vec<f32>; 3] = std::array::from_fn(|d| vec![0.; grid.face_len(axes[d])]);
        let mut vx = old[0].clone();
        let mut vy = old[1].clone();
        let mut vz = old[2].clone();
        PressureOperator::new(&grid, rho)?.correct_velocity(
            [&old[0], &old[1], &old[2]], &pressure, dt, [&mut vx, &mut vy, &mut vz])?;
        result.push(json!({"velocity":[vx,vy,vz],"counts":counts,"pressure_solves":0,"native_steps_executed":0}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":result}))?);
    Ok(())
}
