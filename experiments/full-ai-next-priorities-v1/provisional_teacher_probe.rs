//! Offline pressure-stage reconstruction checks. No pressure solve or timestep.
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
    let axes = [Axis::X, Axis::Y, Axis::Z];
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
        let velocities = case["velocity"].as_array().context("velocity required")?;
        ensure!(velocities.len() == 3, "three staggered velocities required");
        let mut old: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for d in 0..3 {
            old[d] = velocities[d].as_array().context("face array required")?.iter()
                .map(|v| v.as_f64().context("finite velocity required").map(|x| x as f32))
                .collect::<Result<Vec<_>>>()?;
            ensure!(old[d].len() == grid.face_len(axes[d]), "face layout required");
            ensure!(old[d].iter().all(|v| v.is_finite()), "finite float32 faces required");
        }
        let mut vx = vec![0.; old[0].len()];
        let mut vy = vec![0.; old[1].len()];
        let mut vz = vec![0.; old[2].len()];
        let operator = PressureOperator::new(&grid, rho)?;
        operator.correct_velocity([&old[0], &old[1], &old[2]], &pressure, dt,
            [&mut vx, &mut vy, &mut vz])?;
        let mut rhs = vec![0.; grid.cell_len()];
        operator.build_rhs([&old[0], &old[1], &old[2]], dt, &mut rhs)?;
        result.push(json!({"corrected_velocity":[vx,vy,vz], "rhs":rhs,
            "pressure_solves":0, "native_steps_executed":0}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":result}))?);
    Ok(())
}
