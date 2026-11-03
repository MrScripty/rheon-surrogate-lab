//! Offline native Newtonian-viscosity primitive; no pressure or full step.
use anyhow::{Context, Result, ensure};
use rheon::{Axis, GridGeometry, ViscosityWorkspace};
use serde_json::{Value, json};
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut raw = String::new();
    io::stdin().take(64 * 1024 * 1024 + 1).read_to_string(&mut raw)?;
    ensure!(raw.len() <= 64 * 1024 * 1024, "bounded viscosity input");
    let packet: Value = serde_json::from_str(&raw)?;
    let cases = packet["cases"].as_array().context("cases required")?;
    ensure!(!cases.is_empty() && cases.len() <= 2, "bounded primitive cases");
    let mut outputs = Vec::new();
    for case in cases {
        let mut counts = [0_u64; 3];
        for (d, n) in counts.iter_mut().enumerate() { *n = case["counts"][d].as_u64().context("counts required")?; }
        let h = case["cell_size_m"].as_f64().context("spacing required")?;
        let dt = case["dt_s"].as_f64().context("timestep required")?;
        let rho = case["density"].as_f64().context("density required")?;
        let mu = case["dynamic_viscosity"].as_f64().context("dynamic viscosity required")?;
        let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
        ensure!(grid.cell_len() <= 32_usize.pow(3), "registered grid cap");
        let mut old: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for d in 0..3 {
            old[d] = case["velocity"][d].as_array().context("face field required")?.iter()
                .map(|v| v.as_f64().context("finite face required").map(|x| x as f32)).collect::<Result<Vec<_>>>()?;
            ensure!(old[d].len() == grid.face_len(Axis::ALL[d]), "native staggered layout required");
        }
        let mut vx = vec![0.; old[0].len()];
        let mut vy = vec![0.; old[1].len()];
        let mut vz = vec![0.; old[2].len()];
        let mut workspace = ViscosityWorkspace::new(grid, 16 * 1024 * 1024)?;
        let report = workspace.update([&old[0], &old[1], &old[2]], rho, mu, dt,
            [&mut vx, &mut vy, &mut vz], |_| false)?;
        outputs.push(json!({"updated_velocity":[vx,vy,vz],"beta":mu/rho*dt/(h*h),
            "kinetic_before":report.kinetic_before,"kinetic_after":report.kinetic_after,
            "energy_identity_error":report.energy_identity_error,"energy_rounding_budget":report.energy_rounding_budget,
            "native_steps_executed":0,"pressure_solves":0}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":outputs}))?);
    Ok(())
}
