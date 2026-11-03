//! Offline tracer/displacement teacher. No pressure solve or full timestep.
use anyhow::{Context, Result, ensure};
use rheon::{Axis, GridGeometry, VelocitySampler, advect_tracer};
use serde_json::{Value, json};
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut text = String::new();
    io::stdin().take(64 * 1024 * 1024 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= 64 * 1024 * 1024, "bounded tracer teacher input");
    let packet: Value = serde_json::from_str(&text)?;
    let cases = packet["cases"].as_array().context("cases required")?;
    ensure!(!cases.is_empty() && cases.len() <= 4, "bounded case count");
    let mut outputs = Vec::new();
    for case in cases {
        let mut counts = [0_u64; 3];
        for (d, n) in counts.iter_mut().enumerate() { *n = case["counts"][d].as_u64().context("counts required")?; }
        let h = case["cell_size_m"].as_f64().context("spacing required")?;
        let dt = case["dt_s"].as_f64().context("dt required")?;
        let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
        ensure!(grid.cell_len() <= 64_usize.pow(3), "bounded cells");
        let arrays = case["velocity"].as_array().context("velocity required")?;
        ensure!(arrays.len() == 3, "three face arrays required");
        let axes = [Axis::X, Axis::Y, Axis::Z];
        let mut velocity: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for d in 0..3 {
            velocity[d] = arrays[d].as_array().context("faces required")?.iter()
                .map(|v| v.as_f64().context("finite face required").map(|x| x as f32)).collect::<Result<Vec<_>>>()?;
            ensure!(velocity[d].len() == grid.face_len(axes[d]), "staggered layout required");
        }
        let refs = [&velocity[0][..], &velocity[1][..], &velocity[2][..]];
        let sampler = VelocitySampler::new(&grid, refs)?;
        let dimensions = grid.counts();
        let mut offsets: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.; grid.cell_len()]);
        let mut center_velocity = offsets.clone();
        for row in 0..grid.cell_len() {
            let p = [row % dimensions[0], (row / dimensions[0]) % dimensions[1], row / (dimensions[0] * dimensions[1])];
            let position = grid.cell_position(p).context("cell position required")?;
            let departure = sampler.backtrace(position, dt)?;
            let value = sampler.sample(position)?;
            for d in 0..3 { offsets[d][row] = (departure[d] - position[d]) / h; center_velocity[d][row] = value[d]; }
        }
        let tracers = case["tracers"].as_array().context("tracers required")?;
        ensure!(!tracers.is_empty() && tracers.len() <= 4, "bounded tracer count");
        let mut predicted = Vec::new();
        for values in tracers {
            let old = values.as_array().context("cell tracer required")?.iter()
                .map(|v| v.as_f64().context("finite tracer required").map(|x| x as f32)).collect::<Result<Vec<_>>>()?;
            let mut next = vec![0.; grid.cell_len()];
            advect_tracer(&grid, &old, refs, dt, &mut next, || false)?;
            predicted.push(next);
        }
        outputs.push(json!({"tracers":predicted,"offsets_cells":offsets,"center_velocity":center_velocity,
            "pressure_solves":0,"native_steps_executed":0,"tracer_primitive_calls":tracers.len()}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":outputs}))?);
    Ok(())
}
