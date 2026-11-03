//! Offline labels for cell/face displacements and native transport primitives.
use anyhow::{Context, Result, ensure};
use rheon::{Axis, GridGeometry, VelocitySampler, advect_tracer, advect_velocity};
use serde_json::{Value, json};
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut text = String::new();
    io::stdin().take(64 * 1024 * 1024 + 1).read_to_string(&mut text)?;
    ensure!(text.len() <= 64 * 1024 * 1024, "bounded offline transport input");
    let packet: Value = serde_json::from_str(&text)?;
    let cases = packet["cases"].as_array().context("cases required")?;
    ensure!(!cases.is_empty() && cases.len() <= 2, "bounded case count");
    let axes = [Axis::X, Axis::Y, Axis::Z];
    let mut outputs = Vec::new();
    for case in cases {
        let mut counts = [0_u64; 3];
        for (d, n) in counts.iter_mut().enumerate() { *n = case["counts"][d].as_u64().context("counts required")?; }
        let h = case["cell_size_m"].as_f64().context("spacing required")?;
        let dt = case["dt_s"].as_f64().context("dt required")?;
        let grid = GridGeometry::new(counts, [h; 3], [0.; 3])?;
        ensure!(grid.cell_len() <= 32_usize.pow(3), "registered pilot grid cap");
        let arrays = case["velocity"].as_array().context("velocity required")?;
        ensure!(arrays.len() == 3, "three staggered arrays required");
        let mut old: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for d in 0..3 {
            old[d] = arrays[d].as_array().context("faces required")?.iter()
                .map(|v| v.as_f64().context("finite face required").map(|x| x as f32)).collect::<Result<Vec<_>>>()?;
            ensure!(old[d].len() == grid.face_len(axes[d]), "staggered face layout required");
        }
        let refs = [&old[0][..], &old[1][..], &old[2][..]];
        let sampler = VelocitySampler::new(&grid, refs)?;
        let point_only = case["point_only"].as_bool().unwrap_or(false);
        let mut points = Vec::new();
        for kind in 0..4 {
            let dims = if kind < 3 { grid.face_counts(axes[kind]) } else { grid.counts() };
            let len = dims[0] * dims[1] * dims[2];
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.; len]);
            let mut shifts: [Vec<f64>; 3] = std::array::from_fn(|_| if point_only { Vec::new() } else { vec![0.; len] });
            for row in 0..len {
                let p = [row % dims[0], (row / dims[0]) % dims[1], row / (dims[0] * dims[1])];
                let position = if kind < 3 { grid.face_position(axes[kind], p) } else { grid.cell_position(p) }.context("point position required")?;
                let value = sampler.sample(position)?;
                for d in 0..3 { samples[d][row] = value[d]; }
                if !point_only {
                    let departure = sampler.backtrace(position, dt)?;
                    for d in 0..3 { shifts[d][row] = (departure[d] - position[d]) / h; }
                }
            }
            points.push(json!({"dimensions_xyz":dims,"velocity_samples":samples,"offsets_cells":shifts}));
        }
        let mut velocity = None;
        let mut tracer = None;
        if !point_only {
            let mut vx = vec![0.; old[0].len()];
            let mut vy = vec![0.; old[1].len()];
            let mut vz = vec![0.; old[2].len()];
            advect_velocity(&grid, refs, dt, [&mut vx, &mut vy, &mut vz], || false)?;
            velocity = Some(vec![vx,vy,vz]);
            let values = case["tracer"].as_array().context("tracer required")?.iter()
                .map(|v| v.as_f64().context("finite tracer required").map(|x| x as f32)).collect::<Result<Vec<_>>>()?;
            let mut next = vec![0.; grid.cell_len()];
            advect_tracer(&grid, &values, refs, dt, &mut next, || false)?;
            tracer = Some(next);
        }
        outputs.push(json!({"points":points,"advected_velocity":velocity,"advected_tracer":tracer,
            "point_only":point_only,"pressure_solves":0,"native_steps_executed":0}));
    }
    println!("{}", serde_json::to_string(&json!({"cases":outputs}))?);
    Ok(())
}
