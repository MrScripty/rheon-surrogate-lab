//! Persistent, synchronous process boundary around Rheon's actual 3D kernels.
//! Both paths share every stage except pressure; no teacher states feed a rollout.
use anyhow::{Context, Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace, advect_tracer, advect_velocity,
};
use std::io::{Read, Write};
use std::time::Instant;

type Velocity = [Vec<f32>; 3];
const DT: f64 = 1.0 / 60.0;
const DENSITY: f64 = 1.25;
const ACTUAL_LIMIT: f64 = 1e-5;
fn refs(v: &Velocity) -> [&[f32]; 3] {
    [&v[0], &v[1], &v[2]]
}
fn muts(v: &mut Velocity) -> [&mut [f32]; 3] {
    let [x, y, z] = v;
    [x, y, z]
}
fn send_json(out: &mut impl Write, value: serde_json::Value) -> Result<()> {
    let bytes = serde_json::to_vec(&value)?;
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(&bytes)?;
    Ok(())
}
fn send_f64(out: &mut impl Write, values: &[f64]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.write_all(&bytes)?;
    Ok(())
}
fn send_f32(out: &mut impl Write, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.write_all(&bytes)?;
    Ok(())
}
fn read_f64(input: &mut impl Read, values: &mut [f64]) -> Result<()> {
    let mut bytes = vec![0; values.len() * 8];
    input.read_exact(&mut bytes)?;
    for (v, b) in values.iter_mut().zip(bytes.chunks_exact(8)) {
        *v = f64::from_le_bytes(b.try_into()?);
    }
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite pressure");
    Ok(())
}

// Scene controls vary by whole trajectory seed. Smooth source has all three
// velocity components; the passive concentration is only a visible flow marker.
fn source(grid: &GridGeometry, v: &mut Velocity, tracer: &mut [f32], seed: u64, time: f64) {
    let phase = (seed % 997) as f64 * 0.137;
    let center = [
        0.43 + 0.13 * (phase + time).sin(),
        0.28 + 0.06 * phase.cos(),
        0.45 + 0.13 * (phase - time).cos(),
    ];
    for (d, values) in v.iter_mut().enumerate() {
        let shape = grid.face_counts([Axis::X, Axis::Y, Axis::Z][d]);
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let p = [i, j, k];
                    if p[d] == 0 || p[d] + 1 == shape[d] {
                        continue;
                    }
                    let pos = grid
                        .face_position([Axis::X, Axis::Y, Axis::Z][d], p)
                        .unwrap();
                    let r2: f64 = (0..3).map(|a| (pos[a] - center[a]).powi(2)).sum();
                    let direction = [
                        0.75 * (phase + 2.0 * time).cos(),
                        1.0,
                        0.75 * (phase + 2.0 * time).sin(),
                    ][d];
                    let index = grid.face_index([Axis::X, Axis::Y, Axis::Z][d], p).unwrap();
                    // Small deterministic broadband stirring excites every pressure
                    // mode. It is part of the declared scene on every method/split.
                    let mut bits = (index as u64)
                        .wrapping_add(seed.wrapping_mul(7919))
                        .wrapping_add((time / DT).round() as u64 * 104729)
                        .wrapping_add(d as u64 * 997);
                    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                    bits ^= bits >> 31;
                    let noise = 2.0 * ((bits >> 40) as f64 / 16_777_216.0) - 1.0;
                    let acceleration = 1.0 * (-r2 / 0.012).exp() * direction + 0.05 * noise;
                    values[index] = (f64::from(values[index]) + DT * acceleration) as f32;
                }
            }
        }
    }
    let [nx, ny, nz] = grid.counts();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let pos = grid.cell_position([i, j, k]).unwrap();
                let r2: f64 = (0..3).map(|a| (pos[a] - center[a]).powi(2)).sum();
                let index = i + nx * (j + ny * k);
                tracer[index] =
                    (f64::from(tracer[index]) + DT * 8.0 * (-r2 / 0.006).exp()).min(1.0) as f32;
            }
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: rheon-trajectory-probe N jacobi|sgs|learned SEED"
    );
    let n: usize = args[1].parse()?;
    ensure!((8..=64).contains(&n), "grid must be 8..64");
    let method = args[2].as_str();
    ensure!(
        ["jacobi", "sgs", "learned"].contains(&method),
        "unknown method"
    );
    let seed: u64 = args[3].parse()?;
    let grid = GridGeometry::new([n as u64; 3], [1.0 / n as f64; 3], [0.0; 3])?;
    let operator = PressureOperator::new(&grid, DENSITY)?;
    let implementation = if method == "sgs" {
        PressureImplementation::SymmetricGaussSeidelPcgV1
    } else {
        PressureImplementation::JacobiPcgV1
    };
    let mut workspace =
        PressureWorkspace::with_implementation(&grid, 128 * 1024 * 1024, implementation)?;
    let settings = PressureSettings {
        relative_residual: 1e-8,
        absolute_residual: 1e-12,
        divergence_limit: 1e-7,
        max_iterations: 3000,
    };
    let mut old: Velocity =
        std::array::from_fn(|d| vec![0.0; grid.face_len([Axis::X, Axis::Y, Axis::Z][d])]);
    let mut candidate = old.clone();
    let mut corrected = old.clone();
    let mut tracer = vec![0.0; grid.cell_len()];
    let mut next_tracer = tracer.clone();
    let mut rhs = vec![0.0; grid.cell_len()];
    let mut pressure = rhs.clone();
    let mut product = rhs.clone();
    let mut divergence = rhs.clone();
    let mut input = std::io::stdin().lock();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    send_json(
        &mut output,
        serde_json::json!({"schema":"rheon-trajectory-v1","n":n,
        "dt":DT,"density":DENSITY,"spacing":grid.spacing(),"method":method,
        "seed":seed,"actual_divergence_limit":ACTUAL_LIMIT,
        "native_relative_residual":settings.relative_residual,
        "face_lengths":old.iter().map(Vec::len).collect::<Vec<_>>() }),
    )?;
    output.flush()?;
    let mut frame = 0;
    loop {
        let mut command = [0u8; 1];
        if input.read_exact(&mut command).is_err() {
            break;
        }
        if command[0] == 0 {
            break;
        }
        ensure!([1, 2].contains(&command[0]), "unknown frame command");
        let capture = command[0] == 2;
        let start = Instant::now();
        let stage = Instant::now();
        advect_velocity(&grid, refs(&old), DT, muts(&mut candidate), || false)?;
        source(&grid, &mut candidate, &mut tracer, seed, frame as f64 * DT);
        let advection_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        operator.build_rhs(refs(&candidate), DT, &mut rhs)?;
        let iterations = if method == "learned" {
            // Inference and binary exchange are included in this stage's wall time.
            send_f64(&mut output, &rhs)?;
            output.flush()?;
            read_f64(&mut input, &mut pressure)?;
            0
        } else {
            let report = workspace.solve_rhs(&operator, &rhs, DT, settings, || false)?;
            pressure.copy_from_slice(workspace.pressure());
            report.iterations
        };
        let pressure_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        operator.apply_full(&pressure, &mut product)?;
        let mut residual2 = 0.0;
        let mut rhs2 = 0.0;
        for (a, b) in product.iter().zip(&rhs) {
            residual2 += (a - b).powi(2);
            rhs2 += b * b;
        }
        operator.correct_velocity(refs(&candidate), &pressure, DT, muts(&mut corrected))?;
        operator.divergence(refs(&corrected), &mut divergence)?;
        let div = divergence.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        ensure!(
            div <= ACTUAL_LIMIT,
            "stored divergence {div} exceeds {ACTUAL_LIMIT}"
        );
        let relative_residual = (residual2 / rhs2.max(1e-300)).sqrt();
        // Common gate for both paths; native PCG additionally uses stricter settings.
        ensure!(
            relative_residual <= 1e-5,
            "relative residual {relative_residual} exceeds 1e-5"
        );
        let correction_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        advect_tracer(
            &grid,
            &tracer,
            refs(&corrected),
            DT,
            &mut next_tracer,
            || false,
        )?;
        ensure!(
            next_tracer
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "tracer range"
        );
        let rate: f64 = (0..3)
            .map(|d| {
                corrected[d]
                    .iter()
                    .fold(0.0_f64, |a, b| a.max(f64::from(*b).abs()))
                    * n as f64
            })
            .sum();
        ensure!(DT * rate <= 1.0, "Courant limit");
        std::mem::swap(&mut old, &mut corrected);
        std::mem::swap(&mut tracer, &mut next_tracer);
        frame += 1;
        let tracer_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        send_json(
            &mut output,
            serde_json::json!({"frame":frame,"time":frame as f64*DT,
            "total_ms":total_ms,"pressure_ms":pressure_ms,"advection_ms":advection_ms,
            "correction_ms":correction_ms,"tracer_ms":tracer_ms,"iterations":iterations,
            "relative_residual":relative_residual,"actual_divergence_max":div,
            "courant":DT*rate,"tracer_integral":tracer.iter().map(|v| f64::from(*v)).sum::<f64>()*grid.cell_volume()}),
        )?;
        if capture {
            send_f64(&mut output, &rhs)?;
            send_f64(&mut output, &pressure)?;
            for component in &old {
                send_f32(&mut output, component)?;
            }
            send_f32(&mut output, &tracer)?;
        }
        output.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run().context("trajectory probe") {
        eprintln!("{e:#}");
        std::process::exit(1);
    }
}
