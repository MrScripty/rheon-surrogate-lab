//! Development-only deterministic native capture. Scientific failure is terminal;
//! output is a framed diagnostic stream, publication belongs to the Python caller.
use anyhow::{Context, Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace, ViscosityWorkspace, advect_tracer, advect_velocity,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::time::Instant;
type Velocity = [Vec<f32>; 3];
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.json"));
const RESERVATION: &str = include_str!(env!("RHEON_CAPTURE_RESERVATION"));
fn refs(v: &Velocity) -> [&[f32]; 3] {
    [&v[0], &v[1], &v[2]]
}
fn muts(v: &mut Velocity) -> [&mut [f32]; 3] {
    let [u, v, w] = v;
    [u, v, w]
}
fn emit(out: &mut impl Write, meta: Value, arrays: Vec<Vec<u8>>) -> Result<()> {
    let data = serde_json::to_vec(&meta)?;
    out.write_all(&u32::try_from(data.len())?.to_le_bytes())?;
    out.write_all(&data)?;
    for bytes in arrays {
        out.write_all(&bytes)?;
    }
    out.flush()?;
    Ok(())
}
fn f32bytes(a: &[f32]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn f64bytes(a: &[f64]) -> Vec<u8> {
    a.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn descriptor(name: &str, shape: [usize; 3], dtype: &str, units: &str) -> Value {
    json!({"name":name,"shape_xyz":shape,"dtype":dtype,"units":units,"layout":"x-fastest","bytes":shape.iter().product::<usize>()*if dtype=="<f8" {8}else{4}})
}
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ForceSpec {
    center_fraction: [f64; 3],
    direction: [f64; 3],
    acceleration_m_s2: f64,
    radius_fraction: f64,
    swirl_fraction: f64,
    fluctuation_fraction: f64,
    frequency_hz: f64,
    motion_fraction: f64,
    phase: f64,
    spatial_modes: [usize; 3],
}
fn force(
    grid: &GridGeometry,
    velocity: &mut Velocity,
    specification: &ForceSpec,
    time: f64,
    dt: f64,
) -> Result<()> {
    let h = grid.spacing();
    let extents = grid
        .counts()
        .map(|n| n as f64)
        .into_iter()
        .zip(h)
        .map(|(n, h)| n * h)
        .collect::<Vec<_>>();
    let radius =
        specification.radius_fraction * extents.iter().copied().fold(f64::INFINITY, f64::min);
    let omega = std::f64::consts::TAU * specification.frequency_hz;
    let center = std::array::from_fn::<_, 3, _>(|d| {
        extents[d]
            * (specification.center_fraction[d]
                + specification.motion_fraction
                    * (omega * time + specification.phase + d as f64).sin())
    });
    let pulse = 0.65 + 0.35 * (omega * time + specification.phase).sin();
    for (d, field) in velocity.iter_mut().enumerate() {
        let axis = AXES[d];
        let shape = grid.face_counts(axis);
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let p = [i, j, k];
                    if p[d] == 0 || p[d] + 1 == shape[d] {
                        continue;
                    }
                    let pos = grid.face_position(axis, p).context("force face position")?;
                    let offset = std::array::from_fn::<_, 3, _>(|a| (pos[a] - center[a]) / radius);
                    let r2 = offset.iter().map(|n| n * n).sum::<f64>();
                    // A finite, smoothly tapered force region avoids infinitesimal
                    // Gaussian tails entering the native normal-f32 viscosity contract.
                    let envelope = if r2 < 16.0 {
                        (-0.5 * r2).exp() * (1.0 - r2 / 16.0).powi(2)
                    } else {
                        0.0
                    };
                    let a = (d + 1) % 3;
                    let b = (d + 2) % 3;
                    let rotation = specification.direction[a] * offset[b]
                        - specification.direction[b] * offset[a];
                    let wave = (0..3)
                        .map(|a| {
                            (std::f64::consts::TAU * specification.spatial_modes[a] as f64 * pos[a]
                                / extents[a]
                                + omega * time * (a + 1) as f64
                                + specification.phase
                                + d as f64)
                                .sin()
                        })
                        .sum::<f64>()
                        / 3.0;
                    let acceleration = specification.acceleration_m_s2
                        * pulse
                        * envelope
                        * (specification.direction[d]
                            + specification.swirl_fraction * rotation
                            + specification.fluctuation_fraction * wave);
                    let index = grid.face_index(axis, p).context("force index")?;
                    field[index] = (f64::from(field[index]) + dt * acceleration) as f32;
                }
            }
        }
    }
    Ok(())
}
fn gate(
    operator: &PressureOperator<'_>,
    pre: &Velocity,
    pressure: &[f64],
    rhs: &[f64],
    dt: f64,
    corrected: &mut Velocity,
) -> Result<Value> {
    ensure!(
        pressure[0] == 0.0 && pressure.iter().all(|p| p.is_finite()),
        "pressure gauge/finite"
    );
    let mut product = vec![0.0; rhs.len()];
    operator.apply_full(pressure, &mut product)?;
    let mut l2 = 0.0;
    let mut b2 = 0.0;
    let mut maximum = 0.0_f64;
    for (a, b) in product.iter().zip(rhs) {
        l2 += (b - a).powi(2);
        b2 += b * b;
        maximum = maximum.max((b - a).abs());
    }
    let residual = l2.sqrt();
    let predicted = dt * maximum / operator.geometry().cell_volume();
    ensure!(
        residual <= 1e-12_f64.max(1e-8 * b2.sqrt()),
        "full original-b residual: {residual}"
    );
    ensure!(predicted <= 1e-7, "predicted divergence: {predicted}");
    operator.correct_velocity(refs(pre), pressure, dt, muts(corrected))?;
    operator.divergence(refs(corrected), &mut product)?;
    let actual = product.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    ensure!(actual <= 1e-5, "actual corrected-f32 divergence: {actual}");
    let volume = operator.geometry().cell_volume();
    let net = product.iter().sum::<f64>() * volume;
    let sumabs = product.iter().map(|v| v.abs()).sum::<f64>() * volume;
    ensure!(
        net.abs() <= 1e-12_f64.max(1e-8 * sumabs),
        "net volume flux: {net}"
    );
    let grid = operator.geometry();
    for (d, field) in corrected.iter().enumerate() {
        let dims = grid.face_counts(AXES[d]);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = [i, j, k];
                    if p[d] == 0 || p[d] + 1 == dims[d] {
                        ensure!(
                            field[grid.face_index(AXES[d], p).context("wall face")?] == 0.0,
                            "wall flux"
                        );
                    }
                }
            }
        }
    }
    let courant: f64 = (0..3)
        .map(|d| {
            dt * corrected[d]
                .iter()
                .fold(0.0_f64, |a, b| a.max(f64::from(*b).abs()))
                / grid.spacing()[d]
        })
        .sum();
    ensure!(courant <= 0.5, "Courant: {courant}");
    Ok(
        json!({"residual_l2":residual,"residual_max":maximum,"rhs_l2":b2.sqrt(),"relative_residual":if b2>0.0{Some(residual/b2.sqrt())}else{None},"predicted_divergence_max_s_inv":predicted,"actual_divergence_max_s_inv":actual,"actual_divergence_rms_s_inv":(product.iter().map(|v|v*v).sum::<f64>()/product.len()as f64).sqrt(),"net_flux_m3_s":net,"courant":courant}),
    )
}
fn cubic_grid(extents: [f64; 3], cells_per_metre: u64) -> Result<GridGeometry> {
    ensure!(
        extents.iter().all(|v| v.is_finite() && *v > 0.0),
        "finite positive domain lengths"
    );
    ensure!(
        (1..=128).contains(&cells_per_metre),
        "cells per metre bound"
    );
    let counts =
        std::array::from_fn::<_, 3, _>(|d| (extents[d] * cells_per_metre as f64).round() as usize);
    ensure!(
        counts.iter().all(|n| (2..=128).contains(n)) && counts.iter().product::<usize>() <= 65_536,
        "cell-count budget"
    );
    ensure!(
        (0..3).all(|d| ((extents[d] * cells_per_metre as f64) - counts[d] as f64).abs() <= 1e-10),
        "domain lengths must be whole cell multiples"
    );
    let cell_size = 1.0 / cells_per_metre as f64;
    Ok(GridGeometry::new(
        counts.map(|n| n as u64),
        [cell_size; 3],
        [0.0; 3],
    )?)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    trajectory_id: String,
    fixture: bool,
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--identity"] {
        println!("{IDENTITY}");
        return Ok(());
    }
    ensure!(args == ["--produce"], "usage --produce or --identity");
    ensure!(
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()) == "1",
        "one native thread"
    );
    let mut raw = String::new();
    std::io::stdin().take(65537).read_to_string(&mut raw)?;
    ensure!(raw.len() <= 65536, "request bound");
    let request: Request = serde_json::from_str(&raw)?;
    let reservation: Value = serde_json::from_str(RESERVATION)?;
    let config = &reservation["configuration"];
    ensure!(
        config["forcing_profile"] == "compact-gaussian-taper-v1",
        "qualified finite-support force profile"
    );
    let entry = reservation["scenes"]
        .as_array()
        .context("scenes")?
        .iter()
        .find(|row| row["trajectory_id"].as_str() == Some(&request.trajectory_id))
        .context("unreserved scene")?;
    ensure!(
        !request.fixture || entry["split"] == "train",
        "fixture must be a training scene"
    );
    let counts: [u64; 3] = serde_json::from_value(entry["counts_xyz"].clone())?;
    let density = entry["cells_per_metre"].as_u64().context("resolution")?;
    let grid = cubic_grid(counts.map(|n| n as f64 / density as f64), density)?;
    let counts = grid.counts();
    let rho = config["material"]["density_kg_m3"]
        .as_f64()
        .context("water density")?;
    let mu = config["material"]["dynamic_viscosity_pa_s"]
        .as_f64()
        .context("water viscosity")?;
    ensure!(rho == 1000.0 && mu == 0.001, "qualified water material");
    let operator = PressureOperator::new(&grid, rho)?;
    let mut jacobi = PressureWorkspace::with_implementation(
        &grid,
        128 * 1024 * 1024,
        PressureImplementation::JacobiPcgV1,
    )?;
    let mut viscosity = ViscosityWorkspace::new(grid.clone(), 128 * 1024 * 1024)?;
    let settings = PressureSettings {
        relative_residual: 1e-8,
        absolute_residual: 1e-12,
        divergence_limit: 1e-7,
        max_iterations: 3000,
    };
    let mut old: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
    let mut pre = old.clone();
    let mut viscous = old.clone();
    let mut corrected = old.clone();
    let mut tracer = vec![0.0; grid.cell_len()];
    let mut next_tracer = tracer.clone();
    let extents = counts.map(|n| n as f64 / density as f64);
    let width = 0.08 * extents.into_iter().fold(f64::INFINITY, f64::min);
    for k in 0..counts[2] {
        for j in 0..counts[1] {
            for i in 0..counts[0] {
                let pos = grid.cell_position([i, j, k]).context("cell position")?;
                let r2 = (0..3)
                    .map(|d| (pos[d] - 0.5 * extents[d]).powi(2))
                    .sum::<f64>();
                tracer[i + counts[0] * (j + counts[1] * k)] =
                    (0.25 * (-r2 / (2.0 * width * width)).exp()) as f32;
            }
        }
    }
    let mut base = vec![0.0; grid.cell_len()];
    let mut rhs = base.clone();
    let zero = base.clone();
    let mut residual = base.clone();
    let force_spec: ForceSpec = serde_json::from_value(entry["force"].clone())?;
    let steps = if request.fixture {
        1
    } else {
        config["steps"].as_u64().context("steps")? as usize
    };
    let selected: Vec<usize> = if request.fixture {
        vec![0]
    } else {
        serde_json::from_value(config["snapshot_steps"].clone())?
    };
    ensure!(
        steps <= 64 && selected.iter().all(|n| *n < steps),
        "bounded complete snapshot selection"
    );
    let dt = config["dt_s"].as_f64().context("time step")?;
    ensure!(dt == 1.0 / 60.0, "qualified time step");
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for frame in 0..steps {
        let start = Instant::now();
        advect_velocity(&grid, refs(&old), dt, muts(&mut pre), || false)?;
        advect_tracer(&grid, &tracer, refs(&old), dt, &mut next_tracer, || false)?;
        force(&grid, &mut pre, &force_spec, frame as f64 * dt, dt)?;
        let viscous_report =
            viscosity.update(refs(&pre), rho, mu, dt, muts(&mut viscous), |_| false)?;
        operator.build_rhs(refs(&viscous), dt, &mut rhs)?;
        operator.apply_full(&base, &mut residual)?;
        for (r, b) in residual.iter_mut().zip(&rhs) {
            *r = *b - *r;
        }
        let report =
            jacobi
                .solve_rhs_with_initial_pressure(&operator, &rhs, &zero, dt, settings, || false)?;
        let metrics = gate(
            &operator,
            &viscous,
            jacobi.pressure(),
            &rhs,
            dt,
            &mut corrected,
        )?;
        if selected.contains(&frame) {
            let mut descriptors = Vec::new();
            let mut arrays = Vec::new();
            for (name, field) in [
                ("base", base.as_slice()),
                ("rhs", rhs.as_slice()),
                ("pressure", jacobi.pressure()),
                ("native_input_residual", residual.as_slice()),
            ] {
                descriptors.push(descriptor(
                    name,
                    counts,
                    "<f8",
                    if name == "base" || name == "pressure" {
                        "Pa"
                    } else {
                        "m3/s2"
                    },
                ));
                arrays.push(f64bytes(field));
            }
            descriptors.push(descriptor("tracer", counts, "<f4", "dimensionless"));
            arrays.push(f32bytes(&next_tracer));
            for (d, axis) in ["u", "v", "w"].iter().enumerate() {
                for (prefix, field) in [("pre", &viscous[d]), ("corrected", &corrected[d])] {
                    descriptors.push(descriptor(
                        &format!("{prefix}_{axis}"),
                        grid.face_counts(AXES[d]),
                        "<f4",
                        "m/s",
                    ));
                    arrays.push(f32bytes(field));
                }
            }
            emit(
                &mut out,
                json!({"kind":"frame","index":frame,"time_s":(frame+1) as f64*dt,"arrays":descriptors,"metrics":metrics,"iterations":report.iterations,"viscosity":{"dynamic_viscosity_pa_s":viscous_report.dynamic_viscosity,"kinematic_viscosity_m2_s":viscous_report.kinematic_viscosity,"kinetic_before":viscous_report.kinetic_before,"kinetic_after":viscous_report.kinetic_after},"elapsed_s":start.elapsed().as_secs_f64()}),
                arrays,
            )?;
        }
        base.copy_from_slice(jacobi.pressure());
        std::mem::swap(&mut old, &mut corrected);
        std::mem::swap(&mut tracer, &mut next_tracer);
    }
    emit(
        &mut out,
        json!({"kind":"complete","steps":steps,"snapshots":selected.len(),"native_steps_qualified":steps}),
        Vec::new(),
    )?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("water_capture_failure: {error:#}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn narrow_force_on_long_box_is_admissible_to_water_viscosity() -> Result<()> {
        let grid = cubic_grid([0.625, 1.375, 2.0], 8)?;
        let mut input: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
        let mut output = input.clone();
        let spec = ForceSpec {
            center_fraction: [
                0.46720166755843445,
                0.22715770572461036,
                0.32990075108739936,
            ],
            direction: [
                -0.11219211884632663,
                -0.9901122783836817,
                -0.0842057282056666,
            ],
            acceleration_m_s2: 1.785258832826408,
            radius_fraction: 0.19759954399224947,
            swirl_fraction: 0.6717947718282258,
            fluctuation_fraction: 0.4415767904555133,
            frequency_hz: 1.6458079390315068,
            motion_fraction: 0.05274008957008278,
            phase: 4.347860656691315,
            spatial_modes: [1, 1, 2],
        };
        force(&grid, &mut input, &spec, 0.0, 1.0 / 60.0)?;
        let subnormal = input
            .iter()
            .flatten()
            .filter(|v| **v != 0.0 && !v.is_normal())
            .count();
        let mut viscosity = ViscosityWorkspace::new(grid, 128 * 1024 * 1024)?;
        let result = viscosity.update(
            refs(&input),
            1000.0,
            0.001,
            1.0 / 60.0,
            muts(&mut output),
            |_| false,
        );
        assert!(
            result.is_ok(),
            "force input subnormals={subnormal}; actual viscosity result={result:?}"
        );
        Ok(())
    }
    #[test]
    fn cubic_geometry_and_force_are_deterministic_without_wall_flux() -> Result<()> {
        let grid = cubic_grid([1.0, 0.625, 0.375], 16)?;
        assert_eq!(grid.counts(), [16, 10, 6]);
        let mut first: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
        let mut second = first.clone();
        let spec = ForceSpec {
            center_fraction: [0.5, 0.4, 0.6],
            direction: [0.0, 1.0, 0.0],
            acceleration_m_s2: 1.0,
            radius_fraction: 0.25,
            swirl_fraction: 0.8,
            fluctuation_fraction: 0.4,
            frequency_hz: 1.0,
            motion_fraction: 0.1,
            phase: 0.2,
            spatial_modes: [1, 2, 1],
        };
        force(&grid, &mut first, &spec, 0.2, 1.0 / 60.0)?;
        force(&grid, &mut second, &spec, 0.2, 1.0 / 60.0)?;
        assert_eq!(first, second);
        assert!(first.iter().flatten().any(|v| *v != 0.0));
        let operator = PressureOperator::new(&grid, 1000.0)?;
        let mut rhs = vec![0.0; grid.cell_len()];
        operator.build_rhs(refs(&first), 1.0 / 60.0, &mut rhs)?;
        assert!(rhs.iter().sum::<f64>().abs() < 1e-12);
        Ok(())
    }
}
