//! Development-only deterministic native capture. Scientific failure is terminal;
//! output is a framed diagnostic stream, publication belongs to the Python caller.
use anyhow::{Context, Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace, advect_tracer, advect_velocity,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::time::Instant;
type Velocity = [Vec<f32>; 3];
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.json"));
const RESERVATION: &str =
    include_str!("../../../docs/plans/global-pressure-correction-v1/reservation.json");
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
fn noise(seed: u64, frame: usize, axis: usize, index: usize) -> f64 {
    let mut bits = (index as u64)
        .wrapping_add(seed.wrapping_mul(7919))
        .wrapping_add((frame as u64).wrapping_mul(104729))
        .wrapping_add((axis as u64).wrapping_mul(997));
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    2.0 * ((bits >> 40) as f64 / 16_777_216.0) - 1.0
}
#[derive(Clone, Copy)]
struct ForcingScene {
    extents: [f64; 3],
    seed: u64,
    dt: f64,
    amplitude: f64,
    steady: bool,
}
fn force(
    grid: &GridGeometry,
    velocity: &mut Velocity,
    scene: &ForcingScene,
    frame: usize,
) -> Result<()> {
    let ForcingScene {
        extents,
        seed,
        dt,
        amplitude,
        steady,
    } = *scene;
    let phi = (seed % 997) as f64 * 0.137;
    let t = if steady { 0.0 } else { frame as f64 * dt };
    let center = [
        0.43 + 0.13 * (phi + t).sin(),
        0.28 + 0.06 * phi.cos(),
        0.45 + 0.13 * (phi - t).cos(),
    ];
    let direction = [
        0.75 * (phi + 2.0 * t).cos(),
        1.0,
        0.75 * (phi + 2.0 * t).sin(),
    ];
    let width = 0.1 * extents.into_iter().fold(f64::INFINITY, f64::min);
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
                    let pos = grid.face_position(axis, p).context("face position")?;
                    let r2: f64 = (0..3)
                        .map(|a| (pos[a] - center[a] * extents[a]).powi(2))
                        .sum();
                    let index = grid.face_index(axis, p).context("face index")?;
                    let acceleration = amplitude
                        * ((-r2 / (2.0 * width * width)).exp() * direction[d]
                            + 0.025 * noise(seed, frame, d, index));
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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    preset: String,
    trajectory_id: String,
    frames: usize,
    fixture: bool,
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--identity"] {
        println!("{IDENTITY}");
        return Ok(());
    }
    ensure!(
        args == ["--produce"],
        "usage: --identity | --produce (development request on stdin)"
    );
    ensure!(
        !cfg!(feature = "parallel") || std::env::var("RAYON_NUM_THREADS").as_deref() == Ok("1"),
        "parallel producer requires explicit RAYON_NUM_THREADS=1"
    );
    ensure!(
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()) == "1",
        "native one thread required"
    );
    let mut request = String::new();
    std::io::stdin().take(65537).read_to_string(&mut request)?;
    ensure!(request.len() <= 65536, "request size");
    let request: Request = serde_json::from_str(&request)?;
    ensure!(request.preset == "smoke", "only smoke DEVELOPMENT admitted");
    let id = request.trajectory_id.as_str();
    let reservation: Value = serde_json::from_str(RESERVATION)?;
    let entry = reservation["presets"]["smoke"]["trajectories"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["trajectory_id"] == id)
        .context("unreserved trajectory")?;
    ensure!(
        entry["split"] == "train" || entry["split"] == "validation",
        "final test prohibited"
    );
    let frames = request.frames;
    ensure!((1..=24).contains(&frames), "frame count");
    ensure!(
        frames == 24 || (entry["split"] == "train" && request.fixture),
        "short capture is TRAIN fixture only"
    );
    let seed = entry["seed"].as_u64().unwrap();
    let dt = entry["dt_s"].as_f64().unwrap();
    let amplitude = entry["amplitude_m_s2"].as_f64().unwrap();
    let regime = entry["forcing_regime"].as_str().unwrap();
    ensure!(
        ["steady", "moving"].contains(&regime),
        "heldout forcing prohibited"
    );
    let e = &reservation["geometries_extents_m"][entry["geometry"].as_str().unwrap()];
    let extents = std::array::from_fn(|d| e[d].as_f64().unwrap());
    let grid = GridGeometry::new([16; 3], extents.map(|v| v / 16.0), [0.0; 3])?;
    let operator = PressureOperator::new(&grid, 1.25)?;
    let mut jacobi = PressureWorkspace::with_implementation(
        &grid,
        128 * 1024 * 1024,
        PressureImplementation::JacobiPcgV1,
    )?;
    let mut sgs = PressureWorkspace::with_implementation(
        &grid,
        128 * 1024 * 1024,
        PressureImplementation::SymmetricGaussSeidelPcgV1,
    )?;
    let settings = PressureSettings {
        relative_residual: 1e-8,
        absolute_residual: 1e-12,
        divergence_limit: 1e-7,
        max_iterations: 3000,
    };
    let mut old: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
    let mut pre = old.clone();
    let mut corrected = old.clone();
    let mut tracer = vec![0.0; grid.cell_len()];
    let width = 0.08 * extents.into_iter().fold(f64::INFINITY, f64::min);
    for k in 0..16 {
        for j in 0..16 {
            for i in 0..16 {
                let p = grid.cell_position([i, j, k]).context("cell position")?;
                let r2: f64 = (0..3).map(|d| (p[d] - extents[d] * 0.5).powi(2)).sum();
                tracer[i + 16 * (j + 16 * k)] = (0.25 * (-r2 / (2.0 * width * width)).exp()) as f32;
            }
        }
    }
    let mut next_tracer = tracer.clone();
    let mut rhs = vec![0.0; grid.cell_len()];
    let zero = rhs.clone();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let mut descriptors = vec![descriptor("pressure", [16; 3], "<f8", "Pa")];
    let mut arrays = vec![f64bytes(&zero)];
    for (d, name) in ["u", "v", "w"].iter().enumerate() {
        descriptors.push(descriptor(name, grid.face_counts(AXES[d]), "<f4", "m/s"));
        arrays.push(f32bytes(&old[d]));
    }
    descriptors.push(descriptor("tracer", [16; 3], "<f4", "dimensionless"));
    arrays.push(f32bytes(&tracer));
    emit(
        &mut out,
        json!({"kind":"initial","arrays":descriptors}),
        arrays,
    )?;
    let scene = ForcingScene {
        extents,
        seed,
        dt,
        amplitude,
        steady: regime == "steady",
    };
    for frame in 0..frames {
        let start = Instant::now();
        advect_velocity(&grid, refs(&old), dt, muts(&mut pre), || false)?;
        advect_tracer(&grid, &tracer, refs(&old), dt, &mut next_tracer, || false)?;
        force(&grid, &mut pre, &scene, frame)?;
        operator.build_rhs(refs(&pre), dt, &mut rhs)?;
        // Zero-start reference still exercises the new sealed initial-pressure API.
        let report =
            jacobi
                .solve_rhs_with_initial_pressure(&operator, &rhs, &zero, dt, settings, || false)?;
        let mut metrics = gate(&operator, &pre, jacobi.pressure(), &rhs, dt, &mut corrected)?;
        if frame == 0 {
            let check = sgs.solve_rhs(&operator, &rhs, dt, settings, || false)?;
            let mut witness = old.clone();
            let gates = gate(&operator, &pre, sgs.pressure(), &rhs, dt, &mut witness)?;
            metrics["sgs"] = json!({"iterations":check.iterations,"gates":gates,"pressure_difference_l2":sgs.pressure().iter().zip(jacobi.pressure()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt()});
        }
        ensure!(
            next_tracer.iter().all(|v| v.is_finite()),
            "nonfinite passive tracer"
        );
        let mut descriptors = vec![
            descriptor("rhs", [16; 3], "<f8", "m3/s2"),
            descriptor("pressure", [16; 3], "<f8", "Pa"),
        ];
        let mut arrays = vec![f64bytes(&rhs), f64bytes(jacobi.pressure())];
        for (d, name) in ["u", "v", "w"].iter().enumerate() {
            for (prefix, state) in [("pre", &pre), ("corrected", &corrected)] {
                descriptors.push(descriptor(
                    &format!("{prefix}_{name}"),
                    grid.face_counts(AXES[d]),
                    "<f4",
                    "m/s",
                ));
                arrays.push(f32bytes(&state[d]));
            }
        }
        descriptors.push(descriptor("tracer", [16; 3], "<f4", "dimensionless"));
        arrays.push(f32bytes(&next_tracer));
        let tracer_mass =
            next_tracer.iter().map(|v| f64::from(*v)).sum::<f64>() * grid.cell_volume();
        emit(
            &mut out,
            json!({"kind":"frame","index":frame,"forcing_time_s":frame as f64*dt,"time_s":(frame+1)as f64*dt,"arrays":descriptors,"metrics":metrics,"iterations":report.iterations,"tracer_integral_m3":tracer_mass,"elapsed_s":start.elapsed().as_secs_f64()}),
            arrays,
        )?;
        std::mem::swap(&mut old, &mut corrected);
        std::mem::swap(&mut tracer, &mut next_tracer);
    }
    emit(&mut out, json!({"kind":"complete","frames":frames}), vec![])?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("capture_failure: {error:#}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_mixer_has_literal_wrapping_and_frame_identity() {
        assert_eq!(noise(100000, 0, 0, 1), -0.4016076326370239);
        assert_eq!(noise(100000, 1, 0, 1), -0.726251482963562);
    }
    #[test]
    fn native_dyadic_anisotropic_operator_matches_hand_stencil() -> Result<()> {
        let grid = GridGeometry::new([2, 2, 1], [0.5, 0.25, 2.0], [0.0; 3])?;
        let operator = PressureOperator::new(&grid, 2.0)?;
        let mut product = vec![0.0; 4];
        operator.apply_full(&[0.0, 1.0, 2.0, 3.0], &mut product)?;
        assert_eq!(product, [-4.5, -3.5, 3.5, 4.5]);
        let mut pre: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
        pre[0][1] = 0.125;
        pre[0][4] = 0.125;
        pre[1][2] = 0.5;
        pre[1][3] = 0.5;
        let mut rhs = vec![0.0; 4];
        operator.build_rhs(refs(&pre), 0.125, &mut rhs)?;
        assert_eq!(rhs, product);
        let mut corrected = pre.clone();
        operator.correct_velocity(
            refs(&pre),
            &[0.0, 1.0, 2.0, 3.0],
            0.125,
            muts(&mut corrected),
        )?;
        assert!(corrected.iter().flatten().all(|v| *v == 0.0));
        Ok(())
    }
    #[test]
    fn forcing_preserves_sealed_walls_and_is_bitwise_repeatable() -> Result<()> {
        let grid = GridGeometry::new([16; 3], [1.2 / 16.0, 1.0 / 16.0, 0.8 / 16.0], [0.0; 3])?;
        let mut first: Velocity = std::array::from_fn(|d| vec![0.0; grid.face_len(AXES[d])]);
        let mut second = first.clone();
        let scene = ForcingScene {
            extents: [1.2, 1.0, 0.8],
            seed: 100001,
            dt: 1.0 / 60.0,
            amplitude: 0.5,
            steady: true,
        };
        force(&grid, &mut first, &scene, 3)?;
        force(&grid, &mut second, &scene, 3)?;
        assert_eq!(first, second);
        assert!(first.iter().flatten().all(|v| v.is_finite()));
        for (d, field) in first.iter().enumerate() {
            let dims = grid.face_counts(AXES[d]);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let p = [i, j, k];
                        if p[d] == 0 || p[d] + 1 == dims[d] {
                            assert_eq!(field[grid.face_index(AXES[d], p).unwrap()], 0.0);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
