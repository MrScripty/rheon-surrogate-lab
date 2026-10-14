//! Teacher-state projection diagnostics only. No simulation state is advanced.
use anyhow::{Context, Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureError, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Read, time::Instant};
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.json"));
const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
type Velocity = [Vec<f32>; 3];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidualRequest {
    schema: String,
    counts_xyz: [u64; 3],
    spacing_m: [f64; 3],
    density_kg_m3: f64,
    original_rhs: Vec<f64>,
    base_pressure: Vec<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    counts_xyz: [u64; 3],
    spacing_m: [f64; 3],
    density_kg_m3: f64,
    dt_s: f64,
    original_rhs: Vec<f64>,
    reference_pressure: Vec<f64>,
    previous_pressure: Vec<f64>,
    candidate_pressure: Option<Vec<f64>>,
    candidate_error: Option<String>,
    pre_velocity: Velocity,
}
fn refs(v: &Velocity) -> [&[f32]; 3] {
    [&v[0], &v[1], &v[2]]
}
fn muts(v: &mut Velocity) -> [&mut [f32]; 3] {
    let [u, v, w] = v;
    [u, v, w]
}
fn settings(limit: usize) -> PressureSettings {
    PressureSettings {
        relative_residual: 1e-8,
        absolute_residual: 1e-12,
        divergence_limit: 1e-7,
        max_iterations: limit,
    }
}
fn pressure_error(p: &[f64], n: usize) -> Option<&'static str> {
    if p.len() != n {
        Some("prediction_shape")
    } else if p.iter().any(|v| !v.is_finite()) {
        Some("prediction_nonfinite")
    } else if p[0] != 0.0 {
        Some("prediction_gauge")
    } else {
        None
    }
}
fn walls(grid: &GridGeometry, v: &Velocity) -> Result<()> {
    for (d, field) in v.iter().enumerate() {
        ensure!(
            field.len() == grid.face_len(AXES[d]) && field.iter().all(|v| v.is_finite()),
            "velocity shape/finiteness"
        );
        let shape = grid.face_counts(AXES[d]);
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let p = [i, j, k];
                    if p[d] == 0 || p[d] + 1 == shape[d] {
                        ensure!(
                            field[grid.face_index(AXES[d], p).context("wall index")?] == 0.0,
                            "wall-normal flux"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
fn quality(
    op: &PressureOperator<'_>,
    pre: &Velocity,
    b: &[f64],
    p: &[f64],
    dt: f64,
) -> Result<Value> {
    if let Some(reason) = pressure_error(p, op.geometry().cell_len()) {
        return Ok(json!({"qualified":false,"failures":[reason]}));
    }
    let mut product = vec![0.0; p.len()];
    if let Err(error) = op.apply_full(p, &mut product) {
        return Ok(json!({"qualified":false,"failures":[format!("operator: {error}")]}));
    }
    let rhs2 = b.iter().map(|v| v * v).sum::<f64>();
    let residual2 = product
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let maximum = product
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let residual = residual2.sqrt();
    let rhs = rhs2.sqrt();
    let volume = op.geometry().cell_volume();
    let predicted = dt * maximum / volume;
    let mut corrected: Velocity = std::array::from_fn(|d| vec![0.0; pre[d].len()]);
    if let Err(error) = op.correct_velocity(refs(pre), p, dt, muts(&mut corrected)) {
        return Ok(json!({"qualified":false,"failures":[format!("correction: {error}")]}));
    }
    op.divergence(refs(&corrected), &mut product)?;
    let actual = product.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let net = product.iter().sum::<f64>() * volume;
    let absflux = product.iter().map(|v| v.abs()).sum::<f64>() * volume;
    let courant = (0..3)
        .map(|d| {
            dt * corrected[d]
                .iter()
                .fold(0.0_f64, |a, b| a.max(f64::from(*b).abs()))
                / op.geometry().spacing()[d]
        })
        .sum::<f64>();
    let mut failures = Vec::new();
    if !(residual.is_finite() && residual <= 1e-12_f64.max(1e-8 * rhs)) {
        failures.push("original_rhs_full_residual");
    }
    if !(predicted.is_finite() && predicted <= 1e-7) {
        failures.push("predicted_divergence");
    }
    if !(actual.is_finite() && actual <= 1e-5) {
        failures.push("actual_f32_divergence");
    }
    if !(net.is_finite() && net.abs() <= 1e-12_f64.max(1e-8 * absflux)) {
        failures.push("net_volume_flux");
    }
    if walls(op.geometry(), &corrected).is_err() {
        failures.push("wall_normal_flux");
    }
    if !(courant.is_finite() && courant <= 0.5) {
        failures.push("courant");
    }
    Ok(
        json!({"qualified":failures.is_empty(),"failures":failures,"residual_l2":residual,"residual_max":maximum,"rhs_l2":rhs,"relative_residual":if rhs>0.0 {Some(residual/rhs)}else{None},"predicted_divergence_max_s_inv":predicted,"actual_divergence_max_s_inv":actual,"actual_divergence_rms_s_inv":(product.iter().map(|v|v*v).sum::<f64>()/product.len()as f64).sqrt(),"net_flux_m3_s":net,"wall_normal_flux_exact_zero":walls(op.geometry(),&corrected).is_ok(),"courant":courant,"gauge_cell":p[0]}),
    )
}
fn error_iterations(e: &PressureError) -> Option<usize> {
    match e {
        PressureError::Breakdown { iteration } => Some(*iteration),
        PressureError::IterationLimit { iterations, .. }
        | PressureError::Cancelled { iterations } => Some(*iterations),
        _ => None,
    }
}
fn solve(
    op: &PressureOperator<'_>,
    request: &Request,
    guess: Option<&[f64]>,
    limit: usize,
) -> Result<Value> {
    let start = Instant::now();
    let mut guess_hash = Sha256::new();
    for index in 0..op.geometry().cell_len() {
        guess_hash.update(guess.map_or(0.0, |p| p[index]).to_le_bytes());
    }
    let initial_guess_sha256 = format!("{:x}", guess_hash.finalize());
    // A fresh workspace makes the single reset independent of all failed scratch.
    let mut workspace = PressureWorkspace::with_implementation(
        op.geometry(),
        4 * 1024 * 1024,
        PressureImplementation::JacobiPcgV1,
    )?;
    let report = match guess {
        Some(p) => workspace.solve_rhs_with_initial_pressure(
            op,
            &request.original_rhs,
            p,
            request.dt_s,
            settings(limit),
            || false,
        ),
        None => workspace.solve_rhs(
            op,
            &request.original_rhs,
            request.dt_s,
            settings(limit),
            || false,
        ),
    };
    let solver_s = start.elapsed().as_secs_f64();
    let check_start = Instant::now();
    let q = quality(
        op,
        &request.pre_velocity,
        &request.original_rhs,
        workspace.pressure(),
        request.dt_s,
    )?;
    let check_s = check_start.elapsed().as_secs_f64();
    let (iterations, error) = match report {
        Ok(r) => (Some(r.iterations), None),
        Err(e) => (error_iterations(&e), Some(e.to_string())),
    };
    Ok(
        json!({"qualified":error.is_none() && q["qualified"]==true,"solver_error":error,"iterations":iterations,"iteration_limit":limit,"initial_guess_sha256":initial_guess_sha256,"initial_guess_kind":if guess.is_some(){"supplied_pressure"}else{"zero"},"solver_s":solver_s,"verification_s":check_s,"quality":q}),
    )
}
fn corrected(
    op: &PressureOperator<'_>,
    request: &Request,
    proposal: Option<&[f64]>,
    proposal_quality: &Value,
    base_quality: &Value,
    numerical_error: Option<&str>,
) -> Result<Value> {
    let start = Instant::now();
    let rejection = numerical_error.map(str::to_owned).or_else(|| {
        if proposal_quality["residual_l2"]
            .as_f64()
            .zip(base_quality["residual_l2"].as_f64())
            .is_none_or(|(p, b)| p > b)
        {
            Some("proposal_residual_worse_than_temporal".into())
        } else {
            None
        }
    });
    if rejection.is_none() && proposal_quality["qualified"] == true {
        return Ok(
            json!({"qualified":true,"outcome":"direct_acceptance","proposal_rejection":null,"warm":null,"reset":null,"any_pcg":false,"reset_fallback":false,"elapsed_s":start.elapsed().as_secs_f64()}),
        );
    }
    let warm = if rejection.is_none() {
        Some(solve(op, request, proposal, 32)?)
    } else {
        None
    };
    if warm.as_ref().is_some_and(|w| w["qualified"] == true) {
        return Ok(
            json!({"qualified":true,"outcome":"warm_correction","proposal_rejection":null,"warm":warm,"reset":null,"any_pcg":true,"reset_fallback":false,"elapsed_s":start.elapsed().as_secs_f64()}),
        );
    }
    let reset = solve(op, request, Some(&request.previous_pressure), 3000)?;
    Ok(
        json!({"qualified":reset["qualified"],"outcome":if reset["qualified"]==true {"temporal_reset_fallback"}else{"terminal_failure"},"proposal_rejection":rejection,"warm":warm,"reset":reset,"any_pcg":true,"reset_fallback":true,"elapsed_s":start.elapsed().as_secs_f64()}),
    )
}
fn temporal_full_budget(op: &PressureOperator<'_>, request: &Request) -> Result<Value> {
    // This baseline pays one fresh full-budget attempt, with no preliminary
    // bounded work. It is distinct from the32-step temporal path diagnostic.
    let mut result = solve(op, request, Some(&request.previous_pressure), 3000)?;
    result["comparison_role"] = json!(
        "temporal PCG baseline: one fresh full-budget attempt from previous accepted pressure"
    );
    Ok(result)
}
fn cubic_grid(counts: [u64; 3], spacing: [f64; 3], rho: f64) -> Result<GridGeometry> {
    ensure!(
        counts.iter().all(|n| (1..=65_536).contains(n)) && counts.iter().product::<u64>() <= 65_536,
        "cell budget"
    );
    ensure!(
        spacing.iter().all(|h| h.is_finite() && *h > 0.0)
            && spacing[0] == spacing[1]
            && spacing[1] == spacing[2],
        "cubic cells require one shared spacing"
    );
    ensure!(rho == 1.25, "development density");
    Ok(GridGeometry::new(counts, spacing, [0.0; 3])?)
}
fn evaluate(request: &Request) -> Result<Value> {
    let start = Instant::now();
    ensure!(
        request.schema == "rheon-variable-pressure-teacher-case-v1",
        "case schema"
    );
    ensure!(request.dt_s == 1.0 / 60.0, "development timestep");
    let grid = cubic_grid(request.counts_xyz, request.spacing_m, request.density_kg_m3)?;
    let op = PressureOperator::new(&grid, request.density_kg_m3)?;
    walls(&grid, &request.pre_velocity)?;
    ensure!(
        request.original_rhs.len() == grid.cell_len()
            && request.original_rhs.iter().all(|v| v.is_finite()),
        "RHS shape/finiteness"
    );
    ensure!(
        pressure_error(&request.previous_pressure, grid.cell_len()).is_none(),
        "previous-pressure contract"
    );
    let mut reconstructed = vec![0.0; grid.cell_len()];
    op.build_rhs(
        refs(&request.pre_velocity),
        request.dt_s,
        &mut reconstructed,
    )?;
    ensure!(
        reconstructed == request.original_rhs,
        "native original RHS reconstruction mismatch"
    );
    let reference = quality(
        &op,
        &request.pre_velocity,
        &request.original_rhs,
        &request.reference_pressure,
        request.dt_s,
    )?;
    ensure!(
        reference["qualified"] == true,
        "reference physics failed: {reference}"
    );
    let zero = vec![0.0; grid.cell_len()];
    let temporal_quality = quality(
        &op,
        &request.pre_velocity,
        &request.original_rhs,
        &request.previous_pressure,
        request.dt_s,
    )?;
    let zero_quality = quality(
        &op,
        &request.pre_velocity,
        &request.original_rhs,
        &zero,
        request.dt_s,
    )?;
    let candidate_error = request
        .candidate_error
        .as_deref()
        .or_else(|| {
            request
                .candidate_pressure
                .as_deref()
                .and_then(|p| pressure_error(p, grid.cell_len()))
        })
        .or_else(|| {
            if request.candidate_pressure.is_none() {
                Some("prediction_unavailable")
            } else {
                None
            }
        });
    let raw = if let Some(reason) = candidate_error {
        json!({"qualified":false,"failures":[reason]})
    } else {
        quality(
            &op,
            &request.pre_velocity,
            &request.original_rhs,
            request.candidate_pressure.as_deref().unwrap(),
            request.dt_s,
        )?
    };
    let learned = corrected(
        &op,
        request,
        request.candidate_pressure.as_deref(),
        &raw,
        &temporal_quality,
        candidate_error,
    )?;
    let mut temporal = corrected(
        &op,
        request,
        Some(&request.previous_pressure),
        &temporal_quality,
        &temporal_quality,
        None,
    )?;
    temporal["comparison_role"] = json!(
        "bounded32-step temporal diagnostic with one p_base reset; not the full-budget temporal baseline"
    );
    let temporal_baseline = temporal_full_budget(&op, request)?;
    let zero_jacobi = solve(&op, request, None, 3000)?;
    Ok(
        json!({"schema":"rheon-variable-pressure-teacher-diagnostic-v1","scope":"teacher-state development snapshots; no autonomous rollout or acceleration qualification","reference":reference,"raw_learned":raw,"raw_temporal":temporal_quality,"raw_zero":zero_quality,"learned_jacobi":learned,"temporal_jacobi":temporal,"temporal_full_budget":temporal_baseline,"zero_jacobi":zero_jacobi,"native_elapsed_s":start.elapsed().as_secs_f64(),"state_advanced":false,"cost_scope":"native diagnostic includes solver/correction/verification; excludes model inference and generic custody"}),
    )
}
fn oracle() -> Result<Value> {
    let counts = [3, 2, 1];
    let spacing = [0.25, 0.4, 0.7];
    let rho = 1.25;
    let grid = GridGeometry::new(counts, spacing, [0.0; 3])?;
    let op = PressureOperator::new(&grid, rho)?;
    let p = vec![0.0, 1.0, -2.0, 0.5, 3.0, -1.0];
    let mut action = vec![0.0; 6];
    op.apply_full(&p, &mut action)?;
    Ok(
        json!({"schema":"rheon-global-pressure-hand-oracle-v1","counts_xyz":counts,"spacing_m":spacing,"density_kg_m3":rho,"pressure":p,"native_full_action":action,"scope":"tiny anisotropic full action including row zero; not training data"}),
    )
}
fn residual(request: &ResidualRequest) -> Result<Value> {
    ensure!(
        request.schema == "rheon-variable-pressure-native-residual-v1",
        "residual schema"
    );
    let grid = cubic_grid(request.counts_xyz, request.spacing_m, request.density_kg_m3)?;
    let op = PressureOperator::new(&grid, request.density_kg_m3)?;
    ensure!(
        request.original_rhs.len() == grid.cell_len()
            && request.original_rhs.iter().all(|v| v.is_finite()),
        "residual RHS contract"
    );
    ensure!(
        pressure_error(&request.base_pressure, grid.cell_len()).is_none(),
        "residual base contract"
    );
    let mut action = vec![0.0; grid.cell_len()];
    op.apply_full(&request.base_pressure, &mut action)?;
    let residual: Vec<_> = request
        .original_rhs
        .iter()
        .zip(action)
        .map(|(b, a)| b - a)
        .collect();
    ensure!(
        residual.iter().all(|v| v.is_finite()),
        "residual arithmetic"
    );
    Ok(
        json!({"schema":"rheon-variable-pressure-native-residual-v1","residual":residual,"identity":serde_json::from_str::<Value>(IDENTITY)?,"scope":"native full original-RHS residual, including cell zero; no pressure solve"}),
    )
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--identity"] {
        println!("{IDENTITY}");
        return Ok(());
    }
    ensure!(
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into()) == "1",
        "native one thread required"
    );
    let result = if args == ["--oracle"] {
        oracle()?
    } else {
        ensure!(
            args == ["--evaluate"] || args == ["--residual"],
            "usage --identity | --oracle | --residual | --evaluate"
        );
        let mut bytes = Vec::new();
        std::io::stdin().take(1048577).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 1048576, "request exceeds1MiB");
        if args == ["--residual"] {
            residual(&serde_json::from_slice(&bytes)?)?
        } else {
            evaluate(&serde_json::from_slice(&bytes)?)?
        }
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("teacher diagnostic refused: {e:#}");
        std::process::exit(2);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_projection_request() -> Request {
        Request {
            schema: "rheon-variable-pressure-teacher-case-v1".into(),
            counts_xyz: [3, 1, 1],
            spacing_m: [1.0; 3],
            density_kg_m3: 1.0,
            dt_s: 0.5,
            original_rhs: vec![-4.0, 6.0, -2.0],
            reference_pressure: vec![0.0, 4.0, 2.0],
            previous_pressure: vec![0.0; 3],
            candidate_pressure: Some(vec![0.0, 4.0, 2.0]),
            candidate_error: None,
            pre_velocity: [vec![0.0, 2.0, -1.0, 0.0], vec![0.0; 6], vec![0.0; 6]],
        }
    }
    #[test]
    fn cubic_grid_accepts_variable_shapes_and_rejects_stretched_cells() -> Result<()> {
        let grid = cubic_grid([17, 13, 9], [0.0625; 3], 1.25)?;
        assert_eq!(grid.counts(), [17, 13, 9]);
        assert_eq!(
            cubic_grid([257, 2, 1], [0.0625; 3], 1.25)?.counts(),
            [257, 2, 1]
        );
        assert!(cubic_grid([17, 13, 9], [0.0625, 0.125, 0.0625], 1.25).is_err());
        assert!(cubic_grid([0, 13, 9], [0.0625; 3], 1.25).is_err());
        assert!(cubic_grid([128; 3], [0.0625; 3], 1.25).is_err());
        Ok(())
    }
    #[test]
    fn exact_guess_uses_zero_iterations_and_direct_acceptance_needs_all_physics() {
        let request = tiny_projection_request();
        let grid = GridGeometry::new(request.counts_xyz, request.spacing_m, [0.0; 3]).unwrap();
        let op = PressureOperator::new(&grid, request.density_kg_m3).unwrap();
        let p = request.candidate_pressure.as_deref().unwrap();
        let q = quality(
            &op,
            &request.pre_velocity,
            &request.original_rhs,
            p,
            request.dt_s,
        )
        .unwrap();
        assert_eq!(q["qualified"], true);
        let native = solve(&op, &request, Some(p), 32).unwrap();
        assert_eq!(native["iterations"], 0);
        assert_eq!(native["qualified"], true);
        let base = quality(
            &op,
            &request.pre_velocity,
            &request.original_rhs,
            &request.previous_pressure,
            request.dt_s,
        )
        .unwrap();
        let accepted = corrected(&op, &request, Some(p), &q, &base, None).unwrap();
        assert_eq!(accepted["outcome"], "direct_acceptance");
        assert_eq!(accepted["any_pcg"], false);
    }
    #[test]
    fn worse_proposal_skips_warm_attempt_and_uses_exactly_one_qualified_temporal_reset() {
        let mut request = tiny_projection_request();
        request.previous_pressure = vec![0.0, 2.0, 1.0];
        let grid = GridGeometry::new(request.counts_xyz, request.spacing_m, [0.0; 3]).unwrap();
        let op = PressureOperator::new(&grid, request.density_kg_m3).unwrap();
        let poor = [0.0, -100.0, 33.0];
        let q = quality(
            &op,
            &request.pre_velocity,
            &request.original_rhs,
            &poor,
            request.dt_s,
        )
        .unwrap();
        let base = quality(
            &op,
            &request.pre_velocity,
            &request.original_rhs,
            &request.previous_pressure,
            request.dt_s,
        )
        .unwrap();
        let accepted = corrected(&op, &request, Some(&poor), &q, &base, None).unwrap();
        assert_eq!(accepted["outcome"], "temporal_reset_fallback");
        assert!(accepted["warm"].is_null());
        assert_eq!(
            accepted["proposal_rejection"],
            "proposal_residual_worse_than_temporal"
        );
        assert_eq!(accepted["reset"]["iteration_limit"], 3000);
        assert_eq!(accepted["qualified"], true);
        assert_eq!(accepted["reset"]["initial_guess_kind"], "supplied_pressure");
        let mut expected = Sha256::new();
        for p in &request.previous_pressure {
            expected.update(p.to_le_bytes());
        }
        assert_eq!(
            accepted["reset"]["initial_guess_sha256"],
            format!("{:x}", expected.finalize())
        );
        let zero = solve(&op, &request, None, 3000).unwrap();
        assert_eq!(zero["initial_guess_kind"], "zero");
        assert_ne!(
            zero["initial_guess_sha256"],
            accepted["reset"]["initial_guess_sha256"]
        );
    }
    #[test]
    fn invalid_guess_shapes_gauge_and_nonfinite_are_observed() {
        assert_eq!(pressure_error(&[0.0], 2), Some("prediction_shape"));
        assert_eq!(
            pressure_error(&[0.0, f64::NAN], 2),
            Some("prediction_nonfinite")
        );
        assert_eq!(pressure_error(&[1.0, 2.0], 2), Some("prediction_gauge"));
        assert_eq!(pressure_error(&[0.0, 2.0], 2), None);
    }
    #[test]
    fn temporal_baseline_is_one_full_budget_attempt_from_nonzero_previous_pressure() {
        let mut request = tiny_projection_request();
        request.previous_pressure = vec![0.0, 2.0, 1.0];
        let grid = GridGeometry::new(request.counts_xyz, request.spacing_m, [0.0; 3]).unwrap();
        let op = PressureOperator::new(&grid, request.density_kg_m3).unwrap();
        let baseline = temporal_full_budget(&op, &request).unwrap();
        assert_eq!(baseline["qualified"], true);
        assert_eq!(baseline["iteration_limit"], 3000);
        assert_eq!(baseline["initial_guess_kind"], "supplied_pressure");
        assert!(baseline.get("warm").is_none() && baseline.get("reset").is_none());
        let mut expected = Sha256::new();
        for p in &request.previous_pressure {
            expected.update(p.to_le_bytes());
        }
        assert_eq!(
            baseline["initial_guess_sha256"],
            format!("{:x}", expected.finalize())
        );
        assert!(baseline["solver_s"].as_f64().unwrap() >= 0.0);
        assert!(baseline["verification_s"].as_f64().unwrap() >= 0.0);
    }
    #[test]
    fn tiny_anisotropic_action_matches_hand_face_balance_including_cell_zero() {
        let a = oracle().unwrap();
        let p = [0.0, 1.0, -2.0, 0.5, 3.0, -1.0];
        let wx = 0.25 * 0.4 * 0.7 / (1.25 * 0.25 * 0.25);
        let wy = 0.25 * 0.4 * 0.7 / (1.25 * 0.4 * 0.4);
        let expected = [
            wx * (p[0] - p[1]) + wy * (p[0] - p[3]),
            wx * (2.0 * p[1] - p[0] - p[2]) + wy * (p[1] - p[4]),
            wx * (p[2] - p[1]) + wy * (p[2] - p[5]),
            wx * (p[3] - p[4]) + wy * (p[3] - p[0]),
            wx * (2.0 * p[4] - p[3] - p[5]) + wy * (p[4] - p[1]),
            wx * (p[5] - p[4]) + wy * (p[5] - p[2]),
        ];
        for (actual, want) in a["native_full_action"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected)
        {
            assert!((actual.as_f64().unwrap() - want).abs() < 1e-14);
        }
        assert_ne!(a["native_full_action"][0], 0.0);
    }
}
