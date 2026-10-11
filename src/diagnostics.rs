//! Fixed, bounded development baselines. No fitted solver, warm start, or timing claim.
use crate::{dataset, experiment as ex};
use anyhow::{Result, ensure};
use rheon::{Axis, PressureImplementation, PressureOperator, PressureWorkspace};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};

/// Extract diagonal through the pinned public operator, not a duplicated stencil.
pub fn operator_diagonal() -> Result<Vec<f64>> {
    let g = ex::grid()?;
    let op = PressureOperator::new(&g, ex::DENSITY)?;
    let mut basis = vec![0.0; g.cell_len()];
    let mut product = basis.clone();
    let mut diagonal = basis.clone();
    for row in 0..basis.len() {
        basis[row] = 1.0;
        op.apply_full(&basis, &mut product)?;
        diagonal[row] = product[row];
        basis[row] = 0.0;
    }
    ensure!(
        diagonal.iter().all(|d| d.is_finite() && *d > 0.0),
        "diagonal: invalid"
    );
    Ok(diagonal)
}

/// Pressure at a distant observation cell responds to a boundary-valid single face.
/// A radius-one local CNN cannot see either changed RHS cell at that observation
/// or at its gauge anchor; the model output there is therefore exactly unchanged.
pub fn nonlocal_witness() -> Result<Value> {
    let g = ex::grid()?;
    let op = PressureOperator::new(&g, ex::DENSITY)?;
    let mut velocity: ex::Velocity =
        std::array::from_fn(|d| vec![0.0; g.face_len([Axis::X, Axis::Y, Axis::Z][d])]);
    let face = [7, 5, 4];
    velocity[0][g.face_index(Axis::X, face).unwrap()] = 1.0;
    let mut rhs = vec![0.0; g.cell_len()];
    op.build_rhs(ex::refs(&velocity), ex::DT, &mut rhs)?;
    let mut workspace = PressureWorkspace::new(&g, g.cell_len() * 48)?;
    let report = workspace.solve_rhs(&op, &rhs, ex::DT, ex::settings(), || false)?;
    let p = workspace.pressure();
    let (metrics, _) = ex::qualify(&velocity, &rhs, p)?;
    let observation = [3, 3, 2];
    let index = observation[0] + ex::COUNTS[0] * (observation[1] + ex::COUNTS[1] * observation[2]);
    Ok(
        json!({"purpose":"fixed structural locality witness; not train/validation/test data",
        "counts_xyz":ex::COUNTS,"source_u_face_xyz":face,"source_velocity_m_per_s":1.0,
        "observation_xyz":observation,"observation_index":index,
        "rhs":rhs,"pressure":p,"reference_pressure_at_observation_pa":p[index],
        "iterations":report.iterations,"metrics":metrics}),
    )
}

/// Evaluate five predeclared baselines on train/validation only; no tuning.
pub fn baselines(root: &Path, output: &Path) -> Result<Value> {
    let start = Instant::now();
    let manifest = dataset::verify(root)?;
    ensure!(!output.exists(), "destination: already exists");
    fs::create_dir_all(output)?;
    let diagonal_start = Instant::now();
    let diagonal = operator_diagonal()?;
    let diagonal_seconds = diagonal_start.elapsed().as_secs_f64();
    let mut mean = vec![0.0; diagonal.len()];
    let training: Vec<_> = manifest
        .cases
        .iter()
        .filter(|c| c.spec.split == "train")
        .collect();
    for case in &training {
        for (m, p) in mean
            .iter_mut()
            .zip(dataset::read_f64(root, case, "pressure")?)
        {
            *m += p / training.len() as f64;
        }
    }
    let methods = [
        "zero",
        "train_mean",
        "diagonal_once",
        "jacobi_pcg",
        "sgs_pcg",
    ];
    let mut results = Vec::new();
    for method in methods {
        let directory = output.join(method);
        fs::create_dir(&directory)?;
        let mut solve_receipts = Vec::new();
        for case in manifest
            .cases
            .iter()
            .filter(|c| c.spec.split == "train" || c.spec.split == "validation")
        {
            let mut pressure = match method {
                "zero" => vec![0.0; diagonal.len()],
                "train_mean" => mean.clone(),
                "diagonal_once" => dataset::read_f64(root, case, "rhs")?
                    .iter()
                    .zip(&diagonal)
                    .map(|(b, d)| b / d)
                    .collect(),
                _ => {
                    let implementation = if method == "jacobi_pcg" {
                        PressureImplementation::JacobiPcgV1
                    } else {
                        PressureImplementation::SymmetricGaussSeidelPcgV1
                    };
                    let qualified = ex::solve(&case.spec, implementation)?;
                    solve_receipts
                        .push(json!({"id":case.spec.id,"iterations":qualified.iterations}));
                    qualified.pressure
                }
            };
            let gauge = pressure[0];
            pressure.iter_mut().for_each(|p| *p -= gauge);
            let bytes: Vec<u8> = pressure.iter().flat_map(|p| p.to_le_bytes()).collect();
            fs::write(directory.join(format!("{}.bin", case.spec.id)), bytes)?;
        }
        results.push(json!({"method":method,"solve_receipts":solve_receipts,
            "evaluation":dataset::evaluate_predictions_scoped(root,&directory,false)?}));
    }
    let witness = nonlocal_witness()?;
    fs::write(
        output.join("nonlocal-witness.json"),
        serde_json::to_vec_pretty(&witness)?,
    )?;
    Ok(
        json!({"status":"complete","scope":"train_validation_only","methods":results,
        "train_mean_fit_ids":training.iter().map(|c|&c.spec.id).collect::<Vec<_>>(),
        "diagonal_definition":"one simultaneous p=D^-1 b, then constant gauge shift; D obtained via public apply_full basis probes; no iteration",
        "elapsed_seconds_including_io_verification_and_repeated_reference_solves":start.elapsed().as_secs_f64(),
        "diagonal_setup_seconds":diagonal_seconds,"diagonal_operator_calls":diagonal.len(),
        "timing_limits":"single diagnostic workflow; timings include deliberately repeated verification, not a solver or acceleration benchmark",
        "witness_file":"nonlocal-witness.json"}),
    )
}
