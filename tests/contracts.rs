use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace,
};
use rheon_surrogate_lab::{
    dataset::{self, Manifest},
    experiment as ex,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rheon-lab-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!path.exists());
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if self.0.exists() {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
fn fixture() -> Temp {
    let temp = Temp::new();
    dataset::generate(&temp.0, 1, &"a".repeat(40)).unwrap();
    temp
}
fn change_manifest(root: &Path, f: impl FnOnce(&mut Manifest)) {
    let mut manifest = dataset::read_manifest(root).unwrap();
    f(&mut manifest);
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}
fn rejection(root: &Path, expected: &str) {
    let error = format!("{:#}", dataset::verify(root).unwrap_err());
    assert!(error.contains(expected), "expected {expected}, got {error}");
}
fn mutate_pressure(root: &Path, f: impl FnOnce(&mut Vec<f64>)) {
    let manifest = dataset::read_manifest(root).unwrap();
    let array = &manifest.cases[0].arrays["pressure"];
    let bytes = fs::read(root.join(&array.path)).unwrap();
    let mut p: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    f(&mut p);
    let bytes: Vec<u8> = p.iter().flat_map(|p| p.to_le_bytes()).collect();
    fs::write(root.join(&array.path), &bytes).unwrap();
    change_manifest(root, |m| {
        m.cases[0].arrays.get_mut("pressure").unwrap().sha256 = dataset::hash(&bytes)
    });
}

#[test]
fn roundtrip_reverification_and_deterministic_hashes() {
    let a = fixture();
    let b = fixture();
    let verified = dataset::verify(&a.0).unwrap();
    assert_eq!(verified.cases.len(), 6);
    assert_eq!(
        fs::read(a.0.join("manifest.json")).unwrap(),
        fs::read(b.0.join("manifest.json")).unwrap()
    );
    for case in verified.cases {
        for array in case.arrays.values() {
            assert_eq!(
                fs::read(a.0.join(&array.path)).unwrap(),
                fs::read(b.0.join(&array.path)).unwrap()
            );
        }
    }
    assert!(
        dataset::generate(&a.0, 1, &"a".repeat(40))
            .unwrap_err()
            .to_string()
            .starts_with("destination:")
    );
}

#[test]
fn byte_tamper_is_a_hash_error() {
    let t = fixture();
    let m = dataset::read_manifest(&t.0).unwrap();
    let path = t.0.join(&m.cases[0].arrays["pressure"].path);
    let mut bytes = fs::read(&path).unwrap();
    bytes[17] ^= 1;
    fs::write(path, bytes).unwrap();
    rejection(&t.0, "hash: mismatch");
}
#[test]
fn rehashed_bad_pressure_fails_full_residual() {
    let t = fixture();
    mutate_pressure(&t.0, |p| p[100] += 1.0);
    rejection(&t.0, "residual: full b-Ap");
}
#[test]
fn rehashed_constant_shift_fails_gauge() {
    let t = fixture();
    mutate_pressure(&t.0, |p| p.iter_mut().for_each(|x| *x += 1.0));
    rejection(&t.0, "gauge:");
}
#[test]
fn rehashed_nan_fails_finiteness() {
    let t = fixture();
    mutate_pressure(&t.0, |p| p[100] = f64::NAN);
    rejection(&t.0, "finite:");
}
#[test]
fn truncated_array_fails_shape() {
    let t = fixture();
    let m = dataset::read_manifest(&t.0).unwrap();
    let path = t.0.join(&m.cases[0].arrays["pressure"].path);
    fs::write(path, [0; 8]).unwrap();
    rejection(&t.0, "shape:");
}
#[test]
fn altered_array_metadata_and_path_fail_before_read() {
    let t = fixture();
    change_manifest(&t.0, |m| {
        m.cases[0].arrays.get_mut("pressure").unwrap().path = "../../outside".into()
    });
    rejection(&t.0, "array_metadata:");
}
#[test]
fn incomplete_output_and_missing_case_fail() {
    let t = fixture();
    change_manifest(&t.0, |m| m.status = "incomplete".into());
    rejection(&t.0, "completion:");
    change_manifest(&t.0, |m| {
        m.status = "complete".into();
        m.cases.pop();
    });
    rejection(&t.0, "case_count:");
}
#[test]
fn changed_split_and_weakened_limits_fail() {
    let t = fixture();
    change_manifest(&t.0, |m| m.cases[0].spec.split = "test".into());
    rejection(&t.0, "split_plan:");
    change_manifest(&t.0, |m| {
        m.cases[0].spec.split = "train".into();
        m.contract["solver"]["relative_residual"] = 100.0.into();
    });
    rejection(&t.0, "contract:");
}
#[test]
fn fabricated_metric_or_iteration_receipts_fail() {
    let t = fixture();
    change_manifest(&t.0, |m| m.cases[0].iterations += 1);
    rejection(&t.0, "solver: replay differs");
    change_manifest(&t.0, |m| {
        m.cases[0].iterations -= 1;
        m.cases[0].metrics.corrected_f32_divergence_max = 0.5;
    });
    rejection(&t.0, "metrics:");
}
#[test]
fn wall_violation_and_changed_rhs_fail_at_distinct_boundaries() {
    let spec = &ex::case_plan(1).unwrap()[0];
    let q = ex::solve(spec, PressureImplementation::JacobiPcgV1).unwrap();
    let mut v = q.velocity.clone();
    v[0][0] = 1.0;
    assert!(
        format!("{:#}", ex::qualify(&v, &q.rhs, &q.pressure).unwrap_err())
            .contains("fixed wall velocity")
    );
    let mut b = q.rhs.clone();
    b[20] += 1.0;
    assert!(
        format!(
            "{:#}",
            ex::qualify(&q.velocity, &b, &q.pressure).unwrap_err()
        )
        .contains("rhs: does not match")
    );
}
#[test]
fn genuine_3d_fields_and_disjoint_families_seeds() {
    let plan = ex::case_plan(4).unwrap();
    for a in &plan {
        for b in &plan {
            if a.split != b.split {
                assert_ne!(a.seed, b.seed);
                assert_ne!(a.family, b.family);
            }
        }
    }
    for spec in plan.iter().filter(|c| c.split != "check") {
        let v = ex::velocity(spec).unwrap();
        for (d, field) in v.iter().enumerate() {
            assert!(field.iter().any(|v| *v != 0.0));
            let g = ex::grid().unwrap();
            let shape = g.face_counts([Axis::X, Axis::Y, Axis::Z][d]);
            for axis in 0..3 {
                assert!(
                    varies_on_axis(field, shape, axis),
                    "{} component {d} invariant on axis {axis}",
                    spec.family
                );
            }
        }
    }
}

#[test]
fn exact_3d_manufacture_uses_independent_stencil_and_gradient() {
    // Independent test oracle only, not a second solver or production operator.
    let n: [usize; 3] = [4, 3, 2];
    let h: [f64; 3] = [0.25, 0.5, 0.125];
    let rho = 2.0;
    let dt = 0.125;
    let g = GridGeometry::new(n.map(|n| n as u64), h, [0.0; 3]).unwrap();
    let op = PressureOperator::new(&g, rho).unwrap();
    let p: Vec<f64> = (0..g.cell_len())
        .map(|i| (i % n[0] + 2 * ((i / n[0]) % n[1]) + 4 * (i / (n[0] * n[1]))) as f64)
        .collect();
    let volume: f64 = h.iter().product();
    let strides = [1, n[0], n[0] * n[1]];
    let mut b = vec![0.0; p.len()];
    for (i, value) in b.iter_mut().enumerate() {
        let pos = [i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1])];
        for d in 0..3 {
            let w = volume / (rho * h[d] * h[d]);
            if pos[d] > 0 {
                *value += w * (p[i] - p[i - strides[d]]);
            }
            if pos[d] + 1 < n[d] {
                *value += w * (p[i] - p[i + strides[d]]);
            }
        }
    }
    let mut v: ex::Velocity =
        std::array::from_fn(|d| vec![0.0; g.face_len([Axis::X, Axis::Y, Axis::Z][d])]);
    for (d, field) in v.iter_mut().enumerate() {
        let shape = g.face_counts([Axis::X, Axis::Y, Axis::Z][d]);
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let pos = [i, j, k];
                    if pos[d] > 0 && pos[d] < n[d] {
                        field[g.face_index([Axis::X, Axis::Y, Axis::Z][d], pos).unwrap()] =
                            (dt * [1.0, 2.0, 4.0][d] / (rho * h[d])) as f32;
                    }
                }
            }
        }
    }
    let mut actual_b = vec![0.0; p.len()];
    op.build_rhs(ex::refs(&v), dt, &mut actual_b).unwrap();
    assert_eq!(actual_b, b);
    for method in PressureImplementation::ALL {
        let mut ws = PressureWorkspace::with_implementation(&g, g.cell_len() * 48, method).unwrap();
        ws.solve_rhs(
            &op,
            &b,
            dt,
            PressureSettings {
                relative_residual: 1e-12,
                absolute_residual: 1e-12,
                divergence_limit: 1e-10,
                max_iterations: 1000,
            },
            || false,
        )
        .unwrap();
        assert_eq!(ws.pressure()[0], 0.0);
        assert!(
            ws.pressure()
                .iter()
                .zip(&p)
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        let mut corrected = v.clone();
        let [x, y, z] = &mut corrected;
        op.correct_velocity(ex::refs(&v), ws.pressure(), dt, [x, y, z])
            .unwrap();
        assert!(corrected.iter().flatten().all(|v| v.abs() < 1e-9));
    }
}

#[test]
fn upstream_failure_contract_and_zero_case() {
    let g = ex::grid().unwrap();
    let op = PressureOperator::new(&g, ex::DENSITY).unwrap();
    let mut ws = PressureWorkspace::new(&g, g.cell_len() * 48).unwrap();
    let mut b = vec![0.0; g.cell_len()];
    b[1] = 1.0;
    assert!(matches!(
        ws.solve_rhs(&op, &b, ex::DT, ex::settings(), || false),
        Err(rheon::PressureError::IncompatibleRhs { .. })
    ));
    let q = ex::solve(
        &ex::case_plan(1).unwrap()[0],
        PressureImplementation::JacobiPcgV1,
    )
    .unwrap();
    let mut limits = ex::settings();
    limits.max_iterations = 0;
    assert!(matches!(
        ws.solve_rhs(&op, &q.rhs, ex::DT, limits, || false),
        Err(rheon::PressureError::IterationLimit { iterations: 0, .. })
    ));
    let zero = ex::case_plan(1)
        .unwrap()
        .into_iter()
        .find(|c| c.family == "zero")
        .unwrap();
    let q = ex::solve(&zero, PressureImplementation::JacobiPcgV1).unwrap();
    assert_eq!(q.iterations, 0);
    assert_eq!(q.metrics.corrected_f32_divergence_max, 0.0);
}

#[test]
fn raw_prediction_evaluator_distinguishes_targets_zero_and_predicted_gate() {
    let t = fixture();
    let predictions = Temp::new();
    fs::create_dir(&predictions.0).unwrap();
    let manifest = dataset::read_manifest(&t.0).unwrap();
    for case in manifest.cases.iter().filter(|c| c.spec.split != "check") {
        fs::copy(
            t.0.join(&case.arrays["pressure"].path),
            predictions.0.join(format!("{}.bin", case.spec.id)),
        )
        .unwrap();
    }
    let report = dataset::evaluate_predictions(&t.0, &predictions.0).unwrap();
    assert!(
        report["cases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["reference_qualified"] == true)
    );
    let case = &manifest.cases[0];
    let path = predictions.0.join(format!("{}.bin", case.spec.id));
    let bytes = fs::read(&path).unwrap();
    let mut p: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let g = ex::grid().unwrap();
    let diag = 2.0 * g.cell_volume() / ex::DENSITY
        * ex::SPACING.iter().map(|h| 1.0 / (h * h)).sum::<f64>();
    p[100] += 1.2e-9 * g.cell_volume() / (ex::DT * diag);
    fs::write(
        &path,
        p.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    let report = dataset::evaluate_predictions(&t.0, &predictions.0).unwrap();
    let metrics = &report["cases"][0];
    assert!(
        metrics["full_residual_l2"].as_f64().unwrap()
            <= ex::settings().relative_residual * case.metrics.rhs_l2
    );
    assert!(
        metrics["corrected_f32_divergence_max"].as_f64().unwrap() <= ex::ACTUAL_DIVERGENCE_LIMIT
    );
    assert!(
        metrics["predicted_divergence_max"].as_f64().unwrap() > ex::settings().divergence_limit
    );
    assert_eq!(metrics["reference_qualified"], false);
    fs::write(&path, vec![0; g.cell_len() * 8]).unwrap();
    let report = dataset::evaluate_predictions(&t.0, &predictions.0).unwrap();
    assert_eq!(report["cases"][0]["reference_qualified"], false);
    assert!(report["cases"][0]["pressure_rmse_pa"].as_f64().unwrap() > 0.0);
}

fn varies_on_axis(field: &[f32], shape: [usize; 3], axis: usize) -> bool {
    let stride = [1, shape[0], shape[0] * shape[1]][axis];
    (0..field.len())
        .any(|i| (i / stride) % shape[axis] + 1 < shape[axis] && field[i] != field[i + stride])
}
#[test]
fn variation_oracle_does_not_wrap_rows_or_planes() {
    let shape = [4, 3, 2];
    let yz: Vec<f32> = (0..24).map(|i| ((i / 4) % 3 + i / 12) as f32).collect();
    assert!(!varies_on_axis(&yz, shape, 0));
    assert!(varies_on_axis(&yz, shape, 1));
    let z: Vec<f32> = (0..24).map(|i| (i / 12) as f32).collect();
    assert!(!varies_on_axis(&z, shape, 0));
    assert!(!varies_on_axis(&z, shape, 1));
    assert!(varies_on_axis(&z, shape, 2));
}

#[test]
fn development_baselines_exclude_test_predictions_and_reference_solves_qualify() {
    let data = fixture();
    let output = Temp::new();
    let report = rheon_surrogate_lab::diagnostics::baselines(&data.0, &output.0).unwrap();
    assert_eq!(report["scope"], "train_validation_only");
    assert_eq!(report["methods"].as_array().unwrap().len(), 5);
    for method in report["methods"].as_array().unwrap() {
        let rows = method["evaluation"]["cases"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .all(|c| c["split"] != "test" && c["split"] != "check")
        );
        assert_eq!(
            fs::read_dir(output.0.join(method["method"].as_str().unwrap()))
                .unwrap()
                .count(),
            3
        );
        if ["jacobi_pcg", "sgs_pcg"].contains(&method["method"].as_str().unwrap()) {
            assert!(rows.iter().all(|c| c["reference_qualified"] == true));
        }
    }
    assert_eq!(report["train_mean_fit_ids"].as_array().unwrap().len(), 2);
}

#[test]
fn nonlocal_witness_changes_remote_pressure_despite_identical_local_inputs() {
    let witness = rheon_surrogate_lab::diagnostics::nonlocal_witness().unwrap();
    assert!(
        witness["reference_pressure_at_observation_pa"]
            .as_f64()
            .unwrap()
            .abs()
            > 1e-5
    );
    let b: Vec<f64> = witness["rhs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(b.iter().filter(|b| **b != 0.0).count(), 2);
    assert_eq!(b.iter().sum::<f64>(), 0.0);
    for center in [[0_i32, 0, 0], [3, 3, 2]] {
        for z in (center[2] - 1)..=(center[2] + 1) {
            for y in (center[1] - 1)..=(center[1] + 1) {
                for x in (center[0] - 1)..=(center[0] + 1) {
                    if x >= 0 && y >= 0 && z >= 0 {
                        let index =
                            x as usize + ex::COUNTS[0] * (y as usize + ex::COUNTS[1] * z as usize);
                        assert_eq!(b[index], 0.0);
                    }
                }
            }
        }
    }
    assert!(
        witness["metrics"]["corrected_f32_divergence_max"]
            .as_f64()
            .unwrap()
            <= ex::ACTUAL_DIVERGENCE_LIMIT
    );
}

#[test]
fn exported_operator_matches_public_action_including_gauge_row() {
    let output = Temp::new();
    let metadata = rheon_surrogate_lab::diagnostics::export_operator(&output.0).unwrap();
    let bytes = fs::read(output.0.join("operator.bin")).unwrap();
    assert_eq!(dataset::hash(&bytes), metadata["sha256"]);
    let a: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let g = ex::grid().unwrap();
    let n = g.cell_len();
    assert_eq!(a.len(), n * n);
    let p: Vec<f64> = (0..n).map(|i| ((i * 17) % 31) as f64 / 31.0).collect();
    let mut actual = vec![0.0; n];
    PressureOperator::new(&g, ex::DENSITY)
        .unwrap()
        .apply_full(&p, &mut actual)
        .unwrap();
    for row in 0..n {
        let exported = a[row * n..(row + 1) * n]
            .iter()
            .zip(&p)
            .map(|(a, p)| a * p)
            .sum::<f64>();
        assert!((exported - actual[row]).abs() < 1e-14);
    }
    assert_ne!(actual[0], 0.0);
}

#[test]
fn final_reservation_metadata_is_exact_and_development_cannot_masquerade_as_final() {
    let plan = dataset::final_case_plan();
    assert_eq!(plan.len(), 16);
    for (i, case) in plan.iter().enumerate() {
        assert_eq!(case.seed, 9000 + i as u64);
        assert_eq!(case.family, "oscillatory_modes");
        assert_eq!(case.split, "final_test");
    }
    // No reserved velocity/RHS/target is constructed by this test.
    let data = fixture();
    change_manifest(&data.0, |m| {
        m.schema = dataset::FINAL_SCHEMA.into();
    });
    rejection(&data.0, "freeze:");
    change_manifest(&data.0, |m| {
        m.freeze_sha256 = Some("b".repeat(64));
        m.per_family = 16;
    });
    rejection(&data.0, "case_count:");
}
