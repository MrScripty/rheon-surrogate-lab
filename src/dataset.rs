use crate::experiment::{self as ex, CaseSpec, Metrics};
use anyhow::{Context, Result, ensure};
use rheon::{Axis, PressureImplementation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub const SCHEMA: &str = "rheon-pressure-dataset-v1";
pub const FINAL_SCHEMA: &str = "rheon-pressure-final-v1";

/// Metadata only: this does not materialize the reserved velocity/RHS/targets.
pub fn final_case_plan() -> Vec<CaseSpec> {
    (9000..9016)
        .map(|seed| {
            let id = format!("oscillatory_modes-{seed}");
            CaseSpec {
                id: id.clone(),
                family: "oscillatory_modes".into(),
                seed,
                split: "final_test".into(),
                trajectory: id,
            }
        })
        .collect()
}
const MANIFEST_LIMIT: u64 = 2_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Array {
    pub path: String,
    pub dtype: String,
    pub shape_xyz: [usize; 3],
    pub offset_cells: [f64; 3],
    pub units: String,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub spec: CaseSpec,
    pub status: String,
    pub iterations: usize,
    pub metrics: Metrics,
    pub arrays: BTreeMap<String, Array>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub status: String,
    pub experiment_commit: String,
    pub rheon_commit: String,
    pub per_family: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freeze_sha256: Option<String>,
    pub contract: Value,
    pub cases: Vec<Case>,
}

pub fn contract() -> Value {
    let settings = ex::settings();
    json!({
        "generator": "splitmix64-mac-families-v1",
        "grid": {"counts_xyz": ex::COUNTS, "axes": ["x", "y", "z"],
            "spacing": ex::SPACING, "origin": [0.0,0.0,0.0], "length_units": "m",
            "storage_order": "X-fastest: i + nx*(j + ny*k)", "tensor_order": "N,C,Z,Y,X"},
        "boundaries": "sealed impermeable: boundary normal faces exactly zero; no tangential no-slip constraint",
        "density": {"value": ex::DENSITY, "units": "kg/m^3"},
        "dt": {"value": ex::DT, "units": "s"},
        "operator": "Rheon PressureOperator::new; A=B diag(area/(density*distance)) B^T; b=-V*div(u*)/dt",
        "gauge": "pressure[0]=0; full residual includes cell 0; RHS never mean-shifted",
        "solver": {"id": "jacobi-pcg-v1", "initial_pressure": "zero",
            "relative_residual": settings.relative_residual, "absolute_residual": settings.absolute_residual,
            "predicted_divergence_limit": settings.divergence_limit,
            "actual_f32_divergence_limit": ex::ACTUAL_DIVERGENCE_LIMIT,
            "max_iterations": settings.max_iterations,
            "linear_acceptance": "L2(b-Ap) <= max(abs,rel*L2(original b))",
            "compatibility": "abs(sum(b)) <= 64*f64_epsilon*sum(abs(b))"},
        "split_policy": "whole seeds and whole scenario families; standalone fields, no frames; check cases excluded from training",
        "normalization": "none in export; fit training statistics only in model consumer",
        "hash": "SHA-256 of little-endian raw bytes; integrity, not authentication",
        "reproducibility": "byte determinism tested on the same toolchain/platform; libm differences may change oscillatory family across platforms"
    })
}

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn array_spec(case: &str, name: &str) -> Result<Array> {
    let g = ex::grid()?;
    let (dtype, shape, offset, units) = match name {
        "rhs" => ("<f8", ex::COUNTS, [0.5; 3], "m^3/s^2"),
        "pressure" => ("<f8", ex::COUNTS, [0.5; 3], "Pa"),
        name => {
            let d = match name {
                "u" | "corrected_u" => 0,
                "v" | "corrected_v" => 1,
                "w" | "corrected_w" => 2,
                _ => anyhow::bail!("array: unknown name"),
            };
            let mut offset = [0.5; 3];
            offset[d] = 0.0;
            (
                "<f4",
                g.face_counts([Axis::X, Axis::Y, Axis::Z][d]),
                offset,
                "m/s",
            )
        }
    };
    Ok(Array {
        path: format!("{case}/{name}.bin"),
        dtype: dtype.into(),
        shape_xyz: shape,
        offset_cells: offset,
        units: units.into(),
        bytes: shape.iter().product::<usize>() * if dtype == "<f8" { 8 } else { 4 },
        sha256: String::new(),
    })
}

fn write_array(root: &Path, case: &str, name: &str, bytes: Vec<u8>) -> Result<Array> {
    let mut spec = array_spec(case, name)?;
    ensure!(bytes.len() == spec.bytes, "shape: write array");
    fs::write(root.join(&spec.path), &bytes)?;
    spec.sha256 = hash(&bytes);
    Ok(spec)
}

/// Refuses existing destinations; an interrupted run remains explicitly incomplete.
pub fn generate(root: &Path, per_family: usize, experiment_commit: &str) -> Result<Manifest> {
    generate_plan(root, per_family, experiment_commit, None)
}

/// Call only after the evaluator/checkpoint/protocol freeze is committed.
/// CLI access is through the freeze-checking Python gate, never an arbitrary seed.
pub fn generate_final(
    root: &Path,
    experiment_commit: &str,
    freeze_sha256: &str,
) -> Result<Manifest> {
    ensure!(
        freeze_sha256.len() == 64 && freeze_sha256.bytes().all(|v| v.is_ascii_hexdigit()),
        "freeze: invalid hash"
    );
    generate_plan(root, 16, experiment_commit, Some(freeze_sha256.to_owned()))
}

fn generate_plan(
    root: &Path,
    per_family: usize,
    experiment_commit: &str,
    freeze_sha256: Option<String>,
) -> Result<Manifest> {
    let plan = if freeze_sha256.is_some() {
        final_case_plan()
    } else {
        ex::case_plan(per_family)?
    };
    ensure!(
        valid_commit(experiment_commit),
        "provenance: need full experiment commit"
    );
    ensure!(!root.exists(), "destination: already exists");
    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(root)?;
    let mut manifest = Manifest {
        schema: if freeze_sha256.is_some() {
            FINAL_SCHEMA.into()
        } else {
            SCHEMA.into()
        },
        status: "incomplete".into(),
        experiment_commit: experiment_commit.into(),
        rheon_commit: ex::RHEON_COMMIT.into(),
        per_family,
        freeze_sha256,
        contract: contract(),
        cases: Vec::new(),
    };
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    for spec in plan {
        let q = ex::solve(&spec, PressureImplementation::JacobiPcgV1)
            .with_context(|| format!("case {}", spec.id))?;
        fs::create_dir(root.join(&spec.id))?;
        let mut arrays = BTreeMap::new();
        for (name, values) in [
            ("u", &q.velocity[0]),
            ("v", &q.velocity[1]),
            ("w", &q.velocity[2]),
            ("corrected_u", &q.corrected[0]),
            ("corrected_v", &q.corrected[1]),
            ("corrected_w", &q.corrected[2]),
        ] {
            arrays.insert(
                name.into(),
                write_array(
                    root,
                    &spec.id,
                    name,
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )?,
            );
        }
        for (name, values) in [("rhs", &q.rhs), ("pressure", &q.pressure)] {
            arrays.insert(
                name.into(),
                write_array(
                    root,
                    &spec.id,
                    name,
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )?,
            );
        }
        manifest.cases.push(Case {
            spec,
            status: "qualified".into(),
            iterations: q.iterations,
            metrics: q.metrics,
            arrays,
        });
    }
    manifest.status = "complete".into();
    fs::write(
        root.join("manifest.pending"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::rename(root.join("manifest.pending"), root.join("manifest.json"))?;
    verify(root)?;
    Ok(manifest)
}

fn valid_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn read_regular(path: &Path, expected: u64) -> Result<Vec<u8>> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() == expected,
        "shape: wrong file type or byte length for {}",
        path.display()
    );
    Ok(fs::read(path)?)
}

fn read_array(root: &Path, case: &Case, name: &str) -> Result<Vec<u8>> {
    let array = case.arrays.get(name).context("array: missing")?;
    let expected = array_spec(&case.spec.id, name)?;
    ensure!(
        array.path == expected.path
            && array.dtype == expected.dtype
            && array.shape_xyz == expected.shape_xyz
            && array.offset_cells == expected.offset_cells
            && array.units == expected.units
            && array.bytes == expected.bytes,
        "array_metadata: invalid {name}"
    );
    ensure!(
        fs::symlink_metadata(root.join(&case.spec.id))?
            .file_type()
            .is_dir(),
        "path: case directory must not be symlink"
    );
    let bytes = read_regular(&root.join(&array.path), expected.bytes as u64)?;
    ensure!(
        hash(&bytes) == array.sha256,
        "hash: mismatch for {}",
        array.path
    );
    Ok(bytes)
}

fn read_f32(root: &Path, case: &Case, name: &str) -> Result<Vec<f32>> {
    Ok(read_array(root, case, name)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
pub(crate) fn read_f64(root: &Path, case: &Case, name: &str) -> Result<Vec<f64>> {
    Ok(read_array(root, case, name)?
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

pub fn read_manifest(root: &Path) -> Result<Manifest> {
    let path = root.join("manifest.json");
    let n = fs::symlink_metadata(&path)?.len();
    ensure!(n <= MANIFEST_LIMIT, "manifest: exceeds size limit");
    Ok(serde_json::from_slice(&read_regular(&path, n)?)?)
}

/// Rebuild RHS, full residual, and rounded velocity from persisted data.
/// Fixed contract comparison prevents a manifest from weakening its own gates.
pub fn verify(root: &Path) -> Result<Manifest> {
    let manifest = read_manifest(root)?;
    ensure!(
        [SCHEMA, FINAL_SCHEMA].contains(&manifest.schema.as_str()),
        "schema: unsupported"
    );
    if manifest.schema == FINAL_SCHEMA {
        ensure!(
            manifest.per_family == 16
                && manifest
                    .freeze_sha256
                    .as_ref()
                    .is_some_and(|s| s.len() == 64 && s.bytes().all(|v| v.is_ascii_hexdigit())),
            "freeze: final dataset needs a fixed reservation and freeze hash"
        );
    } else {
        ensure!(
            manifest.freeze_sha256.is_none(),
            "freeze: unexpected in development schema"
        );
    }
    ensure!(
        manifest.status == "complete",
        "completion: dataset is incomplete"
    );
    ensure!(
        valid_commit(&manifest.experiment_commit) && manifest.rheon_commit == ex::RHEON_COMMIT,
        "provenance: invalid commit"
    );
    ensure!(
        manifest.contract == contract(),
        "contract: unsupported or modified"
    );
    let plan = if manifest.schema == FINAL_SCHEMA {
        final_case_plan()
    } else {
        ex::case_plan(manifest.per_family)?
    };
    ensure!(
        manifest.cases.len() == plan.len(),
        "case_count: incomplete plan"
    );
    for (case, expected) in manifest.cases.iter().zip(plan) {
        ensure!(
            case.spec == expected,
            "split_plan: seed/family/split/trajectory mismatch"
        );
        ensure!(
            case.status == "qualified",
            "completion: case is unqualified"
        );
        ensure!(case.arrays.len() == 8, "array: expected eight arrays");
        let v = [
            read_f32(root, case, "u")?,
            read_f32(root, case, "v")?,
            read_f32(root, case, "w")?,
        ];
        let b = read_f64(root, case, "rhs")?;
        let p = read_f64(root, case, "pressure")?;
        let (metrics, corrected) =
            ex::qualify(&v, &b, &p).with_context(|| format!("qualification {}", case.spec.id))?;
        ensure!(
            v == ex::velocity(&case.spec)?,
            "generator: seed/family replay mismatch"
        );
        for (d, name) in ["corrected_u", "corrected_v", "corrected_w"]
            .iter()
            .enumerate()
        {
            ensure!(
                corrected[d] == read_f32(root, case, name)?,
                "correction: stored f32 velocity mismatch"
            );
        }
        ensure!(
            case.metrics == metrics,
            "metrics: recorded diagnostics differ from recomputation"
        );
        // A small fresh solve also checks solver identity, iteration receipt and target provenance.
        let replay = ex::solve(&case.spec, PressureImplementation::JacobiPcgV1)?;
        ensure!(
            case.iterations == replay.iterations && p == replay.pressure,
            "solver: replay differs"
        );
    }
    Ok(manifest)
}

pub fn summary(manifest: &Manifest) -> Value {
    let mut splits = BTreeMap::<String, usize>::new();
    for case in &manifest.cases {
        *splits.entry(case.spec.split.clone()).or_default() += 1;
    }
    json!({"status": "verified", "schema":manifest.schema, "freeze_sha256":manifest.freeze_sha256, "cases": manifest.cases.len(), "splits": splits,
        "experiment_commit": manifest.experiment_commit, "rheon_commit": manifest.rheon_commit,
        "max_full_residual_l2": manifest.cases.iter().map(|c| c.metrics.full_residual_l2).fold(0.0, f64::max),
        "max_corrected_f32_divergence": manifest.cases.iter().map(|c| c.metrics.corrected_f32_divergence_max).fold(0.0, f64::max),
        "max_iterations": manifest.cases.iter().map(|c| c.iterations).max().unwrap_or(0)})
}

/// Measure raw model predictions with Rheon, without granting reference qualification.
pub fn evaluate_predictions(root: &Path, predictions: &Path) -> Result<Value> {
    evaluate_predictions_scoped(root, predictions, true)
}

/// Development evaluation never opens test prediction files or reports test metrics.
pub fn evaluate_predictions_scoped(
    root: &Path,
    predictions: &Path,
    include_test: bool,
) -> Result<Value> {
    let manifest = verify(root)?;
    let g = ex::grid()?;
    let op = rheon::PressureOperator::new(&g, ex::DENSITY)?;
    let mut cases = Vec::new();
    for case in &manifest.cases {
        if case.spec.split == "check"
            || (!include_test && !matches!(case.spec.split.as_str(), "train" | "validation"))
        {
            continue;
        }
        let bytes = read_regular(
            &predictions.join(format!("{}.bin", case.spec.id)),
            (g.cell_len() * 8) as u64,
        )?;
        let p: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            p.iter().all(|v| v.is_finite()) && p[0] == 0.0,
            "prediction: finite values and cell-zero gauge required"
        );
        let b = read_f64(root, case, "rhs")?;
        let target = read_f64(root, case, "pressure")?;
        let mut ap = vec![0.0; g.cell_len()];
        op.apply_full(&p, &mut ap)?;
        let residual: Vec<f64> = b.iter().zip(ap).map(|(b, a)| b - a).collect();
        let residual_l2 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        let residual_max = residual.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let predicted = residual_max * ex::DT / g.cell_volume();
        let v = [
            read_f32(root, case, "u")?,
            read_f32(root, case, "v")?,
            read_f32(root, case, "w")?,
        ];
        let mut corrected = v.clone();
        let [x, y, z] = &mut corrected;
        op.correct_velocity(ex::refs(&v), &p, ex::DT, [x, y, z])?;
        let mut div = vec![0.0; g.cell_len()];
        op.divergence(ex::refs(&corrected), &mut div)?;
        let divergence = div.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let target_rms = (target.iter().map(|v| v * v).sum::<f64>() / target.len() as f64).sqrt();
        let rmse = (p
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / p.len() as f64)
            .sqrt();
        let error: Vec<f64> = p.iter().zip(&target).map(|(p, t)| p - t).collect();
        let mean_error = error.iter().sum::<f64>() / error.len() as f64;
        let centered_rmse = (error.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>()
            / error.len() as f64)
            .sqrt();
        let linear_limit = ex::settings()
            .absolute_residual
            .max(ex::settings().relative_residual * case.metrics.rhs_l2);
        ensure!(
            [
                residual_l2,
                residual_max,
                predicted,
                divergence,
                rmse,
                target_rms
            ]
            .iter()
            .all(|x| x.is_finite()),
            "prediction: nonfinite metrics"
        );
        cases.push(json!({"id": case.spec.id, "split": case.spec.split, "pressure_rmse_pa": rmse,
            "pressure_error_mean_pa": mean_error, "pressure_centered_rmse_pa": centered_rmse,
            "relative_pressure_rmse": if target_rms > 0.0 {Some(rmse/target_rms)} else {None},
            "relative_residual_l2": if case.metrics.rhs_l2 > 0.0 {Some(residual_l2/case.metrics.rhs_l2)} else {None},
            "linear_gate_ratio": residual_l2/linear_limit,
            "predicted_divergence_gate_ratio": predicted/ex::settings().divergence_limit,
            "actual_divergence_gate_ratio": divergence/ex::ACTUAL_DIVERGENCE_LIMIT,
            "full_residual_l2": residual_l2, "full_residual_max": residual_max,
            "predicted_divergence_max": predicted, "corrected_f32_divergence_max": divergence,
            "reference_qualified": residual_l2 <= ex::settings().absolute_residual.max(ex::settings().relative_residual*case.metrics.rhs_l2)
                && predicted <= ex::settings().divergence_limit && divergence <= ex::ACTUAL_DIVERGENCE_LIMIT}));
    }
    Ok(
        json!({"status": "raw_predictions_measured", "warm_start": "unavailable in pinned Rheon",
        "scope": if include_test {"all_noncheck"} else {"train_validation_only"},
        "measurement": "no acceleration or matched-tolerance timing claim", "cases": cases}),
    )
}
