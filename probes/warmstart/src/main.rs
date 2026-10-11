//! Bounded local probe. Driver verifies the complete pinned dataset before timing.
use anyhow::{Result, ensure};
use rheon::{
    Axis, GridGeometry, PressureImplementation, PressureOperator, PressureSettings,
    PressureWorkspace,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

const DT: f64 = 0.03125;
const BASE: &str = "34235d33c886aa05b5bdf95d920b863d1fb2ad6f";
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn bytes(path: &Path, count: usize) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() == count as u64,
        "regular exact-sized array required"
    );
    Ok(fs::read(path)?)
}
fn array(root: &Path, case: &Value, name: &str, count: usize, dtype: &str) -> Result<Vec<u8>> {
    let id = case["spec"]["id"].as_str().unwrap();
    ensure!(
        !id.is_empty()
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "case id"
    );
    let path = format!("{id}/{name}.bin");
    let spec = &case["arrays"][name];
    ensure!(
        spec["path"] == path && spec["dtype"] == dtype && spec["bytes"] == count,
        "array metadata"
    );
    let raw = bytes(&root.join(path), count)?;
    ensure!(spec["sha256"] == hash(&raw), "array hash");
    Ok(raw)
}
fn f64s(raw: &[u8]) -> Vec<f64> {
    raw.chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn refs(v: &[Vec<f32>; 3]) -> [&[f32]; 3] {
    [&v[0], &v[1], &v[2]]
}
fn identity() -> Value {
    json!({"lab_commit":env!("LAB_COMMIT"), "lab_dirty":env!("LAB_DIRTY"),
        "patch_commit":env!("PATCH_COMMIT"), "patch_dirty":env!("PATCH_DIRTY"), "rheon_base":BASE})
}
fn main() -> Result<()> {
    let start = Instant::now();
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--identity" {
        println!("{}", identity());
        return Ok(());
    }
    ensure!(
        (4..=6).contains(&args.len()),
        "usage: probe DATASET METHOD|raw zero|PREDICTIONS [development|final] [quality]"
    );
    let scope = args.get(4).map(String::as_str).unwrap_or("development");
    ensure!(["development", "final"].contains(&scope), "scope");
    let quality = args.get(5).is_some_and(|s| s == "quality");
    ensure!(args.len() != 6 || quality, "quality flag");
    let raw_only = args[2] == "raw";
    let root = Path::new(&args[1]);
    let method = if raw_only {
        PressureImplementation::JacobiPcgV1
    } else {
        PressureImplementation::from_id(&args[2]).ok_or_else(|| anyhow::anyhow!("method"))?
    };
    let warm = args[3] != "zero";
    ensure!(!raw_only || warm, "raw requires a pressure directory");
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    ensure!(
        manifest["schema"]
            == if scope == "final" {
                "rheon-pressure-final-v1"
            } else {
                "rheon-pressure-dataset-v1"
            }
            && manifest["status"] == "complete"
            && manifest["rheon_commit"] == BASE,
        "manifest identity"
    );
    let contract = &manifest["contract"];
    ensure!(
        contract["grid"]["counts_xyz"] == json!([8, 7, 6])
            && contract["grid"]["spacing"] == json!([0.25, 0.3, 0.4])
            && contract["density"]["value"] == 1.25
            && contract["dt"]["value"] == DT,
        "probe fixed physics"
    );
    let grid = GridGeometry::new([8, 7, 6], [0.25, 0.3, 0.4], [0.0; 3])?;
    let op = PressureOperator::new(&grid, 1.25)?;
    let settings = PressureSettings {
        relative_residual: 1e-10,
        absolute_residual: 1e-12,
        divergence_limit: 1e-9,
        max_iterations: 1000,
    };
    let axes = [Axis::X, Axis::Y, Axis::Z];
    let mut results = Vec::new();
    let mut solve_seconds = 0.0;
    let mut correction_seconds = 0.0;
    let mut qualification_seconds = 0.0;
    for case in manifest["cases"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("cases"))?
    {
        let split = case["spec"]["split"].as_str().unwrap_or("");
        if !(if scope == "final" {
            split == "final_test"
        } else {
            matches!(split, "train" | "validation")
        }) {
            continue;
        }
        let id = case["spec"]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("case id"))?;
        if scope == "final" {
            let expected = 9000 + results.len();
            ensure!(
                id == format!("oscillatory_modes-{expected}")
                    && case["spec"]["seed"] == expected
                    && expected < 9016,
                "final reservation"
            );
        }
        let mut velocity: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for d in 0..3 {
            velocity[d] = f32s(&array(
                root,
                case,
                ["u", "v", "w"][d],
                4 * grid.face_len(axes[d]),
                "<f4",
            )?);
        }
        let rhs = f64s(&array(root, case, "rhs", 8 * grid.cell_len(), "<f8")?);
        let mut product = vec![0.0; grid.cell_len()];
        op.build_rhs(refs(&velocity), DT, &mut product)?;
        ensure!(rhs == product, "RHS differs from preprojection velocity");
        let guess_bytes = if warm {
            bytes(
                &Path::new(&args[3]).join(format!("{id}.bin")),
                8 * grid.cell_len(),
            )?
        } else {
            Vec::new()
        };
        let guess = f64s(&guess_bytes);
        let mut ws = PressureWorkspace::with_implementation(&grid, 48 * grid.cell_len(), method)?;
        let mut corrected = velocity.clone();
        let begin = Instant::now();
        let (pressure, iterations) = if raw_only {
            (guess.clone(), None)
        } else {
            let result = if warm {
                ws.solve_rhs_with_initial_pressure(&op, &rhs, &guess, DT, settings, || false)
            } else {
                ws.solve_rhs(&op, &rhs, DT, settings, || false)
            };
            match result {
                Ok(report) => (ws.pressure().to_vec(), Some(report.iterations)),
                Err(error) if args.len() > 4 => {
                    solve_seconds += begin.elapsed().as_secs_f64();
                    results.push(json!({"id":id,"split":split,"qualified":false,"solver_error":error.to_string()}));
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
        };
        solve_seconds += begin.elapsed().as_secs_f64();
        let begin = Instant::now();
        let [x, y, z] = &mut corrected;
        op.correct_velocity(refs(&velocity), &pressure, DT, [x, y, z])?;
        correction_seconds += begin.elapsed().as_secs_f64();
        let begin = Instant::now();
        op.apply_full(&pressure, &mut product)?;
        let residual: Vec<_> = rhs.iter().zip(&product).map(|(b, ap)| b - ap).collect();
        let norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rmax = residual.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let predicted = DT / grid.cell_volume() * rmax;
        let actual = ws.actual_divergence_max(&op, refs(&corrected))?;
        ensure!(
            pressure.iter().all(|v| v.is_finite()) && pressure[0] == 0.0,
            "pressure finite/gauge"
        );
        let linear_limit = settings
            .absolute_residual
            .max(settings.relative_residual * rhs_norm);
        let qualified = norm.is_finite()
            && norm <= linear_limit
            && predicted.is_finite()
            && predicted <= settings.divergence_limit
            && actual.is_finite()
            && actual <= 1e-5;
        if args.len() == 4 {
            ensure!(qualified, "physical qualification failed");
        }
        qualification_seconds += begin.elapsed().as_secs_f64();
        let reference_relative_rmse = if quality {
            let target = f64s(&array(root, case, "pressure", 8 * grid.cell_len(), "<f8")?);
            let error = pressure
                .iter()
                .zip(&target)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>();
            let squared = target.iter().map(|p| p * p).sum::<f64>();
            Some((error / squared).sqrt())
        } else {
            None
        };
        results.push(json!({"id":id, "split":split, "iterations":iterations,
            "pressure_sha256":hash(&pressure.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>()),
            "reference_relative_rmse":reference_relative_rmse,
            "linear_gate_ratio":norm/linear_limit,"predicted_gate_ratio":predicted/settings.divergence_limit,"actual_gate_ratio":actual/1e-5,
            "full_residual_l2":norm, "relative_residual_l2":norm/rhs_norm,
            "predicted_divergence_max":predicted, "actual_f32_divergence_max":actual,
            "qualified":qualified, "initial_pressure_sha256":if warm { Some(hash(&guess_bytes)) } else { None }}));
    }
    ensure!(
        !results.is_empty() && (scope != "final" || results.len() == 16),
        "case count"
    );
    let total = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::to_string(
            &json!({"status":"complete", "method":if raw_only { "raw" } else { method.id() }, "warm":warm,"scope":scope,
        "tool":identity(),
        "cases":results, "timing_seconds":{"native_before_output":total, "solve":solve_seconds, "correction":correction_seconds,
            "qualification":qualification_seconds, "input_hash_rhs_assembly_allocation_and_receipt":total-solve_seconds-correction_seconds-qualification_seconds}})
        )?
    );
    Ok(())
}
