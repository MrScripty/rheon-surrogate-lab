#!/usr/bin/env python3
"""Bounded, no-training diagnostics on train/validation and a structural witness."""
import argparse
import json
from pathlib import Path
import time

import train_cpu as model
import numpy as np


def summary(evaluation):
    result = {}
    for split in ("train", "validation"):
        subset = [c for c in evaluation["cases"] if c["split"] == split]
        result[split] = {
            "cases": len(subset),
            "mean_relative_pressure_rmse": float(np.mean([c["relative_pressure_rmse"] for c in subset])),
            "mean_centered_pressure_rmse_pa": float(np.mean([c["pressure_centered_rmse_pa"] for c in subset])),
            "mean_relative_residual_l2": float(np.mean([c["relative_residual_l2"] for c in subset])),
            "max_corrected_f32_divergence": max(c["corrected_f32_divergence_max"] for c in subset),
            "max_linear_gate_ratio": max(c["linear_gate_ratio"] for c in subset),
            "max_predicted_divergence_gate_ratio": max(c["predicted_divergence_gate_ratio"] for c in subset),
            "qualified": sum(c["reference_qualified"] for c in subset),
        }
    return result


def distribution_stats(cases, norm):
    training_energy = sum(float(np.sum(c["pressure"]**2)) for c in cases if c["split"] == "train")
    stats = {}
    for family in sorted({c["family"] for c in cases}):
        group = [c for c in cases if c["family"] == family]
        rhs = np.stack([c["rhs"] for c in group])
        p = np.stack([c["pressure"] for c in group])
        stats[family] = {
            "split": group[0]["split"], "cases": len(group),
            "rhs_rms": float(np.sqrt(np.mean(rhs**2))),
            "pressure_rms_pa": float(np.sqrt(np.mean(p**2))),
            "rhs_rms_in_training_std_units": float(np.sqrt(np.mean(rhs**2)) / norm["rhs_std"]),
            "fraction_of_zero_prediction_train_squared_error": float(np.sum(p**2) / training_energy) if group[0]["split"] == "train" else None,
        }
    return stats


def predict(params, cases, norm):
    patches, _ = model.encode(cases, norm)
    return model.forward(params, patches)[0] * norm["pressure_rms"]


def model_diagnostics(run, cases, norm, witness, binary, dataset):
    recorded = json.loads((run / "receipt.json").read_text())
    if recorded["normalization"] != norm or model.digest(run / "weights.npz") != recorded["weights_sha256"]:
        raise ValueError("checkpoint or train-only normalization differs from receipt")
    if recorded["dataset_manifest_sha256"] != model.digest(dataset / "manifest.json"):
        raise ValueError("checkpoint belongs to another dataset")
    with np.load(run / "weights.npz", allow_pickle=False) as loaded:
        params = {key: loaded[key] for key in ("kernel", "bias", "output")}
    prediction = predict(params, cases, norm)
    stored = np.stack([np.fromfile(run / "predictions" / f'{c["id"]}.bin', dtype="<f8") for c in cases])
    if not np.array_equal(stored, prediction):
        raise ValueError("checkpoint predictions do not replay exactly")
    physics = model.run_physics(binary, "evaluate-dev", dataset, run / "predictions")
    zero = [{**cases[0], "rhs": np.zeros_like(cases[0]["rhs"])}]
    zero_prediction = predict(params, zero, norm)[0]
    # Fixed probes on the first training field only; no additional weight updates.
    base = cases[0]
    neg = predict(params, [{**base, "rhs": -base["rhs"]}], norm)[0]
    doubled = predict(params, [{**base, "rhs": 2 * base["rhs"]}], norm)[0]
    shape = tuple(reversed(witness["counts_xyz"]))
    witness_case = [{**base, "rhs": np.array(witness["rhs"]).reshape(shape)}]
    witness_prediction = predict(params, witness_case, norm)[0]
    index = witness["observation_index"]
    optimized = set(recorded["training_case_ids"])
    optimized_metrics = [c for c in physics["cases"] if c["id"] in optimized]
    return {
        "source_run": run.name, "model_commit": recorded["model_commit"],
        "weights_sha256": recorded["weights_sha256"], "weight_updates_in_this_diagnostic": 0,
        "development_metrics": summary(physics),
        "max_prediction_replay_error": float(np.max(np.abs(stored - prediction))),
        "zero_rhs_predicted_pressure_rms_pa": float(np.sqrt(np.mean(zero_prediction**2))),
        "zero_rhs_predicted_pressure_max_pa": float(np.max(np.abs(zero_prediction))),
        "first_train_sign_symmetry_defect_rms_pa": float(np.sqrt(np.mean((neg + prediction[0])**2))),
        "first_train_double_amplitude_defect_rms_pa": float(np.sqrt(np.mean((doubled - 2 * prediction[0])**2))),
        "nonlocal_witness": {
            "reference_delta_pressure_pa": witness["reference_pressure_at_observation_pa"],
            "model_delta_pressure_pa": float(witness_prediction[index] - zero_prediction[index]),
            "observation_xyz": witness["observation_xyz"],
            "source_u_face_xyz": witness["source_u_face_xyz"],
        },
        "optimized_cases": optimized_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--lab-bin", type=Path, default=Path("target/release/rheon-surrogate-lab"))
    args = parser.parse_args()
    source = model.source_commit()
    if args.output.exists():
        parser.error("output already exists")
    binary = args.lab_bin.resolve()
    binary_sha = model.digest(binary)
    start = time.perf_counter()
    verification = model.run_physics(binary, "verify", args.dataset)
    manifest_sha = model.digest(args.dataset / "manifest.json")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    cases = model.load_cases(args.dataset, manifest)  # Excludes test before reading its arrays.
    norm = model.fit_normalization(cases)
    args.output.mkdir(parents=True)
    baselines = model.run_physics(binary, "baselines-dev", args.dataset, args.output / "baselines")
    witness = json.loads((args.output / "baselines" / "nonlocal-witness.json").read_text())
    comparisons = [model_diagnostics(run, cases, norm, witness, binary, args.dataset) for run in args.runs]
    if model.digest(binary) != binary_sha or model.digest(args.dataset / "manifest.json") != manifest_sha:
        raise ValueError("evaluator or dataset changed")
    baseline_summary = {}
    for entry in baselines["methods"]:
        baseline_summary[entry["method"]] = summary(entry["evaluation"])
        if entry["solve_receipts"]:
            counts = [c["iterations"] for c in entry["solve_receipts"]]
            baseline_summary[entry["method"]]["iterations"] = {"min": min(counts), "max": max(counts), "total": sum(counts)}
    report = {
        "status": "complete", "source_commit": source, "scope": "train_validation_only",
        "evaluator": {"binary_sha256": binary_sha, **verification["tool"]},
        "dataset_manifest_sha256": manifest_sha,
        "test_policy": "old test metrics are historical/exposed; no test predictions evaluated here; new final test reservation documented before future changes",
        "normalization": norm, "family_statistics": distribution_stats(cases, norm),
        "baselines": baseline_summary, "train_mean_fit_ids": baselines["train_mean_fit_ids"],
        "model_diagnostics": comparisons,
        "compute": {
            "new_optimizer_steps": 0, "new_model_candidates": 0,
            "diagonal_operator_calls": baselines["diagonal_operator_calls"],
            "baseline_workflow_seconds": baselines["elapsed_seconds_including_io_verification_and_repeated_reference_solves"],
            "total_wall_seconds": time.perf_counter() - start,
            "limits": "single CPU timed diagnostic section; includes repeated verification, allocation, subprocesses and data I/O; excludes Python startup/imports, initial Git/binary-hash checks and final receipt writing; no matched-quality speed claim",
        },
    }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baselines": baseline_summary, "family_statistics": report["family_statistics"], "compute": report["compute"]}, indent=2))


if __name__ == "__main__":
    main()
