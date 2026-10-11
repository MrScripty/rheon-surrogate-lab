#!/usr/bin/env python3
"""Preregistered 335-gain global spectral model: one 500-step CPU run."""
import argparse
import json
from pathlib import Path
import platform
import time

import train_cpu as common
import numpy as np

STEPS = 500
LEARNING_RATE = 0.05


def cosine_basis(shape_xyz):
    axes = []
    for n in shape_xyz:
        i = np.arange(n, dtype=np.float64)[:, None]
        k = np.arange(n, dtype=np.float64)[None, :]
        c = np.sqrt(2 / n) * np.cos(np.pi * (i + 0.5) * k / n)
        c[:, 0] = 1 / np.sqrt(n)
        axes.append(c)
    # Both spatial and frequency indices are X-fastest.
    return np.kron(axes[2], np.kron(axes[1], axes[0]))[:, 1:].copy()


def predict(log_gain, rhs, basis):
    raw = ((rhs @ basis) * np.exp(log_gain)) @ basis.T
    return raw - raw[:, :1]


def loss_gradient(log_gain, rhs, target, basis, operator):
    pressure_denominator = np.sum(target**2, axis=1)
    rhs_denominator = np.sum(rhs**2, axis=1)
    if not (np.isfinite(pressure_denominator).all() and np.isfinite(rhs_denominator).all()
            and np.all(pressure_denominator > 0) and np.all(rhs_denominator > 0)):
        raise ValueError("relative loss needs positive finite training case norms")
    gain = np.exp(log_gain)
    coefficients = rhs @ basis
    raw = (coefficients * gain) @ basis.T
    pressure = raw - raw[:, :1]
    error = pressure - target
    residual = pressure @ operator.T - rhs
    pressure_loss = np.mean(np.sum(error**2, axis=1) / pressure_denominator)
    residual_loss = np.mean(np.sum(residual**2, axis=1) / rhs_denominator)
    gradient_p = (error / pressure_denominator[:, None]
                  + (residual / rhs_denominator[:, None]) @ operator) / len(rhs)
    gradient_p[:, 0] -= gradient_p.sum(axis=1)
    gradient = np.sum((gradient_p @ basis) * coefficients, axis=0) * gain
    return float(0.5 * (pressure_loss + residual_loss)), gradient, {
        "relative_pressure_mse": float(pressure_loss),
        "relative_full_residual_mse": float(residual_loss),
    }


def fit(rhs, target, basis, operator):
    # A single scalar diagonal initialization; no analytic eigenvalues or inverse.
    initial_gain = 1 / float(np.mean(np.diag(operator)))
    log_gain = np.full(basis.shape[1], np.log(initial_gain))
    moment = np.zeros_like(log_gain)
    square = np.zeros_like(log_gain)
    history = []
    for step in range(STEPS + 1):
        loss, gradient, components = loss_gradient(log_gain, rhs, target, basis, operator)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("nonfinite training result")
        if step % 50 == 0:
            history.append({"step": step, "loss": loss, **components})
        if step == STEPS:
            break
        iteration = step + 1
        moment = 0.9 * moment + 0.1 * gradient
        square = 0.999 * square + 0.001 * gradient**2
        log_gain -= LEARNING_RATE * (moment / (1 - 0.9**iteration)) / (
            np.sqrt(square / (1 - 0.999**iteration)) + 1e-8)
    return log_gain, history, initial_gain


def split_metrics(physics):
    from diagnose_cpu import summary
    return summary(physics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lab-bin", type=Path, default=Path("target/release/rheon-surrogate-lab"))
    args = parser.parse_args()
    commit = common.source_commit()
    if args.output.exists():
        parser.error("output exists; this protocol permits one recorded candidate run")
    binary = args.lab_bin.resolve()
    binary_sha = common.digest(binary)
    start = time.perf_counter()
    verified = common.run_physics(binary, "verify", args.dataset)
    manifest_sha = common.digest(args.dataset / "manifest.json")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    cases = common.load_cases(args.dataset, manifest)
    training = [case for case in cases if case["split"] == "train"]
    rhs = np.stack([c["rhs"].ravel() for c in training])
    target = np.stack([c["pressure"].ravel() for c in training])
    args.output.mkdir(parents=True)
    exported = common.run_physics(binary, "export-operator", args.output / "operator")
    operator_path = args.output / "operator" / "operator.bin"
    if common.digest(operator_path) != exported["sha256"]:
        raise ValueError("operator hash mismatch")
    operator = np.fromfile(operator_path, dtype="<f8").reshape(exported["shape"])
    basis = cosine_basis(exported["counts_xyz"])
    if not (np.isfinite(operator).all() and np.array_equal(operator, operator.T)
            and np.max(np.abs(operator.sum(axis=1))) < 1e-12):
        raise ValueError("unsupported operator symmetry/nullspace")
    preparation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    log_gain, history, initial_gain = fit(rhs, target, basis, operator)
    training_seconds = time.perf_counter() - start
    # Validation enters prediction only after all 500 updates and final weights.
    all_rhs = np.stack([case["rhs"].ravel() for case in cases])
    start = time.perf_counter()
    predictions = predict(log_gain, all_rhs, basis)
    inference_seconds = time.perf_counter() - start
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite prediction")
    zero_response = predict(log_gain, np.zeros_like(rhs[:1]), basis)
    if not np.array_equal(zero_response, np.zeros_like(zero_response)):
        raise ValueError("zero-response contract failed")
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir()
    for case, p in zip(cases, predictions):
        p.astype("<f8").tofile(prediction_dir / f'{case["id"]}.bin')
    np.savez(args.output / "weights.npz", log_gain=log_gain)
    start = time.perf_counter()
    physics = common.run_physics(binary, "evaluate-dev", args.dataset, prediction_dir)
    evaluation_seconds = time.perf_counter() - start
    if common.digest(binary) != binary_sha or common.digest(args.dataset / "manifest.json") != manifest_sha:
        raise ValueError("evaluator or dataset changed during experiment")
    metrics = split_metrics(physics)
    validation = metrics["validation"]
    # Relative pressure metric is included in the shared summary.
    signal = (validation["mean_relative_residual_l2"] < 0.7581104033832886
              and validation["max_corrected_f32_divergence"] < 0.7246515639126301
              and validation["mean_relative_pressure_rmse"] <= 0.9640346380051443)
    receipt = {
        "status": "complete", "candidate": "positive learned global cosine-mode gains v1",
        "parameters": len(log_gain), "steps": STEPS, "learning_rate": LEARNING_RATE,
        "source_commit": commit, "dataset_manifest_sha256": manifest_sha,
        "evaluator": {"binary_sha256": binary_sha, **verified["tool"]},
        "operator_sha256": exported["sha256"], "operator_calls": exported["operator_calls"],
        "weights_sha256": common.digest(args.output / "weights.npz"),
        "prediction_sha256": {case["id"]: common.digest(prediction_dir / f'{case["id"]}.bin') for case in cases},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "basis": "orthonormal tensor-product cosine, constant mode omitted; global X-fastest",
        "prior_limit": "tailored uniform sealed rectangular operator; no analytic inverse gains used",
        "initial_gain": initial_gain, "zero_rhs_pressure_max": float(np.max(np.abs(zero_response))),
        "training_case_ids": [c["id"] for c in training],
        "loss": "mean_cases(0.5*pressure_error_squared/target_squared_norm + 0.5*full_residual_squared/rhs_squared_norm)",
        "training_pressure_squared_norms": np.sum(target**2, axis=1).tolist(),
        "training_rhs_squared_norms": np.sum(rhs**2, axis=1).tolist(),
        "history": history, "development_metrics": metrics,
        "validation_evaluations": 1, "test_evaluations": 0,
        "initial_guess_signal_passed": bool(signal),
        "selection": "predeclared final step 500 only; no sweep/restart/early stopping/validation tuning",
        "timing_seconds": {"preparation_with_reference_checks_and_operator_export": preparation_seconds,
                           "training_500_steps": training_seconds,
                           "batch_inference_12_cases_with_gauge": inference_seconds,
                           "evaluation_with_reverification_process_and_io": evaluation_seconds},
        "timing_limits": "one CPU run; excludes Python startup, initial Git/hash checks, prediction/checkpoint writes; no speedup claim",
        "physics": physics,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("initial_guess_signal_passed", "development_metrics", "history", "timing_seconds")}, indent=2))


if __name__ == "__main__":
    main()
