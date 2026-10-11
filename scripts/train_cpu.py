#!/usr/bin/env python3
"""Small NumPy volumetric CNN; Rust/Rheon remains the only physics evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

# Bound CPU threads before importing NumPy. No framework or model download.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(root, manifest, include_test=False):
    cases = []
    for entry in manifest["cases"]:
        if entry["spec"]["split"] == "check" or (entry["spec"]["split"] in ("test", "final_test") and not include_test):
            continue
        fields = {}
        for name in ("rhs", "pressure"):
            spec = entry["arrays"][name]
            path = root / spec["path"]
            if digest(path) != spec["sha256"]:
                raise ValueError("array changed after verification")
            fields[name] = np.fromfile(path, dtype=spec["dtype"]).reshape(
                tuple(reversed(spec["shape_xyz"]))
            )
        cases.append({**entry["spec"], **fields})
    return cases


def fit_normalization(cases):
    """Holdout changes cannot affect fitted statistics or their provenance."""
    training = [case for case in cases if case["split"] == "train"]
    if not training:
        raise ValueError("normalization requires training data")
    rhs = np.stack([case["rhs"] for case in training])
    pressure = np.stack([case["pressure"] for case in training])
    result = {
        "rhs_mean": float(rhs.mean()),
        "rhs_std": float(rhs.std()),
        # Scaling without centering preserves the cell-zero target gauge.
        "pressure_rms": float(np.sqrt(np.mean(pressure**2))),
        "fit_case_ids": [case["id"] for case in training],
    }
    if not all(np.isfinite(result[k]) and result[k] > 0 for k in ("rhs_std", "pressure_rms")):
        raise ValueError("normalization scale must be finite and positive")
    return result


def encode(cases, norm):
    fields = []
    targets = []
    for case in cases:
        rhs = case["rhs"]
        z, y, x = np.meshgrid(
            *(2 * (np.arange(n, dtype=np.float64) + 0.5) / n - 1 for n in rhs.shape),
            indexing="ij",
        )
        fields.append(np.stack(((rhs - norm["rhs_mean"]) / norm["rhs_std"], x, y, z)))
        targets.append((case["pressure"] / norm["pressure_rms"]).ravel())
    fields = np.stack(fields)  # N,C,Z,Y,X
    padded = np.pad(fields, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3, 3), axis=(2, 3, 4))
    patches = windows.transpose(0, 2, 3, 4, 1, 5, 6, 7).reshape(len(cases), -1, 108).copy()
    return patches, np.stack(targets)


def initialize(seed=2026, hidden=16):
    rng = np.random.default_rng(seed)
    return {
        "kernel": rng.normal(0, 1 / np.sqrt(108), (108, hidden)),
        "bias": np.zeros(hidden),
        "output": rng.normal(0, 1 / np.sqrt(hidden), hidden),
    }


def forward(params, patches):
    hidden = np.tanh(patches @ params["kernel"] + params["bias"])
    raw = hidden @ params["output"]
    return raw - raw[:, :1], hidden


def loss_and_gradients(params, patches, target):
    prediction, hidden = forward(params, patches)
    error = prediction - target
    loss = float(np.mean(error**2))
    raw_gradient = 2 * error / error.size
    # Every pressure value depends on the subtracted gauge-cell prediction.
    raw_gradient[:, 0] -= raw_gradient.sum(axis=1)
    hidden_gradient = raw_gradient[:, :, None] * params["output"] * (1 - hidden**2)
    gradients = {
        "kernel": patches.reshape(-1, patches.shape[-1]).T @ hidden_gradient.reshape(-1, hidden.shape[-1]),
        "bias": hidden_gradient.sum(axis=(0, 1)),
        "output": np.einsum("nv,nvh->h", raw_gradient, hidden),
    }
    return loss, gradients


def train(params, patches, target, steps, learning_rate=0.01):
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    squares = {key: np.zeros_like(value) for key, value in params.items()}
    initial = loss_and_gradients(params, patches, target)[0]
    history = [{"step": 0, "normalized_mse": initial}]
    for step in range(1, steps + 1):
        loss, gradients = loss_and_gradients(params, patches, target)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise ValueError("nonfinite training result")
        for key, gradient in gradients.items():
            moments[key] = 0.9 * moments[key] + 0.1 * gradient
            squares[key] = 0.999 * squares[key] + 0.001 * gradient**2
            params[key] -= learning_rate * (moments[key] / (1 - 0.9**step)) / (
                np.sqrt(squares[key] / (1 - 0.999**step)) + 1e-8
            )
        if step % 50 == 0 or step == steps:
            history.append({"step": step, "normalized_mse": loss_and_gradients(params, patches, target)[0]})
    return history


def run_physics(binary, command, *paths):
    result = subprocess.run([str(binary), command, *(str(p) for p in paths)], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    tool = report.get("tool", {})
    if (tool.get("schema") != "rheon-pressure-dataset-v1"
            or tool.get("rheon_commit") != "34235d33c886aa05b5bdf95d920b863d1fb2ad6f"
            or tool.get("source_dirty") is not False
            or len(tool.get("experiment_commit", "")) != 40):
        raise ValueError("evaluator must identify a clean source commit and supported schema/Rheon pin")
    return report


def source_commit():
    root = Path(__file__).resolve().parents[1]
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError("commit source changes before recording a model run")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tiny-overfit", action="store_true", help="fit the first training case only; report all cases separately")
    parser.add_argument("--include-test", action="store_true", help="explicit final evaluation only; development defaults to train/validation")
    parser.add_argument("--lab-bin", type=Path, default=Path("target/release/rheon-surrogate-lab"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be 1..1000")
    model_commit = source_commit()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    binary = args.lab_bin.resolve()
    binary_sha = digest(binary)
    verification = run_physics(binary, "verify", args.dataset)
    manifest_path = args.dataset / "manifest.json"
    manifest_sha = digest(manifest_path)
    cases = load_cases(args.dataset, json.loads(manifest_path.read_text()), include_test=args.include_test)
    start = time.perf_counter()
    normalization = fit_normalization(cases)
    patches, targets = encode(cases, normalization)
    train_indices = [i for i, case in enumerate(cases) if case["split"] == "train"]
    if args.tiny_overfit:
        train_indices = train_indices[:1]
    preprocessing_seconds = time.perf_counter() - start
    params = initialize()
    start = time.perf_counter()
    history = train(params, patches[train_indices], targets[train_indices], args.steps)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    normalized_prediction, _ = forward(params, patches)
    predictions = normalized_prediction * normalization["pressure_rms"]
    inference_seconds = time.perf_counter() - start
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite predictions")
    args.output.mkdir(parents=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir()
    for case, p in zip(cases, predictions):
        p.astype("<f8").tofile(prediction_dir / f'{case["id"]}.bin')
    np.savez(args.output / "weights.npz", **params)
    start = time.perf_counter()
    physics = run_physics(binary, "evaluate" if args.include_test else "evaluate-dev", args.dataset, prediction_dir)
    physics_seconds = time.perf_counter() - start
    if digest(manifest_path) != manifest_sha:
        raise ValueError("dataset manifest changed during run")
    if digest(binary) != binary_sha or physics["tool"] != verification["tool"]:
        raise ValueError("evaluator changed during run")
    summary = {}
    for split in ("train", "validation", "test"):
        subset = [case for case in physics["cases"] if case["split"] == split]
        if not subset:
            continue
        summary[split] = {
            "cases": len(subset),
            "mean_pressure_rmse_pa": float(np.mean([c["pressure_rmse_pa"] for c in subset])),
            "mean_relative_pressure_rmse": float(np.mean([c["relative_pressure_rmse"] for c in subset])),
            "max_full_residual_l2": max(c["full_residual_l2"] for c in subset),
            "max_corrected_f32_divergence": max(c["corrected_f32_divergence_max"] for c in subset),
            "reference_qualified_cases": sum(c["reference_qualified"] for c in subset),
        }
    receipt = {
        "status": "complete", "model": "conv3d(4,16,3,pad=1)-tanh-conv1x1(16,1)-cell-zero-gauge",
        "parameters": sum(v.size for v in params.values()), "dtype": "float64", "seed": 2026,
        "model_commit": model_commit, "dataset_verification": verification,
        "evaluator": {"binary_sha256": binary_sha, **physics["tool"]},
        "dataset_manifest_sha256": manifest_sha, "weights_sha256": digest(args.output / "weights.npz"),
        "prediction_sha256": {c["id"]: digest(prediction_dir / f'{c["id"]}.bin') for c in cases},
        "normalization": normalization, "optimizer": "full-batch Adam lr=0.01 beta=(0.9,0.999) eps=1e-8",
        "mode": "tiny_overfit" if args.tiny_overfit else "heldout_smoke",
        "evaluation_scope": "all_noncheck" if args.include_test else "train_validation_only",
        "training_case_ids": [cases[i]["id"] for i in train_indices], "steps": args.steps,
        "history": history, "final_to_initial_training_mse": history[-1]["normalized_mse"] / history[0]["normalized_mse"],
        "split_metrics": summary, "physics": physics,
        "timing_seconds": {"preprocessing": preprocessing_seconds, "training": training_seconds,
                           "batch_inference_with_gauge_and_scale": inference_seconds,
                           "physical_evaluation_including_dataset_reverification_and_process": physics_seconds},
        "timing_limits": "single CPU smoke; excludes initial dataset loading, checkpoint/prediction writes; not a matched-quality acceleration benchmark",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("mode", "steps", "final_to_initial_training_mse", "split_metrics", "timing_seconds")}, indent=2))


if __name__ == "__main__":
    main()
