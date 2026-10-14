"""Captured-data Adam diagnostic using the lab's unchanged small NumPy CNN.

This separate recipe demonstrates optimization and restore. It makes no pressure
solver qualification or speedup claim; native physics evaluation stays separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import train_cpu as model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def event(stage, completed, total, metrics=None):
    print(json.dumps({"format": "tuldok_progress_v1", "stage": stage, "completed": completed,
                      "total": total, "metrics": metrics or {}}, allow_nan=False), flush=True)


def data(dataset, splits_path, include_test):
    splits = json.loads(Path(splits_path).read_text())
    if (set(splits) != {"schema", "shape_xyz", "dtype", "rheon_revision", "dataset_sha256",
                       "manifest_sha256", "generator_revision", "policy", "cases"}
            or splits["schema"] != "rheon-cnn-splits-v1" or splits["shape_xyz"] != [8, 7, 6]
            or splits["dtype"] != "<f8" or splits["rheon_revision"] != "34235d33c886aa05b5bdf95d920b863d1fb2ad6f"
            or splits["dataset_sha256"] != sha(dataset)
            or splits["policy"] != "whole seeds and scenario families; standalone fields; checks excluded"):
        raise ValueError("Captured dataset/split contract differs")
    families = {"train": {"white_noise", "polynomial_mix"},
                "validation": {"localized_blob"}, "test": {"oscillatory_modes"}}
    counts = {"train": 8, "validation": 4, "test": 4}
    rows = splits["cases"]
    if (len(rows) != 16 or len({row["id"] for row in rows}) != 16
            or len({row["seed"] for row in rows}) != 16
            or any(set(row) != {"id", "family", "seed", "split", "trajectory"}
                   or row["split"] not in families or row["family"] not in families[row["split"]]
                   or row["trajectory"] != row["id"] for row in rows)):
        raise ValueError("Split identity/family custody differs")
    cases = []
    with np.load(dataset, allow_pickle=False) as archive:
        if set(archive.files) != {split + "_" + field for split in counts for field in ("rhs", "pressure")}:
            raise ValueError("Captured dataset array inventory differs")
        for split, count in counts.items():
            selected = [row for row in rows if row["split"] == split]
            if len(selected) != count:
                raise ValueError("Persisted split counts differ")
            if split == "test" and not include_test:
                continue  # Never load heldout array members in the training recipe.
            arrays = {field: archive[split + "_" + field] for field in ("rhs", "pressure")}
            if any(array.dtype != np.dtype("<f8") or array.shape != (count, 6, 7, 8)
                   or not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("Captured dataset shape/dtype/values differ")
            for index, row in enumerate(selected):
                cases.append({**row, **{field: values[index] for field, values in arrays.items()}})
    return cases, splits


def weights(path):
    expected = {"kernel": (108, 16), "bias": (16,), "output": (16,)}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Checkpoint parameter inventory differs")
        params = {key: archive[key] for key in expected}
    if any(value.dtype != np.dtype("<f8") or value.shape != expected[key]
           or not np.isfinite(value).all() for key, value in params.items()):
        raise ValueError("Checkpoint shape/dtype/finiteness differs")
    return params


def run(args):
    start = time.perf_counter()
    output = Path(args.output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Output must be an existing empty directory")
    if np.__version__ != "2.3.5":
        raise ValueError("Use the pinned NumPy2.3.5 runtime")
    source_hash = sha(model.__file__)
    cases, splits = data(args.dataset, args.splits, args.command == "restore")
    training = [case for case in cases if case["split"] == "train"]
    event("loading", len(cases), len(cases))
    if args.command == "train":
        normalization = model.fit_normalization(cases)
        patches, targets = model.encode(cases, normalization)
        indices = [i for i, case in enumerate(cases) if case["split"] == "train"]
        params = model.initialize(seed=2026, hidden=16)
        event("optimizing", 0, args.steps)
        fit_start = time.perf_counter()
        history = model.train(params, patches[indices], targets[indices], args.steps)
        fitting_seconds = time.perf_counter() - fit_start
        event("optimizing", args.steps, args.steps, {"normalized_mse": history[-1]["normalized_mse"]})
        np.savez(output / "weights.npz", **params)
        restored = weights(output / "weights.npz")
        if not all(np.array_equal(params[key], restored[key]) for key in params):
            raise ValueError("Saved checkpoint differs from trained arrays")
    else:
        if not all((args.checkpoint, args.receipt, args.normalization, args.predictions)):
            raise ValueError("Restore requires the exact training artifacts")
        prior = json.loads(Path(args.receipt).read_text())
        normalization = json.loads(Path(args.normalization).read_text())
        if (prior.get("schema") != "rheon-cnn-training-v1" or prior.get("status") != "complete"
                or prior["dataset_sha256"] != sha(args.dataset) or prior["splits_sha256"] != sha(args.splits)
                or prior["weights_sha256"] != sha(args.checkpoint)
                or prior["normalization_sha256"] != sha(args.normalization)
                or prior["predictions_sha256"] != sha(args.predictions)
                or prior["source_sha256"] != source_hash
                or prior["wrapper_sha256"] != sha(__file__)
                or prior["numpy"] != np.__version__
                or prior["training_ids"] != [case["id"] for case in training]
                or normalization != model.fit_normalization(cases)):
            raise ValueError("Restore input/source/normalization/split identity differs")
        params = weights(args.checkpoint)
        patches, targets = model.encode(cases, normalization)
        history = []
        fitting_seconds = 0.0
        event("restoring", 1, 1)
        # Preserve original checkpoint bytes for generic verified downloads.
        (output / "weights.npz").write_bytes(Path(args.checkpoint).read_bytes())
    prediction, _ = model.forward(params, patches)
    prediction *= normalization["pressure_rms"]
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite model prediction")
    if args.command == "restore":
        with np.load(args.predictions, allow_pickle=False) as original:
            ids = original["case_ids"].tolist()
            indices = [next(i for i, case in enumerate(cases) if case["id"] == name) for name in ids]
            if (ids != [case["id"] for case in cases if case["split"] != "test"]
                    or not np.array_equal(prediction[indices], original["pressure"])):
                raise ValueError("Restored predictions differ from training-process observations")
    np.savez(output / "predictions.npz", pressure=prediction,
             case_ids=np.array([case["id"] for case in cases], dtype="U80"))
    metrics = {}
    for split in ("train", "validation", "test"):
        indices = [i for i, case in enumerate(cases) if case["split"] == split]
        if indices:
            errors = [float(np.linalg.norm(prediction[i] - cases[i]["pressure"].ravel()) /
                            max(np.linalg.norm(cases[i]["pressure"]), 1e-30)) for i in indices]
            metrics[split] = {"cases": len(indices), "mean_relative_pressure_l2": float(np.mean(errors)),
                              "max_relative_pressure_l2": max(errors)}
    (output / "splits.json").write_bytes(Path(args.splits).read_bytes())
    write(output / "normalization.json", normalization)
    receipt = {"schema": "rheon-cnn-training-v1" if args.command == "train" else "rheon-cnn-restored-v1",
               "status": "complete", "dataset_sha256": sha(args.dataset), "splits_sha256": sha(args.splits),
               "weights_sha256": sha(output / "weights.npz"), "normalization_sha256": sha(output / "normalization.json"),
               "predictions_sha256": sha(output / "predictions.npz"), "source_sha256": source_hash,
               "wrapper_sha256": sha(__file__), "numpy": np.__version__, "python": sys.version,
               "model": "original 1760-parameter radius-one NumPy CNN", "dtype": "float64",
               "seed": 2026, "optimizer": "original full-batch Adam lr0.01 beta(0.9,0.999) eps1e-8",
               "optimizer_steps": args.steps if args.command == "train" else 0,
               "training_ids": [case["id"] for case in training], "normalization": normalization,
               "history": history, "split_metrics": metrics, "fitting_seconds": fitting_seconds,
               "wall_seconds_after_imports": time.perf_counter() - start,
               "restored_predictions_equal": args.command == "restore",
               "scope": "neural optimization/restore diagnostic; no native physics or acceleration qualification; historical exposed test family"}
    write(output / "receipt.json", receipt)
    write(output / "summary.json", {"format": "tuldok_result_summary_v1", "title": "Captured CNN " + args.command,
          "checkpoint": "weights.npz", "metrics": [
              {"name": split + "_pressure_relative_l2", "value": values["mean_relative_pressure_l2"], "unit": "ratio"}
              for split, values in metrics.items()],
          "scope": ["Original radius-one CNN has known global-pressure limitations",
                    "Actual Adam optimization; restore does not fit holdout",
                    "Diagnostic numeric errors; no solver quality or speedup claim"]})
    event("publishing", 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "restore"))
    for name in ("dataset", "splits", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=200)
    for name in ("checkpoint", "receipt", "normalization", "predictions"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be1..1000")
    run(args)


if __name__ == "__main__":
    main()
