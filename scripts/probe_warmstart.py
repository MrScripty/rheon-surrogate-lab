#!/usr/bin/env python3
"""Fixed 20-pair development-only corrected-solve probe; never trains a model."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import train_cpu as common  # Sets bounded BLAS thread defaults before NumPy.
import global_candidate as model
import numpy as np

REPEATS = 20
METHODS = ("jacobi-pcg-v1", "sgs-pcg-v1")
BASE = "34235d33c886aa05b5bdf95d920b863d1fb2ad6f"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def provenance(report, commit, patch, ids, warm, hashes):
    if report["tool"] != {"lab_commit": commit, "lab_dirty": "false", "patch_commit": patch,
                          "patch_dirty": "false", "rheon_base": BASE}:
        raise ValueError("probe build provenance does not match clean current source")
    if report["status"] != "complete" or report["warm"] != warm or [c["id"] for c in report["cases"]] != ids:
        raise ValueError("probe output identity")
    for case in report["cases"]:
        if not case["qualified"] or case["initial_pressure_sha256"] != (hashes[case["id"]] if warm else None):
            raise ValueError("unqualified or changed prediction")


def batch(binary, dataset, output, method, warm, rhs, basis, gains, ids):
    start = time.perf_counter()
    stages = {"inference": 0.0, "prediction_directory_write": 0.0, "prediction_cleanup": 0.0}
    if warm:
        begin = time.perf_counter()
        predicted = model.predict(gains, rhs, basis)
        stages["inference"] = time.perf_counter() - begin
        begin = time.perf_counter()
        directory = tempfile.TemporaryDirectory(prefix="transfer-", dir=output)
        for name, pressure in zip(ids, predicted):
            pressure.astype("<f8").tofile(Path(directory.name) / f"{name}.bin")
        stages["prediction_directory_write"] = time.perf_counter() - begin
        location = directory.name
    else:
        location = "zero"
    begin = time.perf_counter()
    response = subprocess.check_output([str(binary), str(dataset), method, location], text=True)
    report = json.loads(response)
    stages["child_process_with_io_and_json"] = time.perf_counter() - begin
    if warm:
        begin = time.perf_counter()
        directory.cleanup()
        stages["prediction_cleanup"] = time.perf_counter() - begin
    elapsed = time.perf_counter() - start
    return report, elapsed, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lab-bin", type=Path, default=Path("target/release/rheon-surrogate-lab"))
    parser.add_argument("--probe-bin", type=Path, default=Path("target/release/rheon-warmstart-probe"))
    args = parser.parse_args()
    start = time.perf_counter()
    commit = common.source_commit()
    patch_path = Path(__file__).resolve().parents[2] / "rheon-warmstart"
    patch = git(patch_path, "rev-parse", "HEAD")
    patch_manifest = json.loads((Path(__file__).resolve().parents[1] / "patches/manifest.json").read_text())
    if git(patch_path, "rev-parse", "HEAD^{tree}") != patch_manifest["reviewed_tree"]:
        parser.error("patched checkout differs from the reviewed source tree")
    if git(patch_path, "status", "--porcelain"):
        parser.error("commit the reviewed local patch before probing")
    if args.output.exists():
        parser.error("output already exists")
    binary = args.probe_bin.resolve()
    binary_hash = common.digest(binary)
    lab_binary = args.lab_bin.resolve()
    lab_hash = common.digest(lab_binary)
    verified = common.run_physics(lab_binary, "verify", args.dataset)
    manifest_hash = common.digest(args.dataset / "manifest.json")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    candidate = json.loads((args.candidate / "receipt.json").read_text())
    weight_path = args.candidate / "weights.npz"
    if (not candidate["initial_guess_signal_passed"] or candidate["steps"] != 500
            or candidate["dataset_manifest_sha256"] != manifest_hash
            or candidate["weights_sha256"] != common.digest(weight_path)):
        raise ValueError("candidate identity or initial-guess gate")
    # Load only development RHS; no pressure targets enter this probe driver.
    entries = [c for c in manifest["cases"] if c["spec"]["split"] in ("train", "validation")]
    ids = [c["spec"]["id"] for c in entries]
    fields = []
    for case in entries:
        spec = case["arrays"]["rhs"]
        path = args.dataset / spec["path"]
        if common.digest(path) != spec["sha256"]:
            raise ValueError("RHS hash")
        fields.append(np.fromfile(path, dtype="<f8"))
    rhs = np.stack(fields)
    basis = model.cosine_basis(manifest["contract"]["grid"]["counts_xyz"])
    gains = np.load(weight_path)["log_gain"]
    predicted = model.predict(gains, rhs, basis)
    hashes = {name: hashlib.sha256(p.astype("<f8").tobytes()).hexdigest() for name, p in zip(ids, predicted)}
    if hashes != candidate["prediction_sha256"]:
        raise ValueError("frozen predictions do not replay exactly")
    args.output.mkdir(parents=True)
    preparation_seconds = time.perf_counter() - start
    measurements = {f"{method}/{mode}": [] for method in METHODS for mode in ("zero", "warm")}
    reports = {}
    for method in METHODS:
        for repeat in range(REPEATS):
            for warm in ((False, True) if repeat % 2 == 0 else (True, False)):
                report, elapsed, stages = batch(binary, args.dataset, args.output, method, warm, rhs, basis, gains, ids)
                provenance(report, commit, patch, ids, warm, hashes)
                if report["method"] != method:
                    raise ValueError("method changed")
                key = f'{method}/{"warm" if warm else "zero"}'
                if key in reports and reports[key] != report["cases"]:
                    raise ValueError("numerical result changed between repetitions")
                reports[key] = report["cases"]
                measurements[key].append({"wall_seconds": elapsed, "python_stages": stages,
                                          "native_stages": report["timing_seconds"]})
    if (binary_hash != common.digest(binary) or lab_hash != common.digest(lab_binary)
            or manifest_hash != common.digest(args.dataset / "manifest.json")
            or candidate["weights_sha256"] != common.digest(weight_path)):
        raise ValueError("source artifacts changed during probe")
    summaries = {}
    for key, values in measurements.items():
        wall = [x["wall_seconds"] for x in values]
        cases = reports[key]
        summaries[key] = {
            "batch_cases": len(cases), "batches": REPEATS, "qualified_case_executions": len(cases)*REPEATS,
            "iterations_per_batch": sum(c["iterations"] for c in cases),
            "iterations_range": [min(c["iterations"] for c in cases), max(c["iterations"] for c in cases)],
            "wall_seconds": {"median": statistics.median(wall), "min": min(wall), "max": max(wall)},
            "mean_python_stage_seconds": {k: statistics.mean(v["python_stages"][k] for v in values) for k in values[0]["python_stages"]},
            "mean_native_stage_seconds": {k: statistics.mean(v["native_stages"][k] for v in values) for k in values[0]["native_stages"]},
            "max_actual_f32_divergence": max(c["actual_f32_divergence_max"] for c in cases),
            "max_relative_residual_l2": max(c["relative_residual_l2"] for c in cases),
            "max_predicted_divergence": max(c["predicted_divergence_max"] for c in cases),
        }
    receipt = {
        "status": "complete", "source_commit": commit, "rheon_base": BASE, "local_patch_commit": patch,
        "reviewed_patch_tree": patch_manifest["reviewed_tree"],
        "candidate_source_commit": candidate["source_commit"], "candidate_weights_sha256": candidate["weights_sha256"],
        "candidate_receipt_sha256": common.digest(args.candidate / "receipt.json"),
        "probe_binary_sha256": binary_hash, "evaluator_binary_sha256": lab_hash, "evaluator": verified["tool"],
        "dataset_manifest_sha256": manifest_hash, "prediction_sha256": hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "preparation_seconds": preparation_seconds, "optimizer_steps": 0, "test_predictions": 0,
        "repeats": REPEATS, "paired_order": "alternate zero/warm by repeat index, for each preconditioner",
        "limits": {"relative_residual_original_rhs": 1e-10, "absolute_residual": 1e-12,
                   "predicted_divergence": 1e-9, "actual_f32_divergence": 1e-5},
        "timing_scope": "12-case warm-cache batch wall includes inference, prediction file transfer/cleanup, native child startup/read/hash/RHS assembly/allocation/PCG/correction/qualification and JSON return; preparation separately timed",
        "timing_limits": "336 cells, 20 pairs, one CPU environment; no Python parent startup or cold-cache cost; no simulator integration; no acceleration claim",
        "summaries": summaries, "cases": reports, "measurements": measurements,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
