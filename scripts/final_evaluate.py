#!/usr/bin/env python3
"""Freeze, unlock and close the bounded fixed-operator final comparison."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import select
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from final_worker import METHODS

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "target/release/rheon-surrogate-lab"
PROBE = ROOT / "target/release/rheon-warmstart-probe"
WEIGHTS = ROOT / "experiments/final/weights.npz"
PATCH = ROOT.parent / "rheon-warmstart"
BASE = "34235d33c886aa05b5bdf95d920b863d1fb2ad6f"
LIMITS = {"relative_residual_original_rhs": 1e-10, "absolute_residual": 1e-12,
          "predicted_divergence": 1e-9, "actual_f32_divergence": 1e-5, "max_iterations": 1000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def require_clean():
    if git("status", "--porcelain") or git("status", "--porcelain", cwd=PATCH):
        raise ValueError("commit source/freeze changes before this action")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_files():
    exact = {"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml", "requirements.txt",
             "docs/decisions/0005-final-evaluation.md", "docs/final-test-reservation.json"}
    prefixes = ("src/", "scripts/", "tests/", "probes/warmstart/", "patches/", "experiments/final/weights.npz", "experiments/final/review.json")
    return sorted(p for p in git("ls-files").splitlines() if p in exact or p.startswith(prefixes))


def orders():
    base = [0, 1, 5, 2, 4, 3]
    rows = [[METHODS[(v + shift) % 6] for v in base] for shift in range(6)]
    return rows + [list(reversed(row)) for row in rows]


def runtime():
    return {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
            "platform": platform.platform(), "worker_openblas_threads": 1, "worker_omp_threads": 1}


def freeze(path):
    require_clean()
    if path.exists():
        raise ValueError("freeze already exists; do not replace it")
    review = json.loads((ROOT / "experiments/final/review.json").read_text())
    if review["status"] != "approved_before_freeze":
        raise ValueError("independent source/development review is required")
    patch_manifest = json.loads((ROOT / "patches/manifest.json").read_text())
    if git("rev-parse", "HEAD^{tree}", cwd=PATCH) != patch_manifest["reviewed_tree"]:
        raise ValueError("unreviewed patch tree")
    candidate_path = ROOT / "docs/plans/pressure-data/reports/global-candidate.json"
    candidate = json.loads(candidate_path.read_text())
    if digest(WEIGHTS) != candidate["weights_sha256"] or candidate["steps"] != 500:
        raise ValueError("candidate differs from original500-step checkpoint")
    code_commit = git("rev-parse", "HEAD")
    native = json.loads(subprocess.check_output([str(PROBE), "--identity"], text=True))
    if (native["lab_commit"] != code_commit or native["lab_dirty"] != "false"
            or native["patch_commit"] != git("rev-parse", "HEAD", cwd=PATCH) or native["patch_dirty"] != "false"):
        raise ValueError("rebuild probe from clean reviewed code")
    development = json.loads(subprocess.check_output([str(LAB), "verify", str(ROOT / "data/smoke")], text=True))
    if development["tool"]["experiment_commit"] != code_commit or development["tool"]["source_dirty"]:
        raise ValueError("rebuild generator from clean reviewed code")
    value = {
        "schema": "rheon-final-freeze-v1", "status": "frozen_before_final_generation", "created_at_utc": stamp(),
        "code_commit": code_commit, "source_sha256": {p: digest(ROOT / p) for p in source_files()},
        "binaries": {str(p.relative_to(ROOT)): digest(p) for p in (LAB, PROBE)},
        "native_identity": native, "rheon_base": BASE, "reviewed_patch_tree": patch_manifest["reviewed_tree"],
        "candidate_source_commit": candidate["source_commit"], "candidate_weights_sha256": digest(WEIGHTS),
        "candidate_receipt_sha256": digest(candidate_path), "optimizer_steps_after_candidate": 0,
        "dataset_schema": "rheon-pressure-final-v1", "family": "oscillatory_modes", "seeds": list(range(9000,9016)),
        "methods": list(METHODS), "limits": LIMITS, "method_orders": orders(), "repeats_per_regime": 12,
        "regimes": ["fresh_python_model", "resident_python_model"],
        "regime_order": "fresh/resident on even rows, resident/fresh on odd rows",
        "quality_passes_per_method": 1, "test_cases": 16, "runtime": runtime(),
        "cache_scope": "OS caches uncontrolled; fresh Python/model startup versus resident model only; native child fresh for every call",
        "protocol": "docs/decisions/0005-final-evaluation.md", "no_post_test_tuning": True,
    }
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"freeze_sha256": digest(path), "code_commit": code_commit,
                      "next": "Commit this freeze before generate/run. Do not rebuild the frozen binaries."}, indent=2))


def validate_freeze(path):
    require_clean()
    relative = path.resolve().relative_to(ROOT)
    git("ls-files", "--error-unmatch", str(relative))
    f = json.loads(path.read_text())
    if (f["schema"] != "rheon-final-freeze-v1" or f["status"] != "frozen_before_final_generation"
            or f["limits"] != LIMITS or f["methods"] != list(METHODS) or f["method_orders"] != orders()
            or f["seeds"] != list(range(9000,9016)) or f["runtime"] != runtime()):
        raise ValueError("frozen protocol/runtime changed")
    if list(f["source_sha256"]) != source_files():
        raise ValueError("frozen source file set changed")
    for relative, expected in {**f["source_sha256"], **f["binaries"]}.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen artifact changed: {relative}")
    if digest(WEIGHTS) != f["candidate_weights_sha256"] or git("rev-parse", "HEAD^{tree}", cwd=PATCH) != f["reviewed_patch_tree"]:
        raise ValueError("candidate/patch changed")
    native = json.loads(subprocess.check_output([str(PROBE), "--identity"], text=True))
    if native != f["native_identity"]:
        raise ValueError("native build identity changed")
    return f


def generate(path, dataset):
    begin = time.perf_counter()
    f = validate_freeze(path)
    if dataset.exists():
        raise ValueError("reserved destination already exists; no generation retry")
    response = json.loads(subprocess.check_output([str(LAB), "generate-final-frozen", str(dataset), str(path)], text=True))
    if response["freeze_sha256"] != digest(path) or response["cases"] != 16:
        raise ValueError("materialization identity")
    report = {"status": "materialized_after_freeze", "at_utc": stamp(), "freeze_sha256": digest(path),
              "freeze_commit": git("rev-parse", "HEAD"), "code_commit": f["code_commit"],
              "dataset_manifest_sha256": digest(dataset / "manifest.json"),
              "gate_generation_verification_seconds": time.perf_counter() - begin, "verification": response}
    dataset.with_name(dataset.name + "-generation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def line(process):
    ready, _, _ = select.select([process.stdout], [], [], 30)
    if not ready:
        raise TimeoutError("resident worker did not respond")
    data = process.stdout.readline()
    if not data:
        raise RuntimeError("worker exited: " + process.stderr.read())
    return json.loads(data)


def request(process, quality=False):
    begin = time.perf_counter()
    process.stdin.write(json.dumps({"op": "run", "quality": quality}) + "\n")
    process.stdin.flush()
    value = line(process)
    return value, time.perf_counter() - begin


def fingerprint(cases):
    return [{k:v for k,v in c.items() if k != "reference_relative_rmse"} for c in cases]


def summarize(cases):
    def maximum(name):
        values = [c[name] for c in cases if c.get(name) is not None]
        return max(values) if values else None
    errors = [c["reference_relative_rmse"] for c in cases if c.get("reference_relative_rmse") is not None]
    iterations = [c["iterations"] for c in cases if c.get("iterations") is not None]
    return {"cases": len(cases), "qualified": sum(c["qualified"] for c in cases),
            "failed_ids": [c["id"] for c in cases if not c["qualified"]],
            "solver_errors": {c["id"]: c["solver_error"] for c in cases if "solver_error" in c},
            "mean_reference_relative_rmse": statistics.mean(errors) if errors else None,
            "iterations_total": sum(iterations) if iterations else None,
            "iterations_range": [min(iterations),max(iterations)] if iterations else None,
            **{key: maximum(key) for key in ("relative_residual_l2", "linear_gate_ratio", "predicted_gate_ratio",
                                             "actual_gate_ratio", "actual_f32_divergence_max", "predicted_divergence_max")}}


def compare(dataset, output, scope, rows, expected_identity):
    if output.exists():
        raise ValueError("comparison output exists; no silent rerun")
    output.mkdir(parents=True)
    start = time.perf_counter()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    def command(method):
        return [sys.executable, str(ROOT / "scripts/final_worker.py"), str(dataset), method,
                str(WEIGHTS), str(PROBE), str(output), scope]
    manifest = json.loads((dataset / "manifest.json").read_text())
    ids = [c["spec"]["id"] for c in manifest["cases"] if c["spec"]["split"] in
           (("final_test",) if scope == "final" else ("train", "validation"))]
    def validate(value, method):
        native = value["native"]
        if value["method"] != method or native["tool"] != expected_identity or native["scope"] != scope:
            raise ValueError("worker/native identity")
        if [c["id"] for c in native["cases"]] != ids:
            raise ValueError("case identity")
    workers = {}
    setup = {}
    shutdown = {}
    quality = {}
    quality_seconds = {}
    records = []
    try:
        for method in METHODS:
            begin = time.perf_counter()
            p = subprocess.Popen(command(method), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, bufsize=1)
            workers[method] = p
            ready = line(p)
            if ready["status"] != "ready" or ready["method"] != method:
                raise ValueError("worker startup")
            setup[method] = {"parent_wall_seconds": time.perf_counter() - begin, **ready}
        for method in METHODS:
            value, elapsed = request(workers[method], quality=True)
            validate(value, method)
            quality[method] = value["native"]["cases"]
            quality_seconds[method] = elapsed
        with (output / "journal.jsonl").open("w") as journal:
            for row_index, row in enumerate(rows):
                regimes = ("fresh_python_model", "resident_python_model")
                if row_index % 2:
                    regimes = tuple(reversed(regimes))
                for regime in regimes:
                    for method in row:
                        if regime == "fresh_python_model":
                            begin = time.perf_counter()
                            value = json.loads(subprocess.check_output(command(method)+["--once"], env=env, text=True, timeout=30))
                            elapsed = time.perf_counter() - begin
                        else:
                            value, elapsed = request(workers[method])
                        validate(value, method)
                        if fingerprint(value["native"]["cases"]) != fingerprint(quality[method]):
                            raise ValueError("numerical result changed across frozen timing repetitions")
                        record = {"row": row_index, "regime": regime, "method": method, "wall_seconds": elapsed, **value}
                        records.append(record)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
        for method,p in workers.items():
            begin = time.perf_counter()
            p.stdin.write('{"op":"stop"}\n')
            p.stdin.flush()
            if line(p)["status"] != "stopped" or p.wait(timeout=10) != 0:
                raise ValueError("worker shutdown")
            shutdown[method] = time.perf_counter() - begin
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)
    total_wall = time.perf_counter() - start
    measurements = {}
    for method in METHODS:
        measurements[method] = {}
        for regime in ("fresh_python_model", "resident_python_model"):
            samples = [r for r in records if r["method"] == method and r["regime"] == regime]
            wall = [r["wall_seconds"] for r in samples]
            result = {"batches":len(wall), "wall_samples_seconds":wall,
                      "median_seconds":statistics.median(wall), "min_seconds":min(wall), "max_seconds":max(wall),
                      "mean_worker_stages_seconds": {k: statistics.mean(r["worker_stages_seconds"][k] for r in samples)
                                                      for k in samples[0]["worker_stages_seconds"]},
                      "mean_native_stages_seconds": {k: statistics.mean(r["native"]["timing_seconds"][k] for r in samples)
                                                      for k in samples[0]["native"]["timing_seconds"]}}
            if regime == "resident_python_model":
                active = setup[method]["parent_wall_seconds"] + quality_seconds[method] + sum(wall) + shutdown[method]
                result.update({"startup":setup[method], "quality_warmup_seconds":quality_seconds[method],
                               "shutdown_seconds":shutdown[method], "total_active_wall_seconds":active,
                               "amortized_total_seconds_per_timed_batch":active/len(wall)})
            else:
                result["mean_setup_inside_worker_seconds"] = statistics.mean(r["setup_seconds"] for r in samples)
            measurements[method][regime] = result
    return {"status":"complete", "scope":scope, "at_utc":stamp(), "cases_per_batch":len(ids),
            "quality_projections":len(ids)*len(METHODS), "timed_projections":len(ids)*len(records),
            "quality_summaries":{m:summarize(c) for m,c in quality.items()}, "quality_cases":quality,
            "measurements":measurements, "method_orders":rows, "total_harness_wall_seconds":total_wall,
            "native_identity":expected_identity, "journal_sha256":digest(output/"journal.jsonl"),
            "optimizer_steps":0, "runtime":runtime(), "limits":LIMITS,
            "timing_scope":"ADR0005; fresh Python/model startup vs resident model; OS caches uncontrolled; native child fresh every call; amortized resident cost includes startup/quality-warmup/12 calls/teardown",
            "no_post_test_tuning":True}


def run(path, dataset, output, development=False):
    begin = time.perf_counter()
    if development:
        require_clean()
        f = None
        identity = json.loads(subprocess.check_output([str(PROBE), "--identity"], text=True))
        if identity["lab_dirty"] != "false" or identity["lab_commit"] != git("rev-parse","HEAD"):
            raise ValueError("clean current development build required")
    else:
        f = validate_freeze(path)
        identity = f["native_identity"]
    verified = json.loads(subprocess.check_output([str(LAB), "verify", str(dataset)], text=True))
    manifest_hash = digest(dataset / "manifest.json")
    if not development and (verified["schema"] != f["dataset_schema"] or verified["freeze_sha256"] != digest(path)):
        raise ValueError("dataset is not bound to this freeze")
    if development and verified["schema"] != "rheon-pressure-dataset-v1":
        raise ValueError("development smoke cannot open reserved schema")
    preparation = time.perf_counter() - begin
    receipt = compare(dataset, output, "development" if development else "final",
                      [list(METHODS),list(reversed(METHODS))] if development else f["method_orders"], identity)
    if digest(dataset / "manifest.json") != manifest_hash:
        raise ValueError("dataset changed")
    if not development:
        validate_freeze(path)
    receipt.update({"source_commit":identity["lab_commit"], "dataset_manifest_sha256":manifest_hash,
                    "freeze_sha256":None if development else digest(path),
                    "candidate_weights_sha256":digest(WEIGHTS), "preparation_gate_verification_seconds":preparation,
                    "total_gate_comparison_seconds_before_receipt_write":time.perf_counter()-begin})
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"quality":receipt["quality_summaries"], "timed_projections":receipt["timed_projections"],
                      "timing":{m:{r:{k:v for k,v in d.items() if k in ("median_seconds","amortized_total_seconds_per_timed_batch")}
                                   for r,d in regimes.items()} for m,regimes in receipt["measurements"].items()}},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","generate","run","dev-smoke"))
    parser.add_argument("--freeze",type=Path,default=ROOT/"experiments/final/freeze.json")
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/final-9000")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/final-v1")
    args = parser.parse_args()
    if args.action == "freeze": freeze(args.freeze.resolve())
    elif args.action == "generate": generate(args.freeze.resolve(),args.dataset.resolve())
    else: run(args.freeze.resolve(),args.dataset.resolve(),args.output.resolve(),args.action == "dev-smoke")


if __name__ == "__main__":
    main()
