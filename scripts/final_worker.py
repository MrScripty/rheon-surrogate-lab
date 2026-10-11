#!/usr/bin/env python3
"""One fixed method in a fresh Python process or a resident model process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

METHODS = ("analytic", "zero_jacobi", "learned_jacobi", "zero_sgs", "learned_sgs", "raw_learned")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(method, weights):
    if method.startswith("zero_"):
        return None
    # Baselines do not pay an artificial NumPy import/model initialization cost.
    import final_spectral as spectral
    import numpy as np
    basis = spectral.cosine_basis([8, 7, 6])
    if method == "analytic":
        parameters = spectral.eigenvalues([8, 7, 6], [.25, .3, .4], 1.25)
    else:
        with np.load(weights) as checkpoint:
            parameters = checkpoint["log_gain"].copy()
        if parameters.shape != (335,) or not np.isfinite(parameters).all():
            raise ValueError("frozen checkpoint shape/finite")
    return np, spectral, basis, parameters


def run(args, state, quality):
    start = time.perf_counter()
    prediction_seconds = 0.0
    input_seconds = 0.0
    transfer_seconds = 0.0
    cleanup_seconds = 0.0
    directory = None
    if state is not None:
        np, spectral, basis, parameters = state
        begin = time.perf_counter()
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        splits = ("final_test",) if args.scope == "final" else ("train", "validation")
        entries = [c for c in manifest["cases"] if c["spec"]["split"] in splits]
        rhs = []
        for case in entries:
            spec = case["arrays"]["rhs"]
            path = args.dataset / spec["path"]
            if digest(path) != spec["sha256"]:
                raise ValueError("RHS hash mismatch")
            rhs.append(np.fromfile(path, dtype="<f8"))
        rhs = np.stack(rhs)
        input_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        if args.method == "analytic":
            pressure = spectral.solve(rhs, basis, parameters)
        else:
            from global_candidate import predict
            pressure = predict(parameters, rhs, basis)
        if not np.isfinite(pressure).all():
            raise ValueError("nonfinite pressure prediction")
        prediction_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        directory = tempfile.TemporaryDirectory(prefix="pressure-", dir=args.scratch)
        for case, p in zip(entries, pressure):
            p.astype("<f8").tofile(Path(directory.name) / f'{case["spec"]["id"]}.bin')
        transfer_seconds = time.perf_counter() - begin
        location = directory.name
    else:
        location = "zero"
    native_method = ("raw" if args.method in ("analytic", "raw_learned") else
                     "sgs-pcg-v1" if args.method.endswith("sgs") else "jacobi-pcg-v1")
    command = [str(args.probe), str(args.dataset), native_method, location, args.scope]
    if quality:
        command.append("quality")
    begin = time.perf_counter()
    try:
        native = json.loads(subprocess.check_output(command, text=True, timeout=30))
    finally:
        child_seconds = time.perf_counter() - begin
        if directory is not None:
            begin = time.perf_counter()
            directory.cleanup()
            cleanup_seconds = time.perf_counter() - begin
    return {"method": args.method, "quality_pass": quality, "native": native,
            "worker_active_seconds": time.perf_counter() - start,
            "worker_stages_seconds": {"rhs_read_hash": input_seconds, "prediction": prediction_seconds,
                                      "prediction_write": transfer_seconds, "native_process_json": child_seconds,
                                      "prediction_cleanup": cleanup_seconds}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("weights", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("scope", choices=("development", "final"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    state = prepare(args.method, args.weights)
    setup = time.perf_counter() - start
    if args.once:
        print(json.dumps({"setup_seconds": setup, **run(args, state, False)}, allow_nan=False))
        return
    print(json.dumps({"status": "ready", "method": args.method, "setup_seconds": setup}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "stop":
            print(json.dumps({"status": "stopped"}), flush=True)
            return
        if request["op"] != "run":
            raise ValueError("worker request")
        print(json.dumps(run(args, state, request.get("quality", False)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
