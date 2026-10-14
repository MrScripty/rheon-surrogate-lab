"""Fixed CPU pressure experiment over captured, unchanged lab algorithms."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

from native import NativeLaunches, sha

RHEON_REVISION = "2209012bc5f902693d0f695ec1c2d854126af317"
ALGORITHMS = {
    "trajectory_cpu.py": "f8479f2667fd91fe83bf6c2d66fd1fb9cd9b60d960615b25f528ccac700ba039",
    "trajectory_experiment.py": "e1fd053f693730d39ded6a7ee50fd6e562bef37b2171bb7ed26acb770605e2f6",
    "trajectory_model.py": "83e93e75bbe05a8dfc8e8dc416b333d6d2e2614613ba45b4f77f99bb94f9a0b4",
    "trajectory_cpu_viewer.py": "f7501c6ab65d05b869b69d4ec472f6255e52d12e8cc4cd4ec5bba0abda9395e9",
    "verify_parallel_results.py": "9d4bd626bbd4352102c11765a47040a37e729a15d5074211f5e60b5bc9ab31ad",
}
METHODS = ("jacobi", "sgs", "learned")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(stage, completed, total, metrics=None):
    print(json.dumps({"format": "tuldok_progress_v1", "stage": stage,
                      "completed": completed, "total": total, "metrics": metrics or {}},
                     allow_nan=False), flush=True)


def read_config(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate configuration field")
            value[key] = item
        return value
    config = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    expected = {"schema": "rheon-tuldok-pressure-config-v1", "n": 32, "device": "cpu",
                "threads": 8, "training_seeds": list(range(100, 112)), "training_frames": 8,
                "validation_seeds": [500, 501], "validation_frames": 60,
                "test_seeds": [900, 901, 902, 903], "test_frames": 120, "repetitions": 3,
                "test_status": "previously exposed seeds; reproduction, not a fresh holdout",
                "physics": "uniform sealed unit box; prescribed forcing; passive tracer; fixed operator",
                "timing_scope": "resident CPU model; native startup, all simulation stages, inference, pipes, qualification, report parsing and native teardown; excludes imports/model setup/warmup/capture/rendering"}
    if config != expected or any(type(config[k]) is not int for k in
                                 ("n", "threads", "training_frames", "validation_frames", "test_frames", "repetitions")):
        raise ValueError("Unsupported pressure configuration; version a changed protocol")
    return config


def finite(value, maximum=math.inf, strict=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (value >= maximum if strict else value > maximum)):
        raise ValueError("Nonfinite or failed numerical acceptance")


def audit_comparison(result):
    config = result["config"]
    if result.get("status") != "verified":
        raise ValueError("Incomplete comparison")
    identities = [(repeat, seed, method) for repeat in range(3)
                  for index, seed in enumerate(range(900, 904))
                  for method in METHODS[(repeat + index) % 3:] + METHODS[:(repeat + index) % 3]]
    if [(r["repetition"], r["seed"], r["method"]) for r in result["records"]] != identities:
        raise ValueError("Timed trajectory coverage differs")
    residuals, divergence = [], []
    for record in result["records"]:
        finite(record["wall_ms_including_native_startup_teardown"])
        if record["wall_ms_including_native_startup_teardown"] <= 0 or len(record["reports"]) != 120:
            raise ValueError("Invalid timing/frame count")
        for frame, report in enumerate(record["reports"], 1):
            if report["frame"] != frame or not math.isclose(report["time"], frame / 60):
                raise ValueError("Frame/time order differs")
            finite(report["relative_residual"], 1e-8)
            finite(report["actual_divergence_max"], 1e-5)
            finite(report["courant"], 1)
            if record["method"] == "learned" and report["iterations"] != 0:
                raise ValueError("Learned path performed iterative corrections")
            residuals.append(report["relative_residual"])
            divergence.append(report["actual_divergence_max"])
    if (config["device"] != "cpu" or config["cuda_visible_devices"] != ""
            or config["native_threads_all_methods"] != 8 or config["torch_threads"] != 8
            or config["torch_interop_threads"] != 1 or not config["all_model_tensors_on_cpu"]
            or result["qualified_timed_frames"] != 4320
            or result["max_relative_residual"] != max(residuals)
            or result["max_divergence"] != max(divergence)):
        raise ValueError("Comparison scope/maxima differ")
    medians = {method: statistics.median(r["wall_ms_including_native_startup_teardown"]
               for r in result["records"] if r["method"] == method) for method in METHODS}
    if (result["median_120_frame_wall_ms"] != medians
            or result["speedup_vs_jacobi"] != medians["jacobi"] / medians["learned"]
            or result["speedup_vs_sgs"] != medians["sgs"] / medians["learned"]):
        raise ValueError("Timing arithmetic differs")
    if [case["seed"] for case in result["accuracy"]] != list(range(900, 904)):
        raise ValueError("Accuracy seed coverage differs")
    for case in result["accuracy"]:
        for method in ("jacobi", "sgs"):
            rows = case[method]["frames"]
            if len(rows) != 120:
                raise ValueError("Accuracy frame coverage differs")
            for quantity in ("velocity", "tracer"):
                key = quantity + "_relative_l2"
                for row in rows:
                    finite(row[key], 1e-5, strict=True)
                if case[method]["max_" + key] != max(row[key] for row in rows):
                    raise ValueError("Accuracy maxima differ")
    finite(result["python_peak_rss_bytes"], 8_000_000_000)
    return medians


def run(args):
    started = time.perf_counter()
    config = read_config(args.config)
    output = Path(args.output)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Output must be an existing empty directory")
    root = Path(__file__).resolve().parent
    scripts = root / "lab/scripts"
    for name, digest in ALGORITHMS.items():
        if sha(scripts / name) != digest:
            raise ValueError("Original numerical source changed: " + name)
    binary = root / "lab/target/release/rheon-trajectory-parallel"
    build = json.loads(Path(args.build_receipt).read_text())
    if (build.get("schema") != "rheon-tuldok-build-v1" or build.get("status") != "built"
            or build.get("rheon_revision") != RHEON_REVISION
            or build.get("lab_status") != "clean" or build.get("rheon_status") != "clean"
            or build.get("binary_sha256") != sha(binary)):
        raise ValueError("Native build receipt differs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["threads"])
    sys.path.insert(0, str(scripts))
    import numpy as np
    import torch
    import trajectory_experiment as exp
    import trajectory_cpu as cpu
    from trajectory_model import ModalPressure
    from trajectory_cpu_viewer import frames as viewer_frames
    if torch.__version__ != "2.8.0+cpu" or np.__version__ != "2.3.5":
        raise ValueError("Use the qualified Torch2.8.0+cpu/NumPy2.3.5 runtime")
    transport = NativeLaunches(binary, args.native_policy)
    exp.subprocess = transport
    exp.BINARY = binary
    source = {"binary_sha256": sha(binary), "rheon_commit": RHEON_REVISION,
              "source_sha256": {**ALGORITHMS, "worker.py": sha(__file__), "native.py": sha(root / "native.py")},
              "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
              "python_executable": sys.executable, "cuda_available": False}
    exp.provenance = lambda: source
    captured = {}
    original_rollout = exp.rollout
    def rollout(n, method, seed, frames, model=None, capture=False):
        record, states = original_rollout(n, method, seed, frames, model, capture)
        for report in record["reports"]:
            finite(report["relative_residual"], 1e-8)
            finite(report["actual_divergence_max"], 1e-5)
            finite(report["courant"], 1)
        if args.command == "compare" and capture:
            filename = f"{method}-{seed}.npz"
            np.savez_compressed(output / filename, rhs=np.stack([s["rhs"] for s in states]),
                                pressure=np.stack([s["pressure"] for s in states]),
                                tracer=np.stack([s["tracer"] for s in states]),
                                **{f"velocity_{d}": np.stack([s["velocity"][d] for s in states]) for d in range(3)})
            if seed == 900 and method in ("jacobi", "learned"):
                captured[method] = viewer_frames(record["reports"], states)
        progress("trajectory", len(transport.evidence), 48 if args.command == "compare" else 17)
        return record, states
    exp.rollout = rollout
    progress("initializing", 0, 1)
    with tempfile.TemporaryDirectory(prefix="rheon-project-") as temporary:
        stage = Path(temporary) / "run"
        if args.command == "train":
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            exp.train(SimpleNamespace(output=stage, device="cpu", n=32,
                                     training_frames=8, validation_frames=60))
            freeze = json.loads((stage / "freeze.json").read_text())
            failed = [item for item in transport.evidence if item.get("exit_code") != 0]
            if (len(failed) != 1 or failed[0]["exit_code"] != 1
                    or not re.search(r"(?:stored divergence|relative residual).*exceeds", failed[0]["stderr"])
                    or freeze["ablation"] != "untrained_unit_gains_rejected_by_physics_gate"):
                raise ValueError("Ablation did not fail at the native physics gate")
            for case in freeze["validation"]:
                finite(case["max_relative_velocity_l2"], 1e-5, strict=True)
                finite(case["max_relative_tracer_l2"], 1e-5, strict=True)
            checkpoint = torch.load(stage / "model.pt", weights_only=True, map_location="cpu")
            restored = ModalPressure(32, "cpu")
            restored.load_state_dict(checkpoint["state_dict"])
            with np.load(stage / "training.npz", allow_pickle=False) as arrays:
                fitted = ModalPressure(32, "cpu")
                fitted.fit(torch.from_numpy(arrays["rhs"]), torch.from_numpy(arrays["pressure"]))
                if not all(torch.equal(a, b) for a, b in zip(fitted.state_dict().values(), restored.state_dict().values(), strict=True)):
                    raise ValueError("Restored checkpoint differs from replayed supervised fit")
            finite(freeze["memory"]["python_peak_rss_bytes"], 20_000_000_000)
            metrics = [{"name": "training_pressure_relative_l2", "value": freeze["training"]["training_mean_relative_pressure_l2"], "unit": "ratio"}]
            for file in stage.iterdir():
                shutil.copyfile(file, output / file.name)
            write(output / "training-result.json", {"status": "verified", "freeze_sha256": sha(output / "freeze.json"),
                  "checkpoint_sha256": sha(output / "model.pt"), "restored_fit_equal": True,
                  "training": freeze["training"], "validation": freeze["validation"]})
        else:
            if not args.checkpoint or not args.freeze:
                raise ValueError("Comparison requires checkpoint and freeze inputs")
            stage.mkdir()
            shutil.copyfile(args.checkpoint, stage / "model.pt")
            shutil.copyfile(args.freeze, stage / "freeze.json")
            freeze = json.loads((stage / "freeze.json").read_text())
            if (freeze["n"] != 32 or freeze["test_frames"] != 120
                    or freeze["test_seeds"] != [900, 901, 902, 903]
                    or freeze["checkpoint_sha256"] != sha(stage / "model.pt")):
                raise ValueError("Checkpoint/freeze contract differs")
            previous = sys.argv
            try:
                sys.argv = ["trajectory_cpu.py", str(stage), str(stage / "comparison"),
                            "--threads", "8", "--repetitions", "3"]
                cpu.main()
            finally:
                sys.argv = previous
            result = json.loads((stage / "comparison/result.json").read_text())
            audit_comparison(result)
            for name in ("timing.json", "result.json"):
                shutil.copyfile(stage / "comparison" / name, output / name)
            shutil.copyfile(stage / "comparison/freeze.json", output / "comparison-freeze.json")
            shutil.copyfile(stage / "model.pt", output / "model.pt")
            shutil.copyfile(stage / "freeze.json", output / "input-freeze.json")
            worst = max(case[m]["max_velocity_relative_l2"] for case in result["accuracy"] for m in ("jacobi", "sgs"))
            write(output / "viewport-display.json", {"n": 32, "seed": 900, "dt": 1 / 60,
                  "device": "cpu", "torch": torch.__version__, "native_threads": 8, "reference": captured["jacobi"],
                  "learned": captured["learned"], "accuracy": {"max_relative_velocity_l2": worst},
                  "display": {"threshold": .025, "opacity_decimal_places": 5,
                              "accuracy_source": "full retained float32 fields, float64 norms"}})
            metrics = [{"name": "speedup_vs_" + m, "value": result["speedup_vs_" + m], "unit": "ratio"} for m in ("jacobi", "sgs")]
            metrics.append({"name": "max_velocity_relative_l2", "value": worst, "unit": "ratio"})
    write(output / "result-summary.json", {"format": "tuldok_result_summary_v1",
          "title": "CPU pressure " + args.command, "metrics": metrics, "checkpoint": "model.pt",
          "scope": ["Fixed sealed 32-cubed box; agreement with Rheon, not general physical truth",
                    "Previously exposed test seeds; no new holdout claim",
                    "New Torch2.8 CPU runtime; historical eightfold result is not guaranteed"]})
    write(output / "recipe-receipt.json", {"schema": "rheon-tuldok-receipt-v1", "status": "verified",
          "command": args.command, "config_sha256": sha(args.config),
          "build_receipt_sha256": sha(args.build_receipt), "source": source,
          "native_launches": transport.evidence, "whole_worker_wall_s": time.perf_counter() - started,
          "whole_worker_scope": "after worker imports, before final receipt write; includes model setup, accuracy retention, compression; not resident rollout speed",
          "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}})
    progress("publishing", 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "compare"))
    for name in ("config", "build-receipt", "native-policy", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--freeze")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
