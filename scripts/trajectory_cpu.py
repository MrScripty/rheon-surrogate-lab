"""CPU-only, equal-thread comparison using the frozen trained pressure model.

All three paths use the same Rheon binary and native thread limit. PyTorch
inference uses that same CPU thread limit; CUDA devices are hidden before import.
No fitting, checkpoint changes, or test-based parameter selection occurs.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import time

# This experiment intentionally excludes GPU execution, including CUDA startup.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

import trajectory_experiment as exp
from trajectory_model import ModalPressure
from verify_parallel_results import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.threads <= 32 or not 1 <= args.repetitions <= 10:
        parser.error("threads must be 1..32 and repetitions 1..10")
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.run / "freeze.json").read_text())
    checkpoint_path = args.run / "model.pt"
    if exp.sha(checkpoint_path) != original["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    exp.BINARY = exp.ROOT / "target/release/rheon-trajectory-parallel"
    start = time.perf_counter()
    model = ModalPressure(original["n"], "cpu")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    model_setup_s = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            model(torch.ones(original["n"]**3, dtype=torch.float64))
    warmup_s = time.perf_counter() - start
    config = {"schema": "rheon-cpu-only-comparison-v1", "device": "cpu",
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "all_model_tensors_on_cpu": True, "n": original["n"],
              "native_threads_all_methods": args.threads,
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": 1,
              "seeds": original["test_seeds"], "frames": original["test_frames"],
              "repetitions": args.repetitions, "torch": torch.__version__,
              "python_executable": os.sys.executable,
              "checkpoint_sha256": exp.sha(checkpoint_path),
              "binary_sha256": exp.sha(exp.BINARY),
              "source_sha256": {p.name: exp.sha(p) for p in [Path(__file__),
                   exp.ROOT/"scripts/trajectory_experiment.py", exp.ROOT/"scripts/trajectory_model.py"]},
              "model_setup_s": model_setup_s, "warmup_s": warmup_s,
              "timing_scope": "resident CPU model; native startup, all simulation stages, CPU inference, pipes, qualification, report parsing and native teardown; excludes imports/model setup/warmup/capture/rendering",
              "test_status": "previously exposed seeds; backend/hardware comparison, no retraining"}
    exp.write_json(args.output / "freeze.json", config)
    records = []
    methods = ["jacobi", "sgs", "learned"]
    for repeat in range(args.repetitions):
        for index, seed in enumerate(config["seeds"]):
            offset = (repeat + index) % 3
            for method in methods[offset:] + methods[:offset]:
                record, _ = exp.rollout(config["n"], method, seed, config["frames"], model)
                record["repetition"] = repeat
                records.append(record)
                print(f"CPU {method} seed {seed} repeat {repeat+1}: "
                      f"{record['wall_ms_including_native_startup_teardown']/1000:.3f} s", flush=True)
    medians = {m: statistics.median(r["wall_ms_including_native_startup_teardown"]
               for r in records if r["method"] == m) for m in methods}
    maximum_residual = max(f["relative_residual"] for r in records for f in r["reports"])
    maximum_divergence = max(f["actual_divergence_max"] for r in records for f in r["reports"])
    assert maximum_residual <= 1e-8 and maximum_divergence <= 1e-5
    assert all(f["iterations"] == 0 for r in records if r["method"] == "learned" for f in r["reports"])
    result = {"config": config, "records": records, "median_120_frame_wall_ms": medians,
              "speedup_vs_jacobi": medians["jacobi"]/medians["learned"],
              "speedup_vs_sgs": medians["sgs"]/medians["learned"],
              "qualified_timed_frames": sum(len(r["reports"]) for r in records),
              "max_relative_residual": maximum_residual, "max_divergence": maximum_divergence,
              "python_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    # Publish timing before the separately captured accuracy replay. Partial work
    # cannot masquerade as a fully verified result.
    exp.write_json(args.output / "timing.json", result)
    accuracy = []
    for seed in config["seeds"]:
        _, predicted = exp.rollout(config["n"], "learned", seed, config["frames"], model, capture=True)
        case = {"seed": seed}
        for method in ("jacobi", "sgs"):
            _, teacher = exp.rollout(config["n"], method, seed, config["frames"], capture=True)
            case[method] = metrics(teacher, predicted)
            assert case[method]["max_velocity_relative_l2"] < 1e-5
            assert case[method]["max_tracer_relative_l2"] < 1e-5
            print(f"CPU accuracy {seed} / {method}: "
                  f"{case[method]['max_velocity_relative_l2']:.3g}", flush=True)
        accuracy.append(case)
    result["accuracy"] = accuracy
    result["status"] = "verified"
    result["python_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    exp.write_json(args.output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("median_120_frame_wall_ms", "speedup_vs_jacobi", "speedup_vs_sgs", "max_relative_residual")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
