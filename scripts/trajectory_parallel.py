"""Check CPU thread tuning on validation, then benchmark the frozen model.

This is a separate CPU implementation/runtime comparison. It never changes
weights or uses test errors to select a model. It uses the same native source.
"""
import json
import os
from pathlib import Path
import statistics
import time

import torch

import trajectory_experiment as experiment
from trajectory_model import ModalPressure


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    destination = args.run / "parallel"
    destination.mkdir(exist_ok=False)
    freeze = json.loads((args.run / "freeze.json").read_text())
    if experiment.sha(args.run / "model.pt") != freeze["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    torch.set_num_threads(1)
    experiment.BINARY = experiment.ROOT / "target/release/rheon-trajectory-parallel"
    model = ModalPressure(freeze["n"], freeze["device"])
    checkpoint = torch.load(args.run / "model.pt", weights_only=True, map_location=freeze["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tuning = []
    for threads in (1, 2, 4, 8, 16):
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        for method in ("jacobi", "sgs", "learned"):
            times = []
            for _ in range(2):
                record, _ = experiment.rollout(freeze["n"], method, 500, 8, model)
                times.append(record["wall_ms_including_native_startup_teardown"])
            tuning.append({"threads": threads, "method": method, "median_8_frame_wall_ms": statistics.median(times)})
            print(tuning[-1], flush=True)
    chosen = {m: min([r for r in tuning if r["method"] == m], key=lambda r:r["median_8_frame_wall_ms"])["threads"]
              for m in ("jacobi", "sgs", "learned")}
    # Frozen before replaying test seeds with the chosen execution settings.
    config = {"checkpoint_sha256": freeze["checkpoint_sha256"], "binary_sha256": experiment.sha(experiment.BINARY),
              "tuning": tuning, "chosen_threads": chosen, "test_seeds": freeze["test_seeds"],
              "test_frames": 120, "repetitions": 3, "device": freeze["device"],
              "cpu": os.cpu_count(), "scope": "CPU thread tuning uses only seed 500; model frozen"}
    experiment.write_json(destination / "freeze.json", config)
    records = []
    methods = ["jacobi", "sgs", "learned"]
    start = time.perf_counter()
    for repeat in range(config["repetitions"]):
        for index, seed in enumerate(config["test_seeds"]):
            offset=(repeat+index)%3
            for method in methods[offset:]+methods[:offset]:
                os.environ["RAYON_NUM_THREADS"] = str(chosen[method])
                record, _ = experiment.rollout(freeze["n"], method, seed, 120, model)
                record["threads"] = chosen[method]
                record["repetition"] = repeat
                records.append(record)
        print(f"parallel repetition {repeat+1} complete", flush=True)
    medians = {m: statistics.median([r["wall_ms_including_native_startup_teardown"] for r in records if r["method"] == m]) for m in methods}
    result = {"config": config, "records": records, "median_120_frame_wall_ms": medians,
              "speedup_vs_jacobi": medians["jacobi"]/medians["learned"],
              "speedup_vs_sgs": medians["sgs"]/medians["learned"],
              "benchmark_wall_s": time.perf_counter()-start}
    experiment.write_json(destination / "result.json", result)
    print({"medians": medians,"speedup_vs_jacobi":result["speedup_vs_jacobi"],"speedup_vs_sgs":result["speedup_vs_sgs"]}, flush=True)


if __name__ == "__main__":
    main()
