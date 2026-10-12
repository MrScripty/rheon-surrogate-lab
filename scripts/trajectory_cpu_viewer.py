"""Capture a CPU-only reference/model pair for the existing 3D timeline viewer."""
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

import trajectory_experiment as exp
from trajectory_model import ModalPressure
from verify_parallel_results import metrics


def frames(reports, states):
    return [{"report": r, "voxels": [[i, round(float(v), 5)]
             for i, v in enumerate(s["tracer"]) if v >= 0.025]}
            for r, s in zip(reports, states, strict=True)]


def main():
    run, destination = map(Path, sys.argv[1:])
    result = json.loads((destination / "result.json").read_text())
    config = result["config"]
    assert result["status"] == "verified" and config["device"] == "cpu"
    os.environ["RAYON_NUM_THREADS"] = str(config["native_threads_all_methods"])
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(1)
    exp.BINARY = exp.ROOT / "target/release/rheon-trajectory-parallel"
    assert exp.sha(exp.BINARY) == config["binary_sha256"]
    assert exp.sha(run / "model.pt") == config["checkpoint_sha256"]
    model = ModalPressure(config["n"], "cpu")
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True, map_location="cpu")["state_dict"])
    model.eval()
    seed = config["seeds"][0]
    ref, truth = exp.rollout(config["n"], "jacobi", seed, config["frames"], capture=True)
    ai, guess = exp.rollout(config["n"], "learned", seed, config["frames"], model, capture=True)
    accuracy = metrics(truth, guess)
    assert accuracy["max_velocity_relative_l2"] < 1e-5
    for method, states in (("jacobi", truth), ("learned", guess)):
        np.savez_compressed(destination/f"viewer-{method}-{seed}.npz",
            tracer=np.stack([s["tracer"] for s in states]),
            pressure=np.stack([s["pressure"] for s in states]),
            **{f"velocity_{d}": np.stack([s["velocity"][d] for s in states]) for d in range(3)})
    data = {"n": config["n"], "seed": seed, "dt": 1/60, "device": "cpu",
            "native_threads": config["native_threads_all_methods"],
            "reference": frames(ref["reports"], truth), "learned": frames(ai["reports"], guess),
            "accuracy": {"max_relative_velocity_l2": accuracy["max_velocity_relative_l2"]},
            "display": {"threshold": 0.025, "opacity_decimal_places": 5,
                        "accuracy_source": "full stored float32 fields; float64 norms"}}
    (destination/"viewport-display.json").write_text(json.dumps(data, separators=(",", ":"))+"\n")
    exp.write_json(destination/"viewport-receipt.json", {
        "device": "cpu", "accuracy": accuracy,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "artifact_sha256": {p.name: exp.sha(p) for p in destination.glob("viewer-*.npz")},
        "display_sha256": exp.sha(destination/"viewport-display.json")})
    print(f"CPU-only viewport captured; maximum velocity error {accuracy['max_velocity_relative_l2']:.3g}")


if __name__ == "__main__":
    main()
