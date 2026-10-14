"""Cold accuracy audit and existing-viewer installation for a completed recipe."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from trajectory_model import ModalPressure
    sys.path.insert(0, str(ROOT / "tuldok"))
    from worker import audit_comparison
    directory = Path(directory)
    receipt = json.loads((directory / "recipe-receipt.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("command") != "compare":
        raise ValueError("Not a completed comparison receipt")
    for name, digest in receipt["artifact_sha256"].items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError("Retained artifact differs: " + name)
    result = json.loads((directory / "result.json").read_text())
    audit_comparison(result)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
    model = ModalPressure(32, "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if (checkpoint.get("schema") != "rheon-modal-trajectory-v1" or checkpoint.get("n") != 32
            or any(t.dtype != torch.float64 or not torch.isfinite(t).all() for t in model.state_dict().values())
            or not (model.gains > 0).all()):
        raise ValueError("Invalid restored modal checkpoint")
    expected = {"rhs": (np.dtype("<f8"), (120, 32**3)),
                "pressure": (np.dtype("<f8"), (120, 32**3)),
                "tracer": (np.dtype("<f4"), (120, 32**3)),
                **{f"velocity_{d}": (np.dtype("<f4"), (120, 33*32*32)) for d in range(3)}}
    worst_velocity = worst_tracer = worst_replay = 0.0
    for case in result["accuracy"]:
        seed = case["seed"]
        loaded = {}
        for method in ("learned", "jacobi", "sgs"):
            with np.load(directory / f"{method}-{seed}.npz", allow_pickle=False) as archive:
                if set(archive.files) != set(expected):
                    raise ValueError("Full field inventory differs")
                arrays = {key: archive[key] for key in expected}
            for key, (dtype, shape) in expected.items():
                array = arrays[key]
                if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError("Invalid retained field: " + key)
            if not ((arrays["tracer"] >= 0) & (arrays["tracer"] <= 1)).all():
                raise ValueError("Tracer outside accepted bounds")
            loaded[method] = arrays
        guess = loaded["learned"]
        for frame in range(120):
            with torch.inference_mode():
                replay = model(torch.from_numpy(guess["rhs"][frame])).numpy()
            error = np.linalg.norm(replay - guess["pressure"][frame]) / max(np.linalg.norm(guess["pressure"][frame]), 1e-30)
            if not math.isfinite(error) or error >= 1e-12:
                raise ValueError("Restored pressure predictions differ from retained native input")
            worst_replay = max(worst_replay, float(error))
            predicted_velocity = np.concatenate([guess[f"velocity_{d}"][frame] for d in range(3)]).astype(np.float64)
            for method in ("jacobi", "sgs"):
                truth = loaded[method]
                velocity = np.concatenate([truth[f"velocity_{d}"][frame] for d in range(3)]).astype(np.float64)
                tracer = truth["tracer"][frame].astype(np.float64)
                velocity_error = float(np.linalg.norm(predicted_velocity - velocity) / np.linalg.norm(velocity))
                tracer_error = float(np.linalg.norm(guess["tracer"][frame].astype(np.float64) - tracer) / np.linalg.norm(tracer))
                row = case[method]["frames"][frame]
                if (not math.isclose(velocity_error, row["velocity_relative_l2"], rel_tol=1e-12, abs_tol=1e-20)
                        or not math.isclose(tracer_error, row["tracer_relative_l2"], rel_tol=1e-12, abs_tol=1e-20)
                        or velocity_error >= 1e-5 or tracer_error >= 1e-5):
                    raise ValueError("Cold full-field accuracy differs from published metrics")
                worst_velocity = max(worst_velocity, velocity_error)
                worst_tracer = max(worst_tracer, tracer_error)
    return {"status": "verified", "result_sha256": sha(directory / "result.json"),
            "receipt_sha256": sha(directory / "recipe-receipt.json"), "replayed_frames": 480,
            "qualified_timed_frames": 4320, "worst_velocity_relative_l2": worst_velocity,
            "worst_tracer_relative_l2": worst_tracer, "worst_restored_pressure_relative_l2": worst_replay,
            "scope": "Cold full-field norms and restored CPU pressure prediction equality; fixed box/exposed seeds"}


def install_viewer(directory, run_name):
    if not run_name or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_name):
        raise ValueError("Viewer run name must contain letters, digits, underscore or dash")
    # Match the unchanged viewer's existing CPU replay layout. Only this new
    # ignored directory is created; no historical run is replaced.
    destination = ROOT / "runs" / run_name / "cpu-only-8threads"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("viewport-display.json", "result.json", "comparison-freeze.json", "recipe-receipt.json"):
        shutil.copyfile(Path(directory) / name, destination / name)
    return "/viewer/?run=" + run_name + "&compute=cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--viewer-run")
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Audit receipt already exists")
    result = audit(args.directory)
    if args.viewer_run:
        result["viewer_url"] = install_viewer(args.directory, args.viewer_run)
    args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
