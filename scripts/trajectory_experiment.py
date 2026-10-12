"""Training, held-out rollouts and complete CPU-Rheon/AI timing receipts.

Run with Pumas' managed PyTorch 2.14 interpreter. Data, checkpoints, timings,
full float32 trajectories and UI payload live in an exclusive output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

import numpy as np
import torch

from trajectory_model import ModalPressure

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/rheon-trajectory-probe"
RHEON = ROOT.parent / "Rheon"
TRAIN_SEEDS = tuple(range(100, 112))
VAL_SEEDS = (500, 501)
TEST_SEEDS = (900, 901, 902, 903)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n")


def provenance() -> dict:
    files = [*sorted((RHEON / "src").rglob("*.rs")),
             *sorted((ROOT / "probes/trajectory").rglob("*.rs")),
             ROOT / "probes/trajectory/Cargo.toml", ROOT / "probes/trajectory/Cargo.lock",
             Path(__file__), ROOT / "scripts/trajectory_model.py"]
    return {"binary_sha256": sha(BINARY), "source_sha256": {
        str(p.relative_to(ROOT.parent)): sha(p) for p in files},
        "rheon_commit": subprocess.check_output(
            ["git", "-C", str(RHEON), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "numpy": np.__version__,
        "python": os.sys.version, "python_executable": os.sys.executable,
        "cuda_available": torch.cuda.is_available()}


def exact_read(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            raise RuntimeError("native probe closed before complete reply")
        chunks.extend(part)
    return chunks


class Native:
    def __init__(self, n, method, seed):
        self.process = subprocess.Popen([str(BINARY), str(n), method, str(seed)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.n, self.method = n, method
        try:
            self.ready = self.read_json()
        except BaseException:
            self.close()
            raise

    def read_json(self):
        length, = struct.unpack("<I", exact_read(self.process.stdout, 4))
        if length > 65536:
            raise RuntimeError("oversized native reply")
        return json.loads(exact_read(self.process.stdout, length))

    def array(self, count, dtype):
        dtype = np.dtype(dtype)
        return np.frombuffer(exact_read(self.process.stdout, count * dtype.itemsize),
                             dtype=dtype).copy()

    def step(self, model=None, capture=False):
        start = time.perf_counter()
        self.process.stdin.write(bytes([2 if capture else 1]))
        self.process.stdin.flush()
        inference_ms = 0.0
        if self.method == "learned":
            rhs = self.array(self.n**3, "<f8")
            t0 = time.perf_counter()
            with torch.inference_mode():
                x = torch.from_numpy(rhs).to(model.gains.device)
                p = model(x).cpu().numpy()
            inference_ms = (time.perf_counter() - t0) * 1000
            self.process.stdin.write(p.astype("<f8", copy=False).tobytes())
            self.process.stdin.flush()
        report = self.read_json()
        arrays = None
        if capture:
            arrays = {"rhs": self.array(self.n**3, "<f8"),
                      "pressure": self.array(self.n**3, "<f8"),
                      "velocity": [self.array(count, "<f4") for count in self.ready["face_lengths"]],
                      "tracer": self.array(self.n**3, "<f4")}
        report["python_frame_wall_ms"] = (time.perf_counter() - start) * 1000
        report["model_transfer_inference_ms"] = inference_ms
        return report, arrays

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b"\0")
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            # The child can reject pressure between poll(), the quit flush,
            # and closing the buffered pipe. Its failure still propagates after
            # reaping the process; cleanup must not mask it or leak descriptors.
            pass
        self.process.wait(timeout=10)
        self.process.stdout.close()
        if self.process.returncode:
            raise RuntimeError(f"native trajectory failed: exit {self.process.returncode}")


def rollout(n, method, seed, frames, model=None, capture=False):
    start = time.perf_counter()
    native = Native(n, method, seed)
    reports, arrays = [], []
    try:
        for _ in range(frames):
            report, fields = native.step(model, capture)
            reports.append(report)
            if fields:
                arrays.append(fields)
    finally:
        native.close()
    return {"method": method, "seed": seed, "frames": frames,
            "wall_ms_including_native_startup_teardown": (time.perf_counter()-start)*1000,
            "reports": reports}, arrays


def relative(a, b):
    return float(np.linalg.norm(a.astype(np.float64)-b) / max(np.linalg.norm(b), 1e-30))


def compare(n, seed, frames, model):
    ref, teacher = rollout(n, "jacobi", seed, frames, capture=True)
    ai, predicted = rollout(n, "learned", seed, frames, model, capture=True)
    errors = []
    for truth, pred in zip(teacher, predicted, strict=True):
        tv = np.concatenate(truth["velocity"])
        pv = np.concatenate(pred["velocity"])
        tp = truth["pressure"] - truth["pressure"].mean()
        pp = pred["pressure"] - pred["pressure"].mean()
        errors.append({"relative_velocity_l2": relative(pv, tv),
                       "relative_tracer_l2": relative(pred["tracer"], truth["tracer"]),
                       "tracer_rmse": float(np.sqrt(np.mean((pred["tracer"]-truth["tracer"])**2))),
                       "centered_relative_pressure_l2": relative(pp, tp)})
    return {"seed": seed, "frames": frames, "errors": errors,
            "max_relative_velocity_l2": max(e["relative_velocity_l2"] for e in errors),
            "max_relative_tracer_l2": max(e["relative_tracer_l2"] for e in errors),
            "max_ai_divergence": max(r["actual_divergence_max"] for r in ai["reports"])}, ref, ai, teacher, predicted


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    initial_provenance = provenance()
    device = args.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = ModalPressure(args.n, device)
    rhs, pressure, manifest = [], [], []
    for seed in TRAIN_SEEDS:
        record, states = rollout(args.n, "jacobi", seed, args.training_frames, capture=True)
        rhs.extend(s["rhs"] for s in states)
        pressure.extend(s["pressure"] for s in states)
        manifest.append(record)
        print(f"generated training seed {seed}", flush=True)
    rhs, pressure = np.stack(rhs), np.stack(pressure)
    np.savez(args.output / "training.npz", rhs=rhs, pressure=pressure)
    start = time.perf_counter()
    fitted = model.fit(torch.from_numpy(rhs).to(device), torch.from_numpy(pressure).to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    fitted["training_wall_s"] = time.perf_counter()-start
    model.eval()
    torch.save({"schema": "rheon-modal-trajectory-v1", "n": args.n,
                "state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
                "training_seeds": TRAIN_SEEDS}, args.output / "model.pt")
    # Validation informs runtime choice and accuracy qualification, never weights.
    validation = []
    for seed in VAL_SEEDS:
        result, *_ = compare(args.n, seed, args.validation_frames, model)
        validation.append(result)
        print(f"validation seed {seed}: velocity error {result['max_relative_velocity_l2']:.3g}", flush=True)
    ablation = "unexpectedly_accepted"
    try:
        rollout(args.n, "learned", VAL_SEEDS[0], 1, ModalPressure(args.n, device))
    except RuntimeError:
        ablation = "untrained_unit_gains_rejected_by_physics_gate"
    memory = {"python_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0}
    # Freeze all test conditions before generating any test trajectory.
    freeze = {"schema": "rheon-trajectory-freeze-v1", "n": args.n, "device": device,
              "training_frames": args.training_frames, "validation_frames": args.validation_frames,
              "test_frames": 120, "test_seeds": TEST_SEEDS, "benchmark_repetitions": 5,
              "methods": ["jacobi", "sgs", "learned"], "checkpoint_sha256": sha(args.output / "model.pt"),
              "training_data_sha256": sha(args.output / "training.npz"),
              "source": initial_provenance, "training": fitted, "training_trajectories": manifest,
              "validation": validation, "ablation": ablation, "memory": memory,
              "limits": {"training_bytes": 20_000_000_000, "inference_bytes": 8_000_000_000},
              "scope": "fixed uniform sealed unit box, prescribed forcing, passive tracer; no free liquid surface"}
    write_json(args.output / "freeze.json", freeze)
    print(json.dumps({"fitted": fitted, "memory": memory, "ablation": ablation}, indent=2), flush=True)


def evaluate(args):
    freeze_path = args.output / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if sha(args.output / "model.pt") != freeze["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen training")
    current = provenance()
    for key in ("binary_sha256", "source_sha256", "torch"):
        if current[key] != freeze["source"][key]:
            raise ValueError(f"frozen provenance changed: {key}")
    destination = args.output / "evaluation"
    destination.mkdir(exist_ok=False)
    n, device = freeze["n"], freeze["device"]
    checkpoint = torch.load(args.output / "model.pt", weights_only=True, map_location=device)
    model = ModalPressure(n, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tests = []
    for seed in freeze["test_seeds"]:
        result, ref, ai, teacher, predicted = compare(n, seed, freeze["test_frames"], model)
        tests.append(result)
        for method, states in (("jacobi", teacher), ("learned", predicted)):
            np.savez_compressed(destination / f"{method}-{seed}.npz",
                tracer=np.stack([s["tracer"] for s in states]),
                pressure=np.stack([s["pressure"] for s in states]),
                **{f"velocity_{d}": np.stack([s["velocity"][d] for s in states]) for d in range(3)})
        if seed == freeze["test_seeds"][0]:
            # Full 3D scalar fields, x fastest. View never changes numeric data.
            write_json(destination / "viewport.json", {"n": n, "seed": seed, "dt": 1/60,
                "reference": [{"report": r, "tracer": s["tracer"].tolist()} for r,s in zip(ref["reports"], teacher)],
                "learned": [{"report": r, "tracer": s["tracer"].tolist()} for r,s in zip(ai["reports"], predicted)],
                "accuracy": result})
        print(f"test seed {seed}: velocity error {result['max_relative_velocity_l2']:.3g}", flush=True)
    # Warm the resident model; recorded separately from measured rollout calls.
    warm_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            model(torch.ones(n**3, dtype=torch.float64, device=device))
    if device == "cuda":
        torch.cuda.synchronize()
    warm_s = time.perf_counter() - warm_start
    records = []
    methods = freeze["methods"]
    for repetition in range(freeze["benchmark_repetitions"]):
        for index, seed in enumerate(freeze["test_seeds"]):
            order = methods[(repetition+index)%3:] + methods[:(repetition+index)%3]
            for method in order:
                record, _ = rollout(n, method, seed, freeze["test_frames"], model)
                record["repetition"] = repetition
                records.append(record)
        print(f"benchmark repetition {repetition+1} complete", flush=True)
    medians = {m: float(np.median([r["wall_ms_including_native_startup_teardown"]
                for r in records if r["method"] == m])) for m in methods}
    result = {"schema": "rheon-trajectory-result-v1", "freeze_sha256": sha(freeze_path),
              "checkpoint_sha256": freeze["checkpoint_sha256"], "accuracy": tests,
              "benchmark": records, "median_120_frame_wall_ms": medians,
              "speedup_vs_jacobi": medians["jacobi"]/medians["learned"],
              "speedup_vs_sgs": medians["sgs"]/medians["learned"],
              "timing_scope": "resident model; each rollout includes native startup, all kernels, inference, transfers, gates, report parsing and native teardown; excludes capture/rendering, Python/model initialization and explicit warmup",
              "model_warmup_s": warm_s,
              "python_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0}
    result["artifact_sha256"] = {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination / "result.json", result)
    print(json.dumps({k: result[k] for k in ("median_120_frame_wall_ms", "speedup_vs_jacobi", "speedup_vs_sgs")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--training-frames", type=int, default=8)
    parser.add_argument("--validation-frames", type=int, default=60)
    args = parser.parse_args()
    if not 8 <= args.n <= 64 or min(args.training_frames, args.validation_frames) < 1:
        parser.error("invalid grid or frame count")
    torch.set_num_threads(1)
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
