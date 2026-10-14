"""Cubic-cell capture adapter; frozen independent MAC oracle and solver gates."""
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import grid
import build_native

CUBIC_HERE = Path(__file__).resolve().parent

from contract import HERE, LAB, MIB, canonical, decode, fresh, load_module, read, require, reservation, sha, write

ORIGINAL = LAB / "experiments/global-pressure-v1"
sys.path.insert(0, str(ORIGINAL))
import prepare_dependency
import verify_capture as oracle

legacy = load_module("local_capture_process_primitives", ORIGINAL / "capture.py")


def capture_sources():
    return build_native.sources("capture")


def build(config, prepared, cargo, output):
    return build_native.build("capture", config, prepared, cargo, output)


def admit_build(config, binary):
    receipt = build_native.admit(binary)
    identity = receipt["identity"]
    require(identity["binding"]["experiment_sha256"] == sha(canonical(config))
            and identity["reservation_sha256"] == sha(canonical(grid.native_reservation(config)))
            and identity["parallel"] is False, "build_identity", "config/reservation/serial binding")
    return receipt


def generate(config, binary, output, fixture=False, observer=None, qualified_fixture=None):
    config = grid.config(config)
    if not fixture:
        require(qualified_fixture is not None, "fixture_required", "Qualify a one-frame fixture first")
        fixture_report = verify(qualified_fixture)
        require(fixture_report["status"] == "verified_fixture" and fixture_report["experiment_sha256"] == sha(canonical(config)), "fixture_identity", "same selected experiment")
    receipt = admit_build(config, binary)
    output = fresh(output)
    write(output / "experiment.json", config)
    grids = grid.plan(config)["grids"]
    entries = [next(e for e in grids if e["split"] == "train")] if fixture else grids
    manifest = {"schema": "rheon-cubic-pressure-capture-v1", "status": "candidate", "fixture": fixture, "experiment_sha256": sha(canonical(config)), "build_receipt": receipt, "trajectories": []}
    if not fixture:
        manifest['qualified_fixture_manifest_sha256'] = fixture_report['manifest_sha256']
    started = time.monotonic()
    payload_bytes = 0
    for entry in entries:
        tid = entry["trajectory_id"]
        directory = output / tid
        directory.mkdir()
        extents = entry["extents_m"]
        counts = entry["counts_xyz"]
        geometry = {"counts_xyz": counts, "extents_m": extents, "origin_m": [0.0] * 3, "density_kg_m3": 1.25, "boundary": "sealed_zero_normal_mac", "units": "SI"}
        trajectory = {"trajectory_id": tid, "reservation_entry": next(e for e in config["experiment"]["trajectories"] if e["trajectory_id"] == tid), "geometry": geometry, "initial": {}, "frames": []}
        frames = 1 if fixture else entry["frames"]
        request = {"preset": "smoke", "trajectory_id": tid, "frames": frames, "fixture": fixture}
        with (directory / "producer.stderr.log").open("xb") as stderr:
            with legacy.bounded_process([str(binary), "--produce"], config['experiment']['resources']['wall_seconds'] - (time.monotonic() - started), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env={**os.environ, "RAYON_NUM_THREADS": "1"}) as process:
                process.stdin.write(canonical(request))
                process.stdin.close()
                previous_pressure, previous_state = None, None
                for stage in range(frames + 2):
                    size = struct.unpack("<I", legacy.read_exact(process.stdout, 4))[0]
                    require(0 < size <= 65536, "producer_stream", "bounded header")
                    header = decode(legacy.read_exact(process.stdout, size))
                    expected = "initial" if stage == 0 else "complete" if stage == frames + 1 else "frame"
                    require(header["kind"] == expected, "producer_stream", "stage order")
                    if expected == "complete":
                        require(header["frames"] == frames, "producer_stream", "complete coverage")
                        continue
                    inventory = {}
                    for descriptor in header.pop("arrays"):
                        name = descriptor.pop("name")
                        allowed = {"pressure", "u", "v", "w", "tracer"} if stage == 0 else {"rhs", "pressure", "tracer", "pre_u", "pre_v", "pre_w", "corrected_u", "corrected_v", "corrected_w"}
                        require(name in allowed and name not in inventory and type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= MIB, "producer_stream", "bounded declared array")
                        payload_bytes += descriptor['bytes']
                        require(payload_bytes <= config['experiment']['resources']['max_output_bytes'], 'output_budget', 'Selected capture exceeds its payload byte budget')
                        raw = legacy.read_exact(process.stdout, descriptor["bytes"])
                        path = directory / f'{"initial" if stage == 0 else f"frame-{stage - 1:03d}"}-{name}.bin'
                        with path.open("xb") as stream:
                            stream.write(raw)
                        descriptor.update(path=path.relative_to(output).as_posix(), sha256=sha(raw))
                        inventory[name] = descriptor
                    if stage == 0:
                        trajectory["initial"] = inventory
                        previous_pressure = inventory["pressure"]["path"]
                        previous_state = {n: inventory[n]["path"] for n in ("u", "v", "w", "tracer")}
                    else:
                        header.update(arrays=inventory, previous_pressure=previous_pressure, previous_state=previous_state)
                        trajectory["frames"].append(header)
                        previous_pressure = inventory["pressure"]["path"]
                        previous_state = {n: inventory["corrected_" + n]["path"] for n in ("u", "v", "w")}
                        previous_state["tracer"] = inventory["tracer"]["path"]
                        if observer is not None:
                            preview = observer.frame(output, entry, stage - 1)
                            done = sum(len(t["frames"]) for t in manifest["trajectories"]) + stage
                            total = sum(1 if fixture else e["frames"] for e in entries)
                            observer.publish("Cubic-cell fixture" if fixture else "Cubic-cell capture", f"{tid}: {counts}; cell size {1/config['cells_per_metre']} m", done, total, preview=preview)
                require(process.stdout.read(1) == b"" and process.wait() == 0, "producer_stream", "complete terminal stream")
        manifest["trajectories"].append(trajectory)
    manifest["generation_elapsed_s"] = time.monotonic() - started
    write(output / "candidate.json", manifest)
    result = subprocess.run([sys.executable, str(CUBIC_HERE / "pipeline.py"), "verify", "--capture", str(output), "--candidate"], capture_output=True, timeout=600)
    require(result.returncode == 0, "capture_verification", result.stderr.decode()[-2000:])
    report = decode(result.stdout)
    write(output / "verification.json", report)
    manifest.update(status="fixture_complete" if fixture else "development_complete", verification_sha256=sha(read(output / "verification.json")))
    write(output / "manifest.json", manifest)
    return verify(output)


def verify(root, candidate=False):
    config = grid.config(decode(read(root / "experiment.json", MIB)))
    original = config["experiment"]
    raw = read(root / ("candidate.json" if candidate else "manifest.json"), 4 * MIB)
    manifest = decode(raw)
    require(manifest["schema"] == "rheon-cubic-pressure-capture-v1" and type(manifest["fixture"]) is bool
            and manifest["experiment_sha256"] == sha(canonical(config)), "capture_identity", "selected experiment")
    require(manifest["status"] == ("candidate" if candidate else "fixture_complete" if manifest["fixture"] else "development_complete"), "capture_status", "complete eligible capture")
    require(manifest["build_receipt"]["status"] == "qualified_build"
            and manifest["build_receipt"]["identity"]["binding"]["source_sha256"] == capture_sources()
            and manifest["build_receipt"]["identity"]["reservation_sha256"] == sha(canonical(grid.native_reservation(config))), "capture_source", "measured source/reservation")
    entries = [next(e for e in original["trajectories"] if e["split"] == "train")] if manifest["fixture"] else original["trajectories"]
    require([t["reservation_entry"] for t in manifest["trajectories"]] == entries, "capture_coverage", "all selected whole parents in order")
    reports, seen, paths = [], {}, set()
    for trajectory, entry in zip(manifest["trajectories"], entries):
        tid = entry["trajectory_id"]
        require(trajectory["trajectory_id"] == tid, "trajectory_identity", tid)
        geometry = trajectory["geometry"]
        extents = original["geometries_extents_m"][entry["geometry"]]
        counts = grid.counts(extents, config["cells_per_metre"])
        require(geometry == {"counts_xyz": counts, "extents_m": extents, "origin_m": [0.0] * 3, "density_kg_m3": 1.25, "boundary": "sealed_zero_normal_mac", "units": "SI"}, "geometry", tid)
        h = [1 / config["cells_per_metre"]] * 3
        initial = trajectory["initial"]
        require(set(initial) == {"pressure", "u", "v", "w", "tracer"}, "array_inventory", tid)
        def array(inventory, name):
            cell = name in ("pressure", "rhs", "tracer")
            dims = counts if cell else oracle.shape(("u", "v", "w").index(name[-1]), counts)
            dtype = "<f8" if name in ("pressure", "rhs") else "<f4"
            units = "Pa" if name == "pressure" else "m3/s2" if name == "rhs" else "dimensionless" if name == "tracer" else "m/s"
            path = inventory[name]["path"]
            require(path not in paths, "payload_path", "one descriptor per array")
            paths.add(path)
            return oracle.load_array(root, inventory[name], dims, dtype, units)
        require(all(x == 0 for x in array(initial, "pressure")), "initial_state", "zero pressure")
        for name in ("u", "v", "w"):
            require(all(x == 0 for x in array(initial, name)), "initial_state", "zero velocity")
        tracer = array(initial, "tracer")
        width = 0.08 * min(extents)
        expected = [oracle.f32(0.25 * __import__("math").exp(-sum(((p + 0.5) * h[d] - extents[d] * 0.5) ** 2 for d, p in enumerate((x, y, z))) / (2 * width * width))) for z in range(counts[2]) for y in range(counts[1]) for x in range(counts[0])]
        require(all(abs(a - b) <= 2e-7 * max(abs(a), abs(b), 1e-30) for a, b in zip(tracer, expected)), "initial_tracer", tid)
        frames = trajectory["frames"]
        require(len(frames) == (1 if manifest["fixture"] else 24), "frame_coverage", tid)
        previous_pressure = initial["pressure"]["path"]
        previous_state = {n: initial[n]["path"] for n in ("u", "v", "w", "tracer")}
        for index, frame in enumerate(frames):
            require(type(frame["index"]) is int and frame["index"] == index and frame["time_s"] == (index + 1) * entry["dt_s"] and frame["forcing_time_s"] == index * entry["dt_s"] and frame["previous_pressure"] == previous_pressure and frame["previous_state"] == previous_state, "history_lineage", tid)
            arrays = frame["arrays"]
            require(set(arrays) == {"rhs", "pressure", "tracer", "pre_u", "pre_v", "pre_w", "corrected_u", "corrected_v", "corrected_w"}, "array_inventory", tid)
            fields = {n: array(arrays, n) for n in arrays}
            metrics = oracle.physics([fields["pre_" + n] for n in ("u", "v", "w")], fields["rhs"], fields["pressure"], [fields["corrected_" + n] for n in ("u", "v", "w")], counts, h, 1.25, entry["dt_s"])
            reports.append({"trajectory_id": tid, "frame": index, **metrics})
            previous_pressure = arrays["pressure"]["path"]
            previous_state = {n: arrays["corrected_" + n]["path"] for n in ("u", "v", "w")}
            previous_state["tracer"] = arrays["tracer"]["path"]
        fingerprint = oracle.trajectory_fingerprint(geometry, initial, frames)
        oracle.admit_lineage(seen, fingerprint, entry["split"], tid)
    if not candidate:
        require(sha(read(root / "verification.json", 2 * MIB)) == manifest["verification_sha256"], "verification_identity", "retained verifier report")
    return {"status": "verified_fixture" if manifest["fixture"] else "verified_development", "manifest_sha256": sha(raw), "experiment_sha256": sha(canonical(config)), "trajectories": len(entries), "frames": len(reports), "frame_metrics": reports}
