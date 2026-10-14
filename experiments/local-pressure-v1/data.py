"""Fresh lossless packets; numeric parsing reuses the retained bounded reader."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from contract import HERE, LAB, MIB, canonical, decode, experiment, fresh, load_module, read, require, sha, write


def primitives():
    local = HERE / "original_data.py"
    path = local if local.exists() else LAB / "experiments/global-pressure-v1/training/data.py"
    return load_module("local_packet_numeric_primitives", path)


@dataclass
class Dataset:
    train: dict
    validation: dict
    train_rows: list
    validation_rows: list
    config: dict
    origin: dict
    splits: dict


NAMES = ("train-0.npz", "train-1.npz", "validation.npz")


def pack(root, binary, output):
    import capture
    report = capture.verify(root)
    require(report["status"] == "verified_development", "fixture_ineligible", "one-frame fixture cannot train")
    config = experiment(decode(read(root / "experiment.json", MIB)))
    origin_raw = read(root / "manifest.json", 4 * MIB)
    origin = decode(origin_raw)
    import native
    before = native.admit(binary)
    output = fresh(output)
    reader = primitives()
    trains = [t for t in origin["trajectories"] if t["reservation_entry"]["split"] == "train"]
    validation = [t for t in origin["trajectories"] if t["reservation_entry"]["split"] == "validation"]
    groups = (trains[:len(trains) // 2], trains[len(trains) // 2:], validation)
    split = {"schema": "rheon-local-pressure-packet-v1", "experiment": config, "capture_manifest_sha256": sha(origin_raw), "shards": {}, "native_residual_origin": {"native_build_receipt": before, "capture_manifest_sha256": sha(origin_raw), "rows": []}}
    parent_index = {t["trajectory_id"]: i for i, t in enumerate(origin["trajectories"])}
    for name, trajectories in zip(NAMES, groups):
        rows = len(trajectories) * 24
        arrays = {field: np.empty(shape, dtype=dtype) for field, (shape, dtype, _) in reader.array_spec(rows).items()}
        row_map = []
        fingerprints = []
        for local, trajectory in enumerate(trajectories):
            geometry = trajectory["geometry"]
            for frame in range(24):
                index = local * 24 + frame
                row = reader.original_row(trajectory, frame)
                for field in ("rhs", "reference", "base"):
                    arrays[field][index] = reader.read_raw(root, row["lineage"][field])
                request = {"schema": "rheon-global-pressure-native-residual-v1", "counts_xyz": [16] * 3, "spacing_m": [float(v) / 16 for v in geometry["extents_m"]], "density_kg_m3": 1.25, "original_rhs": arrays["rhs"][index].reshape(-1).tolist(), "base_pressure": arrays["base"][index].reshape(-1).tolist()}
                response, _ = native.call(binary, "--residual", request, timeout=10)
                require(response["schema"] == request["schema"] and response["identity"] == before["identity"], "native_residual_identity", row["row_id"])
                residual = np.asarray(response["residual"], dtype="<f8")
                require(residual.shape == (4096,) and np.isfinite(residual).all(), "native_residual", "finite full native residual")
                arrays["native_input_residual"][index] = residual.reshape(16, 16, 16)
                split["native_residual_origin"]["rows"].append({"row_id": row["row_id"], "sha256": sha(residual.tobytes())})
                arrays["spacing"][index] = np.asarray(geometry["extents_m"]) / 16
                arrays["extents"][index] = geometry["extents_m"]
                arrays["origin"][index] = geometry["origin_m"]
                arrays["rho"][index] = 1.25
                arrays["dt"][index] = trajectory["reservation_entry"]["dt_s"]
                arrays["frame"][index] = frame
                arrays["trajectory"][index] = parent_index[trajectory["trajectory_id"]]
                row_map.append(row)
            fingerprints.append(reader.packed_fingerprint(arrays, local * 24, trajectory))
        with (output / name).open("xb") as stream:
            np.savez(stream, **arrays)
        raw = read(output / name)
        split["shards"][name] = {"split": "validation" if name == NAMES[2] else "train", "trajectory_ids": [t["trajectory_id"] for t in trajectories], "bytes": len(raw), "sha256": sha(raw), "arrays": {n: {"shape": list(a.shape), "dtype": a.dtype.str, "bytes": a.nbytes, "sha256": sha(a.tobytes())} for n, a in arrays.items()}, "rows": row_map}
        split.setdefault("fingerprints", {})[name] = fingerprints
    require(native.admit(binary) == before and read(root / "manifest.json", 4 * MIB) == origin_raw, "pack_drift", "native/capture unchanged")
    for name, raw in (("origin.json", origin_raw), ("splits.json", canonical(split))):
        with (output / name).open("xb") as stream:
            stream.write(raw)
    files = {name: {"sha256": sha(read(output / name)), "bytes": len(read(output / name))} for name in (*NAMES, "splits.json", "origin.json")}
    dataset = load(output)
    write(output / "dataset-receipt.json", {"schema": "rheon-local-pressure-dataset-receipt-v1", "status": "verified_lossless_development", "experiment_sha256": sha(canonical(config)), "capture_manifest_sha256": sha(origin_raw), "native_residual_origin_sha256": sha(canonical(split["native_residual_origin"])), "reader_source_sha256": sha(read(HERE / "data.py", MIB)), "files": files})
    return {"status": "verified_lossless_development", "train_rows": len(dataset.train_rows), "validation_rows": len(dataset.validation_rows), "receipt": str(output / "dataset-receipt.json")}


def load(root=None, *, blobs=None, expected_residual=None):
    reader = primitives()
    if blobs is None:
        blobs = {n: read(root / n, 4 * MIB if n in ("splits.json", "origin.json") else 32 * MIB) for n in (*NAMES, "splits.json", "origin.json")}
    require(sum(map(len, blobs.values())) <= 256 * MIB, "input_budget", "bounded packet")
    split = decode(blobs["splits.json"])
    require(split["schema"] == "rheon-local-pressure-packet-v1", "packet_schema", "fresh packet")
    config = experiment(split["experiment"])
    origin = decode(blobs["origin.json"])
    require(origin["schema"] == "rheon-local-pressure-capture-v1" and origin["status"] == "development_complete" and origin["fixture"] is False and sha(blobs["origin.json"]) == split["capture_manifest_sha256"] and origin["experiment_sha256"] == sha(canonical(config)), "origin_identity", "complete measured capture")
    require([t["reservation_entry"] for t in origin["trajectories"]] == config["trajectories"], "whole_parent_coverage", "config/capture agree")
    anchor = split["native_residual_origin"]
    require(anchor["capture_manifest_sha256"] == split["capture_manifest_sha256"] and (expected_residual is None or sha(canonical(anchor)) == expected_residual), "residual_anchor", "selected native origin")
    trajectories = {t["trajectory_id"]: t for t in origin["trajectories"]}
    parent_index = {t["trajectory_id"]: i for i, t in enumerate(origin["trajectories"])}
    all_arrays, all_rows, seen_ids, seen_fingerprints, residual_rows = [], [], set(), {}, []
    for name in NAMES:
        declaration = split["shards"][name]
        ids = declaration["trajectory_ids"]
        require(ids and len(ids) == len(set(ids)) and not seen_ids.intersection(ids), "parent_overlap", "whole parents in exactly one shard")
        seen_ids.update(ids)
        rows = len(ids) * 24
        arrays = reader.load_npz(blobs[name], declaration, rows)
        expected_split = "validation" if name == NAMES[2] else "train"
        expected_rows, fingerprints = [], []
        for local, tid in enumerate(ids):
            require(tid in trajectories and trajectories[tid]["reservation_entry"]["split"] == expected_split and declaration["split"] == expected_split, "split_lineage", tid)
            trajectory = trajectories[tid]
            for frame in range(24):
                i = local * 24 + frame
                row = reader.original_row(trajectory, frame)
                expected_rows.append(row)
                require(int(arrays["frame"][i]) == frame and int(arrays["trajectory"][i]) == parent_index[tid], "row_mapping", row["row_id"])
                for field in ("rhs", "base", "reference"):
                    require(sha(arrays[field][i].tobytes()) == row["lineage"][field]["sha256"], "raw_numeric_lineage", row["row_id"])
                geometry = trajectory["geometry"]
                for field, expected in {"spacing": np.asarray(geometry["extents_m"]) / 16, "extents": geometry["extents_m"], "origin": geometry["origin_m"], "rho": 1.25, "dt": trajectory["reservation_entry"]["dt_s"]}.items():
                    require(np.array_equal(arrays[field][i], expected), "physical_lineage", field)
                require(arrays["base"][i].flat[0] == arrays["reference"][i].flat[0] == 0, "gauge", row["row_id"])
                residual_rows.append({"row_id": row["row_id"], "sha256": sha(arrays["native_input_residual"][i].tobytes())})
            fingerprint = reader.packed_fingerprint(arrays, local * 24, trajectory)
            require(fingerprint not in seen_fingerprints or seen_fingerprints[fingerprint] == expected_split, "cross_split_duplicate", tid)
            seen_fingerprints[fingerprint] = expected_split
            fingerprints.append(fingerprint)
        require(expected_rows == declaration["rows"] and fingerprints == split["fingerprints"][name], "row_lineage", name)
        all_arrays.append(arrays); all_rows.append(expected_rows)
    require(seen_ids == set(trajectories) and residual_rows == anchor["rows"], "population_coverage", "complete residual/parent mapping")
    train = {field: np.concatenate([all_arrays[0][field], all_arrays[1][field]]) for field in reader.FIELDS}
    return Dataset(train, all_arrays[2], all_rows[0] + all_rows[1], all_rows[2], config, origin, split)
