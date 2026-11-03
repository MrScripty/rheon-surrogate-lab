#!/usr/bin/env python3
"""Lossless development packing and a standalone, bounded numeric recipe reader.

Only ``pack`` invokes the existing independent capture verifier in a fresh process.
``load_data`` uses five flat files and NumPy/stdlib; old provenance paths are opaque.
The origin's full-physics fingerprint is an attestation, while this reader verifies
original field bytes and recomputes the packed numeric whole-history fingerprint.
"""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np


MIB = 1024 * 1024
FILE_LIMIT = 32 * MIB
INPUT_LIMIT = 256 * MIB
SPLITS_LIMIT = 256 * 1024
ORIGIN_SHA256 = "dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d"
SOURCE_COMMIT = "56b2507b763fe603b0206cd23592d488a58b30d0"
NATIVE_BASE = "2209012bc5f902693d0f695ec1c2d854126af317"
FRAMES = 24
COUNTS = (16, 16, 16)
SHARDS = ("train-0.npz", "train-1.npz", "validation.npz")
CELL_FIELDS = ("rhs", "base", "reference", "native_input_residual")
FLOAT_FIELDS = CELL_FIELDS + ("spacing", "extents", "origin", "rho", "dt")
INT_FIELDS = ("frame", "trajectory")
FIELDS = FLOAT_FIELDS + INT_FIELDS


class DataError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition, code, detail):
    if not condition:
        raise DataError(code, detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "json_duplicate", key)
            result[key] = value
        return result

    def number(text):
        value = float(text)
        require(math.isfinite(value), "json_nonfinite", text)
        return value

    def constant(text):
        raise DataError("json_nonfinite", text)

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_float=number,
                          parse_constant=constant)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise DataError("json_invalid", str(error)) from error


def keys(value, expected, detail):
    require(type(value) is dict and set(value) == set(expected), "schema_keys", detail)


def hash_string(value, detail):
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "schema_type", detail)


def regular(path, limit=FILE_LIMIT):
    path = Path(path).absolute()
    require(path == path.resolve(), "unsafe_path", str(path))
    try:
        info = path.lstat()
    except OSError as error:
        raise DataError("unsafe_path", str(path)) from error
    require(stat.S_ISREG(info.st_mode), "unsafe_path", str(path))
    require(0 < info.st_size <= limit, "file_size", str(path))
    return path


def read_bounded(path, limit=FILE_LIMIT):
    path = regular(path, limit)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit, "file_size", str(path))
    return data


def reservation_entries():
    """Exact reviewed development identities, independent of a repository file."""
    result = []
    for split, seeds in (("train", range(100000, 100012)),
                         ("validation", range(100100, 100104))):
        for i, seed in enumerate(seeds):
            moving = split == "train" and 4 <= i < 8
            result.append({
                "trajectory_id": f"smoke-{split}-{seed}", "seed": seed,
                "split": split, "geometry": f"g{i % 4}",
                "forcing_regime": "moving" if moving else "steady",
                "amplitude_m_s2": 1.0 if moving else 0.5,
                "dt_s": 1.0 / 60.0, "frames": FRAMES,
            })
    return result


def read_origin(blob):
    require(digest(blob) == ORIGIN_SHA256, "origin_hash", "immutable development manifest")
    origin = strict_json(blob)
    require(origin["status"] == "development_complete" and origin["fixture"] is False
            and origin["preset"] == "smoke" and origin["counts_xyz"] == list(COUNTS)
            and origin["selection"] == {"split": "development", "trajectory_id": None},
            "development_admission", "complete smoke development only")
    require(origin["provenance"]["lab_commit"] == SOURCE_COMMIT
            and origin["provenance"]["native_receipt"]["base_commit"] == NATIVE_BASE,
            "origin_source", "accepted source/native pins")
    expected = reservation_entries()
    require(len(origin["trajectories"]) == len(expected), "lineage", "whole development set")
    for trajectory, entry in zip(origin["trajectories"], expected):
        require(canonical(trajectory["reservation_entry"]) == canonical(entry)
                and trajectory["trajectory_id"] == entry["trajectory_id"]
                and len(trajectory["frames"]) == FRAMES,
                "lineage", "reserved complete trajectory")
    return origin


def source_fingerprint(trajectory):
    # Same physical identity serialized by the independently qualified verifier.
    geometry = dict(trajectory["geometry"])
    geometry["counts_xyz"] = [int(v) for v in geometry["counts_xyz"]]
    for field in ("extents_m", "origin_m"):
        geometry[field] = [0.0 if v == 0 else float(v) for v in geometry[field]]
    geometry["density_kg_m3"] = float(geometry["density_kg_m3"])
    return digest(canonical({
        "geometry": geometry,
        "initial": {k: v["sha256"] for k, v in trajectory["initial"].items()},
        "frames": [{k: v["sha256"] for k, v in frame["arrays"].items()}
                   for frame in trajectory["frames"]],
    }))


def original_row(trajectory, frame):
    previous = trajectory["initial"]["pressure"] if frame == 0 else (
        trajectory["frames"][frame - 1]["arrays"]["pressure"]
    )
    record = trajectory["frames"][frame]
    entry = trajectory["reservation_entry"]
    return {
        "row_id": f"{trajectory['trajectory_id']}:frame-{frame:03d}",
        "trajectory_id": trajectory["trajectory_id"], "frame": frame,
        "seed": entry["seed"], "split": entry["split"],
        "geometry": entry["geometry"], "forcing_regime": entry["forcing_regime"],
        "amplitude_m_s2": entry["amplitude_m_s2"], "time_s": record["time_s"],
        "lineage": {"rhs": record["arrays"]["rhs"],
                    "reference": record["arrays"]["pressure"], "base": previous},
    }


def array_spec(rows):
    result = {}
    for name in FIELDS:
        shape = (rows, *COUNTS) if name in CELL_FIELDS else (
            (rows, 3) if name in ("spacing", "extents", "origin") else (rows,)
        )
        dtype = "<i4" if name in INT_FIELDS else "<f8"
        result[name] = (shape, dtype, math.prod(shape) * np.dtype(dtype).itemsize)
    return result


def packed_fingerprint(arrays, start, trajectory):
    geometry = trajectory["geometry"]
    physics = {
        "counts": list(COUNTS), "extents": [float(v) for v in geometry["extents_m"]],
        "origin": [float(v) for v in geometry["origin_m"]],
        "rho": float(geometry["density_kg_m3"]),
        "dt": float(trajectory["reservation_entry"]["dt_s"]),
        "frames": [{name: digest(arrays[name][i].tobytes(order="C"))
                    for name in CELL_FIELDS}
                   for i in range(start, start + FRAMES)],
    }
    return digest(canonical(physics))


def load_npz(blob, declaration, rows):
    keys(declaration, {"split", "trajectory_ids", "bytes", "sha256", "arrays", "rows"},
         "shard")
    require(type(declaration["bytes"]) is int and declaration["bytes"] == len(blob),
            "container_size", "NPZ exact bytes")
    hash_string(declaration["sha256"], "container hash")
    require(digest(blob) == declaration["sha256"], "container_hash", "NPZ bytes")
    specifications = array_spec(rows)
    keys(declaration["arrays"], FIELDS, "array inventory")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            members = archive.infolist()
            require(len(members) == len(FIELDS)
                    and {m.filename for m in members} == {f"{n}.npy" for n in FIELDS},
                    "npz_members", "closed unique numeric members")
            expanded = 0
            for member in members:
                require(member.compress_type == zipfile.ZIP_STORED and not member.flag_bits & 1,
                        "npz_compression", "only uncompressed, unencrypted NPZ")
                name = member.filename[:-4]
                shape, dtype, size = specifications[name]
                declared = declaration["arrays"][name]
                keys(declared, {"shape", "dtype", "bytes", "sha256"}, name)
                require(type(declared["shape"]) is list
                        and all(type(n) is int for n in declared["shape"])
                        and declared["shape"] == list(shape)
                        and declared["dtype"] == dtype
                        and type(declared["bytes"]) is int and declared["bytes"] == size,
                        "array_schema", name)
                hash_string(declared["sha256"], name)
                require(size <= FILE_LIMIT and size < member.file_size <= size + 1034,
                        "expanded_size", name)
                with archive.open(member) as stream:
                    require(stream.read(8) == b"\x93NUMPY\x01\x00", "npy_header", name)
                    header_size = int.from_bytes(stream.read(2), "little")
                    require(0 < header_size <= 1024, "npy_header", name)
                    header = ast.literal_eval(stream.read(header_size).decode("latin1"))
                    keys(header, {"descr", "fortran_order", "shape"}, name)
                    require(header["descr"] == dtype and header["fortran_order"] is False
                            and type(header["shape"]) is tuple
                            and all(type(n) is int for n in header["shape"])
                            and header["shape"] == shape
                            and member.file_size == 10 + header_size + size,
                            "npy_header", name)
                expanded += member.file_size
            require(expanded <= FILE_LIMIT, "expanded_size", "total uncompressed shard")
        # Every NPY header/product/dtype has been admitted before NumPy allocates.
        with np.load(io.BytesIO(blob), allow_pickle=False) as loaded:
            result = {name: loaded[name] for name in FIELDS}
    except DataError:
        raise
    except (ValueError, OSError, zipfile.BadZipFile, SyntaxError, KeyError) as error:
        raise DataError("npz_invalid", str(error)) from error
    for name, values in result.items():
        require(np.isfinite(values).all(), "array_nonfinite", name)
        require(digest(values.tobytes(order="C")) == declaration["arrays"][name]["sha256"],
                "array_hash", name)
    return result


@dataclass
class Dataset:
    train: dict
    validation: dict
    train_rows: list
    validation_rows: list
    splits: dict
    input_hashes: dict
    diagnostics: dict


def validate_residual_origin(value, expected_sha256, origin):
    hash_string(expected_sha256, "independently supplied native residual origin digest")
    require(digest(canonical(value)) == expected_sha256,
            "native_residual_origin_hash", "reviewed project/recipe authority")
    keys(value, {"schema_version", "origin_sha256", "native_identity", "binary_sha256",
                 "build_receipt_sha256", "rows"}, "native residual origin")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["origin_sha256"] == ORIGIN_SHA256,
            "native_residual_origin_schema", "typed original source")
    for field in ("binary_sha256", "build_receipt_sha256"):
        hash_string(value[field], field)
    identity = value["native_identity"]
    keys(identity, {"schema", "binding", "native_receipt", "native_threads", "source_sha256"},
         "native identity")
    require(identity["schema"] == "rheon-global-pressure-training-native-v1"
            and type(identity["native_threads"]) is int and identity["native_threads"] == 1,
            "native_residual_origin_schema", "qualified serial native identity")
    hash_string(identity["source_sha256"], "native source hash")
    receipt = identity["native_receipt"]
    keys(receipt, {"schema_version", "base_commit", "base_tree", "patch_sha256",
                   "prepared_tree_sha256", "source_manifest_sha256", "source_files_sha256",
                   "license_sha256"}, "native dependency receipt")
    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
            and receipt["base_commit"] == NATIVE_BASE,
            "native_residual_origin_schema", "typed native dependency pin")
    for field in set(receipt) - {"schema_version", "base_commit", "base_tree"}:
        hash_string(receipt[field], field)
    require(type(receipt["base_tree"]) is str and len(receipt["base_tree"]) == 40,
            "native_residual_origin_schema", "Git tree identity")
    binding = identity["binding"]
    keys(binding, {"schema", "prepared_source", "native_receipt", "resolved_native_manifest",
                   "metadata_sha256", "cargo_command", "source_sha256", "toolchain",
                   "configuration"}, "native build binding")
    require(binding["schema"] == "rheon-global-pressure-training-build-v1"
            and canonical(binding["native_receipt"]) == canonical(receipt),
            "native_residual_origin_schema", "build/dependency binding")
    for field in ("prepared_source", "resolved_native_manifest", "configuration"):
        require(type(binding[field]) is str, "native_residual_origin_schema", field)
    hash_string(binding["metadata_sha256"], "native resolved metadata")
    require(type(binding["cargo_command"]) is list
            and all(type(v) is str for v in binding["cargo_command"]),
            "native_residual_origin_schema", "build invocation")
    require(type(binding["source_sha256"]) is dict and binding["source_sha256"],
            "native_residual_origin_schema", "native source closure")
    for name, sha in binding["source_sha256"].items():
        require(type(name) is str, "native_residual_origin_schema", "source path")
        hash_string(sha, "source hash")
    keys(binding["toolchain"], {"cargo", "rustc_vV", "rustc_path", "rustdoc_path"}, "toolchain")
    require(all(type(v) is str for v in binding["toolchain"].values()),
            "native_residual_origin_schema", "toolchain strings")
    rows = value["rows"]
    expected_ids = [original_row(t, f)["row_id"] for t in origin["trajectories"]
                    for f in range(FRAMES)]
    require(type(rows) is list and len(rows) == len(expected_ids),
            "native_residual_origin_schema", "complete derived residual history")
    for row, row_id in zip(rows, expected_ids):
        keys(row, {"row_id", "sha256"}, "native derived row")
        require(row["row_id"] == row_id, "native_residual_origin_schema", "derived row lineage")
        hash_string(row["sha256"], "native residual row hash")
    return rows


def load_data(train0_path, train1_path, validation_path, splits_path, origin_path, *,
              expected_native_residual_origin_sha256):
    """Admit exact whole development histories; no normalizers or native imports."""
    paths = [regular(p, SPLITS_LIMIT if i == 3 else FILE_LIMIT) for i, p in enumerate(
        (train0_path, train1_path, validation_path, splits_path, origin_path)
    )]
    require(len(set(paths)) == 5, "unsafe_path", "five distinct flat inputs")
    require(sum(p.stat().st_size for p in paths) <= INPUT_LIMIT,
            "input_size", "flat input aggregate")
    # Hash and decode the same byte snapshots, even if caller-owned files change
    # during the read. Never report a later reread as identity of loaded arrays.
    blobs = [read_bounded(path, SPLITS_LIMIT if i == 3 else FILE_LIMIT)
             for i, path in enumerate(paths)]
    require(sum(map(len, blobs)) <= INPUT_LIMIT, "input_size", "actual flat input bytes")
    origin_blob = blobs[4]
    origin = read_origin(origin_blob)
    splits = strict_json(blobs[3])
    keys(splits, {"schema_version", "status", "preset", "origin_sha256", "shards",
                  "trajectories", "native_residual_origin"}, "splits")
    require(type(splits["schema_version"]) is int and splits["schema_version"] == 1
            and splits["status"] == "lossless_development" and splits["preset"] == "smoke",
            "development_admission", "typed original development packet")
    require(splits["origin_sha256"] == digest(origin_blob), "origin_hash", "split origin")
    native_rows = validate_residual_origin(splits["native_residual_origin"],
                                          expected_native_residual_origin_sha256, origin)
    keys(splits["shards"], SHARDS, "shard inventory")
    expected_groups = (origin["trajectories"][:6], origin["trajectories"][6:12],
                       origin["trajectories"][12:])
    require(type(splits["trajectories"]) is list and len(splits["trajectories"]) == 16,
            "lineage", "all whole trajectory attestations")
    capture_seen, packed_seen = {}, {}
    # Admit all row/parent metadata before loading any numeric payload. A crop
    # in a later shard must not reach an earlier shard's array allocation.
    for shard_index, (name, trajectories) in enumerate(zip(SHARDS, expected_groups)):
        declaration = splits["shards"][name]
        keys(declaration, {"split", "trajectory_ids", "bytes", "sha256", "arrays", "rows"}, name)
        ids = [t["trajectory_id"] for t in trajectories]
        require(declaration["split"] == ("validation" if shard_index == 2 else "train")
                and declaration["trajectory_ids"] == ids,
                "lineage", "no relabelled/cropped/cross-shard parent")
        expected_rows = [[t["trajectory_id"], f] for t in trajectories for f in range(FRAMES)]
        require(type(declaration["rows"]) is list and canonical(declaration["rows"]) ==
                canonical(expected_rows), "lineage", "explicit complete row mapping")
    for trajectory, attestation in zip(origin["trajectories"], splits["trajectories"]):
        keys(attestation, {"trajectory_id", "parent_trajectory_id", "split", "frames",
                           "capture_fingerprint_sha256", "packed_fingerprint_sha256"},
             "trajectory attestation")
        require(attestation["trajectory_id"] == trajectory["trajectory_id"]
                and attestation["parent_trajectory_id"] == trajectory["trajectory_id"]
                and attestation["split"] == trajectory["reservation_entry"]["split"]
                and type(attestation["frames"]) is int and attestation["frames"] == FRAMES,
                "lineage", "exact parent identity/coverage")
        capture_fp = source_fingerprint(trajectory)
        require(attestation["capture_fingerprint_sha256"] == capture_fp,
                "capture_fingerprint", trajectory["trajectory_id"])
        hash_string(attestation["packed_fingerprint_sha256"], "packed fingerprint")
        require(capture_fp not in capture_seen, "lineage_duplicate", "original whole history")
        capture_seen[capture_fp] = trajectory["trajectory_id"]
    all_rows, arrays, shared = [], [], {}
    for shard_index, (name, path, trajectories) in enumerate(zip(SHARDS, paths, expected_groups)):
        declaration = splits["shards"][name]
        values = load_npz(blobs[shard_index], declaration, len(trajectories) * FRAMES)
        rows = []
        for local_index, trajectory in enumerate(trajectories):
            global_index = shard_index * 6 + local_index
            start = local_index * FRAMES
            attestation = splits["trajectories"][global_index]
            geometry = trajectory["geometry"]
            for frame in range(FRAMES):
                row = original_row(trajectory, frame)
                i = start + frame
                require(int(values["trajectory"][i]) == global_index
                        and int(values["frame"][i]) == frame,
                        "row_mapping", row["row_id"])
                physical = {
                    "spacing": np.asarray(geometry["extents_m"], dtype="<f8") / 16.0,
                    "extents": np.asarray(geometry["extents_m"], dtype="<f8"),
                    "origin": np.asarray(geometry["origin_m"], dtype="<f8"),
                    "rho": geometry["density_kg_m3"],
                    "dt": trajectory["reservation_entry"]["dt_s"],
                }
                for field, expected in physical.items():
                    require(np.array_equal(values[field][i], expected), "physical_metadata", field)
                for field in ("rhs", "reference", "base"):
                    raw_hash = digest(values[field][i].tobytes(order="C"))
                    require(raw_hash == row["lineage"][field]["sha256"],
                            "raw_lineage", row["row_id"] + ":" + field)
                    shared.setdefault((field, raw_hash), []).append(row["row_id"])
                derived = native_rows[global_index * FRAMES + frame]
                require(digest(values["native_input_residual"][i].tobytes()) == derived["sha256"],
                        "native_residual_lineage", row["row_id"])
                row["lineage"]["native_input_residual"] = {
                    "sha256": derived["sha256"],
                    "source": "qualified native full original-RHS residual including cell zero",
                    "rhs_sha256": row["lineage"]["rhs"]["sha256"],
                    "base_sha256": row["lineage"]["base"]["sha256"],
                    "native_residual_origin_sha256": expected_native_residual_origin_sha256,
                }
                require(values["reference"][i, 0, 0, 0] == 0
                        and values["base"][i, 0, 0, 0] == 0,
                        "pressure_gauge", row["row_id"])
                rows.append(row)
            packed_fp = packed_fingerprint(values, start, trajectory)
            require(attestation["packed_fingerprint_sha256"] == packed_fp,
                    "packed_fingerprint", trajectory["trajectory_id"])
            require(packed_fp not in packed_seen, "lineage_duplicate", "packed numeric whole history")
            packed_seen[packed_fp] = trajectory["trajectory_id"]
        arrays.append(values)
        all_rows.append(rows)
    train = {field: np.concatenate([arrays[0][field], arrays[1][field]]) for field in FIELDS}
    diagnostics = {
        "shared_packed_payloads": [
            {"field": field, "sha256": sha, "row_ids": rows,
             "interpretation": "shared field diagnostic; not independent trajectory evidence"}
            for (field, sha), rows in shared.items() if len(rows) > 1
        ],
        "original_fingerprint_claim": "qualified packager attestation; no fresh reader physics",
        "packed_fingerprint_claim": "recomputed numeric whole history across all three shards",
    }
    input_hashes = {name: digest(blob) for name, blob in zip(
        (*SHARDS, "splits.json", "origin.json"), blobs
    )}
    return Dataset(train, arrays[2], all_rows[0] + all_rows[1], all_rows[2],
                   splits, input_hashes, diagnostics)


def fresh_verify(capture):
    verifier = Path(__file__).resolve().parents[1] / "verify_capture.py"
    result = subprocess.run([sys.executable, str(verifier), str(capture)],
                            capture_output=True, timeout=120, check=False)
    require(result.returncode == 0, "capture_verification",
            result.stderr.decode(errors="replace")[-2000:])
    report = strict_json(result.stdout)
    require(report["status"] == "verified" and report["frames"] == 384
            and report["trajectories"] == 16, "capture_verification", "complete fresh verification")
    return report


def read_raw(capture, descriptor):
    name = descriptor["path"]
    require(type(name) is str, "unsafe_path", "raw array path type")
    relative = Path(name)
    require(not relative.is_absolute()
            and relative.as_posix() == name and ".." not in relative.parts,
            "unsafe_path", "raw array")
    require(descriptor["dtype"] == "<f8" and descriptor["shape_xyz"] == list(COUNTS)
            and type(descriptor["bytes"]) is int and descriptor["bytes"] == 32768,
            "raw_schema", name)
    blob = read_bounded(capture / relative, 32768)
    require(len(blob) == descriptor["bytes"] and digest(blob) == descriptor["sha256"],
            "raw_lineage", name)
    result = np.frombuffer(blob, dtype="<f8").reshape(COUNTS)
    require(np.isfinite(result).all(), "array_nonfinite", name)
    return result


def _write_packet(origin_blob, origin, output, report, raw_reader, native_receipt,
                  native_build_receipt_sha256, residual_reader, final_native_check):
    """Internal lossless writer; caller must have completed fresh capture verification."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        packager_source_hash = digest(read_bounded(Path(__file__)))
        (output / "origin.json").write_bytes(origin_blob)
        splits = {"schema_version": 1, "status": "lossless_development", "preset": "smoke",
                  "origin_sha256": digest(origin_blob), "shards": {}, "trajectories": []}
        residual_origin = {
            "schema_version": 1, "origin_sha256": digest(origin_blob),
            "native_identity": native_receipt["identity"],
            "binary_sha256": native_receipt["binary_sha256"],
            "build_receipt_sha256": native_build_receipt_sha256, "rows": [],
        }
        groups = (origin["trajectories"][:6], origin["trajectories"][6:12],
                  origin["trajectories"][12:])
        for shard_index, (name, trajectories) in enumerate(zip(SHARDS, groups)):
            count = len(trajectories) * FRAMES
            values = {field: np.empty(shape, dtype=dtype)
                      for field, (shape, dtype, _) in array_spec(count).items()}
            rows = []
            for local_index, trajectory in enumerate(trajectories):
                geometry = trajectory["geometry"]
                for frame in range(FRAMES):
                    i = local_index * FRAMES + frame
                    row = original_row(trajectory, frame)
                    for field in ("rhs", "reference", "base"):
                        values[field][i] = raw_reader(row["lineage"][field])
                    values["native_input_residual"][i] = residual_reader(
                        values["rhs"][i], values["base"][i], geometry,
                        trajectory["reservation_entry"]["dt_s"])
                    residual_origin["rows"].append({
                        "row_id": row["row_id"],
                        "sha256": digest(values["native_input_residual"][i].tobytes()),
                    })
                    values["spacing"][i] = np.asarray(geometry["extents_m"]) / 16.0
                    values["extents"][i] = geometry["extents_m"]
                    values["origin"][i] = geometry["origin_m"]
                    values["rho"][i] = geometry["density_kg_m3"]
                    values["dt"][i] = trajectory["reservation_entry"]["dt_s"]
                    values["frame"][i] = frame
                    values["trajectory"][i] = shard_index * 6 + local_index
                    rows.append([trajectory["trajectory_id"], frame])
                splits["trajectories"].append({
                    "trajectory_id": trajectory["trajectory_id"],
                    "parent_trajectory_id": trajectory["trajectory_id"],
                    "split": trajectory["reservation_entry"]["split"], "frames": FRAMES,
                    "capture_fingerprint_sha256": source_fingerprint(trajectory),
                    "packed_fingerprint_sha256": packed_fingerprint(
                        values, local_index * FRAMES, trajectory),
                })
            with (output / name).open("xb") as stream:
                np.savez(stream, **values)
            blob = read_bounded(output / name)
            splits["shards"][name] = {
                "split": "validation" if shard_index == 2 else "train",
                "trajectory_ids": [t["trajectory_id"] for t in trajectories],
                "bytes": len(blob), "sha256": digest(blob), "rows": rows,
                "arrays": {field: {"shape": list(array.shape), "dtype": array.dtype.str,
                                   "bytes": array.nbytes, "sha256": digest(array.tobytes())}
                           for field, array in values.items()},
            }
        final_native_check()
        splits["native_residual_origin"] = residual_origin
        residual_origin_sha256 = digest(canonical(residual_origin))
        split_blob = canonical(splits)
        require(len(split_blob) <= SPLITS_LIMIT, "file_size", "compact splits")
        (output / "splits.json").write_bytes(split_blob)
        admitted = load_data(*(output / name for name in SHARDS),
                             output / "splits.json", output / "origin.json",
                             expected_native_residual_origin_sha256=residual_origin_sha256)
        receipt = {
            "schema_version": 1, "status": "verified_lossless_development_packet",
            "origin_sha256": digest(origin_blob), "capture_source_commit": SOURCE_COMMIT,
            "packager_source_sha256": packager_source_hash,
            "native_residual_origin_sha256": residual_origin_sha256,
            "verification": {"report": report, "sha256": digest(canonical(report))},
            "files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha}
                      for name, sha in admitted.input_hashes.items()},
            "array_payload_bytes": sum(d["bytes"] for s in splits["shards"].values()
                                       for d in s["arrays"].values()),
            "train_rows": len(admitted.train_rows), "validation_rows": len(admitted.validation_rows),
            "diagnostics": admitted.diagnostics, "packing_elapsed_s": time.monotonic() - started,
        }
        receipt_blob = canonical(receipt)
        require(len(receipt_blob) <= FILE_LIMIT, "file_size", "dataset receipt")
        require(sum(v["bytes"] for v in receipt["files"].values()) + len(receipt_blob)
                <= INPUT_LIMIT, "input_size", "complete packet")
        require(digest(read_bounded(Path(__file__))) == packager_source_hash,
                "packager_source_drift", "packaging source changed during execution")
        (output / "dataset-receipt.json").write_bytes(receipt_blob)
        return receipt
    except BaseException:
        shutil.rmtree(output)
        raise


def pack(capture, output, native_binary):
    capture = Path(capture).absolute()
    require(capture == capture.resolve() and capture.is_dir(), "unsafe_path", "capture root")
    origin_blob = read_bounded(capture / "manifest.json", 4 * MIB)
    origin = read_origin(origin_blob)
    report = fresh_verify(capture)
    # Fresh verification may not be silently replaced by a stale stored report.
    require(read_bounded(capture / "manifest.json", 4 * MIB) == origin_blob,
            "origin_hash", "manifest changed during verification")
    output = Path(output).absolute()
    lab = Path(__file__).resolve().parents[3]
    namespace = lab / "runs/global-pressure-v1-training"
    require(output == output.resolve() and namespace in output.parents,
            "unsafe_path", "task-owned training output only")
    require(not output.exists(), "output_exists", "exclusive new packet")
    # Native execution remains a local packaging concern, outside captured recipes.
    path = Path(__file__).with_name("evaluate_native.py")
    spec = importlib.util.spec_from_file_location("qualified_training_native", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native_binary = regular(native_binary)
    try:
        before = native.admit_binary(native_binary)
    except ValueError as error:
        raise DataError("native_build_admission", str(error)) from error
    build_hash = digest(read_bounded(native_binary.parent / "build-receipt.json"))

    def residual_reader(rhs, base, geometry, _dt):
        request = {
            "schema": "rheon-global-pressure-native-residual-v1", "counts_xyz": list(COUNTS),
            "spacing_m": [float(v) / 16.0 for v in geometry["extents_m"]],
            "density_kg_m3": geometry["density_kg_m3"],
            "original_rhs": rhs.reshape(-1).tolist(), "base_pressure": base.reshape(-1).tolist(),
        }
        try:
            response, _ = native.native_call(native_binary, "--residual", request, timeout=10)
        except ValueError as error:
            raise DataError("native_residual_refusal", str(error)) from error
        keys(response, {"schema", "residual", "identity", "scope"}, "native residual reply")
        require(response["schema"] == request["schema"]
                and canonical(response["identity"]) == canonical(before["identity"]),
                "native_residual_identity", "actual qualified producer")
        numbers = response["residual"]
        require(type(numbers) is list and len(numbers) == 4096
                and all(type(v) in (int, float) and math.isfinite(v) for v in numbers),
                "native_residual_schema", "finite f64 cell residual")
        return np.asarray(numbers, dtype="<f8").reshape(COUNTS)

    def final_check():
        try:
            after = native.admit_binary(native_binary)
        except ValueError as error:
            raise DataError("native_residual_identity", str(error)) from error
        require(canonical(after) == canonical(before)
                and digest(read_bounded(native_binary.parent / "build-receipt.json")) == build_hash,
                "native_residual_identity", "source/build drift during packing")

    return _write_packet(origin_blob, origin, output, report,
                         lambda descriptor: read_raw(capture, descriptor),
                         before, build_hash, residual_reader, final_check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument("--capture", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--native-binary", type=Path, required=True)
    verification = sub.add_parser("verify")
    for name in ("train-0", "train-1", "validation", "splits", "origin"):
        verification.add_argument("--" + name, type=Path, required=True)
    verification.add_argument("--expected-native-residual-origin-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.action == "pack":
            result = pack(args.capture, args.output, args.native_binary)
        else:
            dataset = load_data(args.train_0, args.train_1, args.validation,
                                args.splits, args.origin,
                                expected_native_residual_origin_sha256=
                                args.expected_native_residual_origin_sha256)
            result = {"status": "verified_numeric_lossless_packet",
                      "train_rows": len(dataset.train_rows),
                      "validation_rows": len(dataset.validation_rows),
                      "input_hashes": dataset.input_hashes, "diagnostics": dataset.diagnostics}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (DataError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "code": getattr(error, "code", "invalid_data"),
                          "detail": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
