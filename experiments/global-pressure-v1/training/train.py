#!/usr/bin/env python3
"""Fixed 200-update eager CPU training and exact-byte, zero-update restore.

The generic Runs owner governs admission and process custody. This child validates
captured scientific inputs and typed artifact evidence; it adds no executor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import resource
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
import time
import zipfile
from typing import Mapping

import numpy as np
import torch

if __package__:
    from . import model as numerics
else:
    import model as numerics

FIXED_CONFIG = {
    "schema": "rheon-global-pressure-training-config-v1",
    "preset": "smoke",
    "model_format": numerics.ARCHITECTURE["schema"],
    "updates": 200,
    "batch_size": 2,
    "initialization_seed": 20261014,
    "sample_order_seed": 20261015,
    "optimizer": {"name": "Adam", "learning_rate": 1e-3, "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0},
    "gradient_norm_cap": 1.0,
    "log_every_updates": 20,
    "validation_updates": [0, 200],
    "torch_threads": 1,
    "torch_interop_threads": 1,
    "model_dtype": "float32",
    "loss_dtype": "float64",
    "counts_xyz": [16, 16, 16],
    "witness_frames": [0, 23],
    "max_parameters": 1_000_000,
    "max_weights_bytes": 8 * 1024 * 1024,
    "max_predictions_bytes": 8 * 1024 * 1024,
    "max_output_bytes": 32 * 1024 * 1024,
}
SOURCE_SCHEMA = "rheon-global-pressure-training-source-v1"
RECEIPT_SCHEMA = "rheon-global-pressure-training-receipt-v1"
NATIVE_BASE = "2209012bc5f902693d0f695ec1c2d854126af317"
TULDOK_REVISION = "f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6"
CUSTODY_VALIDATOR_SHA256 = "9bf5de04bef27a86ad239e05d9d105985d966a2b114350c3c8e3a92053b1a999"
OUTPUTS = {
    "weights.npz": ("checkpoint", 8 * 1024 * 1024),
    "splits.json": ("training_splits", 256 * 1024),
    "normalization.json": ("preprocessing", 256 * 1024),
    "receipt.json": ("training_result", 512 * 1024),
    "predictions.npz": ("predictions", 8 * 1024 * 1024),
    "summary.json": ("result_summary", 64 * 1024),
}
TRAINED_INPUTS = {
    "trained_weights": ("weights.npz", "checkpoint"),
    "trained_receipt": ("receipt.json", "training_result"),
    "trained_normalization": ("normalization.json", "preprocessing"),
    "trained_predictions": ("predictions.npz", "predictions"),
    "trained_splits": ("splits.json", "training_splits"),
}
HASH = re.compile(r"[a-f0-9]{64}")


class TrainingContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise TrainingContractError(code, message)


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), "schema_keys", label)


def strict_json(raw: bytes | str):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate_json_key", key)
            value[key] = item
        return value
    def constant(value):
        raise TrainingContractError("json_nonfinite", value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    def finite_tree(item):
        if type(item) is float:
            require(math.isfinite(item), "json_nonfinite", "overflowing JSON float")
        elif type(item) is dict:
            for child in item.values(): finite_tree(child)
        elif type(item) is list:
            for child in item: finite_tree(child)
    finite_tree(value)
    return value


def canonical(value) -> bytes:
    return numerics.canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path: Path, maximum: int) -> bytes:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "unsafe_path", str(path))
    require(0 <= info.st_size <= maximum, "resource_bound", str(path))
    return path.read_bytes()


def identity(raw: bytes) -> dict:
    return {"sha256": sha256(raw), "bytes": len(raw)}


def observed_peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB; this is an observation, not aggregate containment.
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    require(0 < peak <= 4 * 1024**3, "observed_ram_bound", "observed process peak exceeds 4 GiB")
    return peak


def progress(stage: str, completed: int, total: int, **metrics) -> None:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", stage) is not None, "progress", stage)
    require(type(completed) is type(total) is int and 0 <= completed <= total <= 1_000_000, "progress", "counts")
    require(len(metrics) <= 8 and all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values()), "progress", "finite metrics")
    print(canonical({"format": "tuldok_progress_v1", "stage": stage, "completed": completed, "total": total, "metrics": metrics}).decode().rstrip(), flush=True)


def sample_order(train_rows: list[dict]) -> np.ndarray:
    """Equal whole-trajectory selection then uniform frame selection, train only."""
    groups = {}
    for index, row in enumerate(train_rows):
        groups.setdefault(row["trajectory_id"], []).append(index)
    require(groups, "sample_order", "empty train rows")
    group_rows = list(groups.values())
    rng = np.random.Generator(np.random.PCG64(20261015))
    result = []
    for _ in range(200):
        batch = []
        for _ in range(2):
            rows = group_rows[int(rng.integers(len(group_rows)))]
            batch.append(rows[int(rng.integers(len(rows)))])
        result.append(batch)
    return np.asarray(result, dtype="<i8")


def tensors(arrays: Mapping[str, np.ndarray], indices, *, labels=True) -> dict[str, torch.Tensor]:
    # Each selected whole frame is copied; tensors cannot mutate input shards.
    result = {}
    names = ("rhs", "reference", "base", "native_input_residual") if labels else ("base", "native_input_residual")
    for name in names:
        result[name] = torch.from_numpy(np.array(arrays[name][indices], dtype="<f8", copy=True)).unsqueeze(1)
    for name in ("spacing", "rho", "dt"):
        result[name] = torch.from_numpy(np.array(arrays[name][indices], dtype="<f8", copy=True))
    return result


def prediction(network, batch, normalization):
    return numerics.predict_pressure(network, batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization)


def losses(network, batch, normalization):
    estimated = prediction(network, batch, normalization)
    return numerics.pressure_loss(estimated, batch["reference"], batch["rhs"], batch["spacing"], batch["rho"], normalization)


def one_update(network, optimizer, arrays, indices, normalization) -> tuple[float, float]:
    """One arithmetic step, reused by the fixed loop and disposable unit fixture."""
    network.train()
    optimizer.zero_grad(set_to_none=True)
    loss = losses(network, tensors(arrays, indices), normalization).mean()
    require(bool(torch.isfinite(loss)), "training_nonfinite", "loss")
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(parameter).all()) for parameter in network.parameters()), "training_nonfinite", "weights")
    return float(loss.detach()), float(gradient)


def evaluate(network, arrays, rows, normalization) -> float:
    network.eval()
    grouped = {}
    with torch.no_grad():
        for first in range(0, len(rows), 2):
            values = losses(network, tensors(arrays, np.arange(first, min(first + 2, len(rows)))), normalization)
            require(bool(torch.isfinite(values).all()), "evaluation_nonfinite", "loss")
            for row, value in zip(rows[first:first + 2], values.tolist()):
                grouped.setdefault(row["trajectory_id"], []).append(value)
    require(grouped, "evaluation", "empty rows")
    return float(math.fsum(math.fsum(values) / len(values) for values in grouped.values()) / len(grouped))


def state_inventory(network) -> dict:
    result = {}
    for name, tensor in network.state_dict().items():
        values = tensor.detach().cpu().numpy()
        require(values.dtype == np.dtype("<f4") and values.flags.c_contiguous and np.isfinite(values).all(), "state_type", name)
        result[name] = {"dtype": "<f4", "shape": list(values.shape), "bytes": values.nbytes, "sha256": sha256(values.tobytes())}
    return result


def save_weights(network, path: Path) -> dict:
    inventory = state_inventory(network)
    with path.open("xb") as stream:
        np.savez(stream, **{name: tensor.detach().cpu().numpy() for name, tensor in network.state_dict().items()})
    require(path.stat().st_size <= numerics.MAX_WEIGHTS_BYTES, "weights_bound", str(path))
    return inventory


def read_npz(path: Path | bytes, inventory: dict, maximum: int) -> dict[str, np.ndarray]:
    """Check closed uncompressed NPY headers/shape/bytes before allocation."""
    payload = path if type(path) is bytes else regular_bytes(path, maximum)
    require(0 < len(payload) <= maximum, "resource_bound", "immutable numeric payload")
    require(type(inventory) is dict and inventory, "npz_inventory", "empty/non-object")
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        require(len(members) == len(inventory) and {member.filename for member in members} == {name + ".npy" for name in inventory}, "npz_inventory", str(path))
        require(len({member.filename for member in members}) == len(members), "npz_inventory", "duplicate members")
        for member in members:
            name = member.filename[:-4]
            spec = inventory[name]
            keys(spec, {"dtype", "shape", "bytes", "sha256"}, "numeric inventory " + name)
            require(type(spec["shape"]) is list and all(type(value) is int and value > 0 for value in spec["shape"]), "npz_shape", name)
            require(type(spec["bytes"]) is int and spec["bytes"] > 0 and type(spec["sha256"]) is str and HASH.fullmatch(spec["sha256"]), "npz_inventory", name)
            require(spec["dtype"] in ("<f4", "<f8", "<i8", "<i4"), "npz_dtype", name)
            expected_bytes = math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
            require(expected_bytes == spec["bytes"] <= maximum, "npz_shape", name)
            require(member.compress_type == zipfile.ZIP_STORED and not member.flag_bits & 1 and member.file_size <= maximum, "npz_encoding", name)
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "npz_header", name)
                header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = header(stream, max_header_size=10_000)
                require(type(shape) is tuple and all(type(value) is int and value > 0 for value in shape), "npz_header_type", name)
                require(list(shape) == spec["shape"] and fortran is False and dtype.str == spec["dtype"], "npz_header", name)
                require(member.file_size - stream.tell() == expected_bytes, "npz_bytes", name)
                raw = stream.read(expected_bytes + 1)
            require(len(raw) == expected_bytes and sha256(raw) == spec["sha256"], "npz_hash", name)
            values = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            require(np.isfinite(values).all(), "npz_nonfinite", name)
            arrays[name] = values
    return arrays


def load_weights(network, path: Path, inventory: dict) -> None:
    expected = network.state_dict()
    require(set(inventory) == set(expected), "weights_inventory", "architecture tensor names")
    for name, tensor in expected.items():
        require(inventory[name]["shape"] == list(tensor.shape) and inventory[name]["dtype"] == "<f4", "weights_inventory", name)
    arrays = read_npz(path, inventory, numerics.MAX_WEIGHTS_BYTES)
    network.load_state_dict({name: torch.from_numpy(values) for name, values in arrays.items()}, strict=True)


def witness_cases(dataset) -> list[dict]:
    rows = dataset.train_rows + dataset.validation_rows
    cases = []
    for row_index, row in enumerate(rows):
        if row["frame"] in (0, 23):
            cases.append({"case_id": f'{row["trajectory_id"]}:frame-{row["frame"]:03d}', "trajectory_id": row["trajectory_id"], "frame": row["frame"], "split": "train" if row_index < len(dataset.train_rows) else "validation", "row_index": row_index})
    require(len(cases) == 32 and len({case["case_id"] for case in cases}) == 32, "witness_cases", "exact fixed 32 cases")
    return cases


def make_predictions(network, dataset, normalization, cases):
    network.eval()
    pressures = []
    clocks = []
    with torch.no_grad():
        for case in cases:
            index = case["row_index"]
            arrays = dataset.train if index < len(dataset.train_rows) else dataset.validation
            local = index if index < len(dataset.train_rows) else index - len(dataset.train_rows)
            started = time.perf_counter()
            value = prediction(network, tensors(arrays, [local], labels=False), normalization)
            require(bool(torch.isfinite(value).all()), "prediction_nonfinite", case["case_id"])
            pressures.append(np.array(value[0, 0].numpy(), dtype="<f8", copy=True))
            clocks.append(time.perf_counter() - started)
    return {"pressure": np.stack(pressures), "row_indices": np.asarray([case["row_index"] for case in cases], dtype="<i8")}, clocks


def prediction_inventory(arrays) -> dict:
    require(set(arrays) == {"pressure", "row_indices"} and arrays["pressure"].shape == (32, 16, 16, 16) and arrays["pressure"].dtype == np.dtype("<f8") and arrays["row_indices"].shape == (32,) and arrays["row_indices"].dtype == np.dtype("<i8"), "predictions_inventory", "fixed numeric witness")
    return {name: {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes, "sha256": sha256(np.ascontiguousarray(array).tobytes())} for name, array in arrays.items()}


def write_json_new(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))


def copy_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)


QUALIFICATION_GATES = ["full_stencil", "gauge_loss_gradient", "model_parameter_gradient", "global_dependence", "shapes_16_32", "train_only_normalization", "validation_invariance", "numeric_weight_roundtrip", "strict_restore_custody"]
QUALIFICATION_METRICS = {"gauge_gradient_max_abs_error", "gauge_finite_difference_max_abs_error", "model_parameter_gradient_abs_error", "distant_input_derivative_abs", "global_pool_gradient_l1", "native_operator_max_abs_error", "dry_forward_s_16", "dry_forward_s_32", "dry_backward_update_s", "dry_peak_rss_bytes"}


def validate_qualification(report, source):
    keys(report, {"format", "status", "executable_sources", "runtime", "parameter_count", "raw_weight_bytes", "tests", "evidence"}, "model qualification")
    require(report["format"] == "rheon-global-pressure-model-qualification-v1" and report["status"] == "passed", "qualification_status", "passed pretraining report required")
    require(canonical(report["executable_sources"]) == canonical(source["executable_sources"]), "qualification_source", "exact executable source closure")
    expected_runtime = {"torch": "2.8.0+cpu", "numpy": "2.3.5", "threads": 1, "interop_threads": 1}
    require(canonical(report["runtime"]) == canonical(expected_runtime), "qualification_runtime", "exact CPU/runtime types")
    require(type(report["parameter_count"]) is type(report["raw_weight_bytes"]) is int and report["parameter_count"] == 372377 and report["raw_weight_bytes"] == 1489508, "qualification_model", "actual model bounds")
    keys(report["tests"], {"suite_sha256", "passed_count"}, "qualification tests")
    require(type(report["tests"]["suite_sha256"]) is str and HASH.fullmatch(report["tests"]["suite_sha256"]) and type(report["tests"]["passed_count"]) is int and report["tests"]["passed_count"] > 0, "qualification_tests", "actual passing suite identity")
    evidence = report["evidence"]
    keys(evidence, {"gates", "native_operator", "metrics"}, "qualification evidence")
    require(canonical(evidence["gates"]) == canonical(QUALIFICATION_GATES), "qualification_gates", "all required gates")
    keys(evidence["native_operator"], set(source["native_qualification"]) | {"oracle_action_sha256"}, "native operator evidence")
    for name, value in source["native_qualification"].items():
        require(evidence["native_operator"][name] == value, "qualification_native", name)
    require(type(evidence["native_operator"]["oracle_action_sha256"]) is str and HASH.fullmatch(evidence["native_operator"]["oracle_action_sha256"]), "qualification_native", "oracle bytes")
    keys(evidence["metrics"], QUALIFICATION_METRICS, "qualification metric inventory")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in evidence["metrics"].values()), "qualification_metrics", "finite nonnegative observed metrics")
    require(type(evidence["metrics"]["dry_peak_rss_bytes"]) is int and 0 < evidence["metrics"]["dry_peak_rss_bytes"] <= 4 * 1024**3 and evidence["metrics"]["distant_input_derivative_abs"] > 0 and evidence["metrics"]["global_pool_gradient_l1"] > 0, "qualification_metrics", "nonzero global dependence/observed RAM")


def validate_source(args, raw_inputs, config_raw, runtime_raw, evidence_raw):
    source_raw = regular_bytes(args.source_witness, 256 * 1024)
    source = strict_json(source_raw)
    keys(source, {"format", "source_commit", "tuldok_revision", "fixedmodel_format", "runtime_snapshot_sha256", "executable_sources", "config_sha256", "package_inputs", "native_qualification", "qualification_report_sha256", "native_residual_origin_sha256"}, "source witness")
    require(source["format"] == SOURCE_SCHEMA and source["fixedmodel_format"] == numerics.ARCHITECTURE["schema"], "source_format", "fixed model source")
    require(type(source["source_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", source["source_commit"]), "source_commit", "exact committed source")
    require(source["tuldok_revision"] == TULDOK_REVISION, "tuldok_revision", "accepted generic owner")
    require(type(source["native_residual_origin_sha256"]) is str and HASH.fullmatch(source["native_residual_origin_sha256"]), "residual_anchor", "approved derived anchor")
    keys(source["executable_sources"], {"data.py", "model.py", "train.py"}, "executable source closure")
    for name, expected in source["executable_sources"].items():
        require(type(expected) is str and HASH.fullmatch(expected) and sha256(regular_bytes(Path(__file__).parent / name, 8 * 1024 * 1024)) == expected, "source_hash", name)
    require(source["config_sha256"] == sha256(config_raw), "config_hash", "captured fixed config")
    require(source["runtime_snapshot_sha256"] == sha256(runtime_raw), "runtime_snapshot_hash", "captured manifest bytes")
    keys(source["package_inputs"], raw_inputs, "fixed package input closure")
    for name, raw in raw_inputs.items():
        require(source["package_inputs"][name] == identity(raw) and canonical(source["package_inputs"][name]) == canonical(identity(raw)), "input_hash", name)
    native = source["native_qualification"]
    keys(native, {"base_commit", "patch_sha256", "prepared_tree_sha256", "operator_binary_sha256"}, "native qualification")
    require(native["base_commit"] == NATIVE_BASE and all(type(native[name]) is str and HASH.fullmatch(native[name]) for name in native if name != "base_commit"), "native_identity", "qualified source closure")
    qualification = regular_bytes(args.source_witness.with_name("model-qualification.json"), 1024 * 1024)
    require(source["qualification_report_sha256"] == sha256(qualification), "qualification_hash", "pretraining source report")
    validate_qualification(strict_json(qualification), source)
    config = strict_json(config_raw)
    require(canonical(config) == canonical(FIXED_CONFIG), "fixed_config", "scientific config changed")
    runtime = strict_json(runtime_raw)
    keys(runtime, {"schema_version", "profile", "python_version", "packages", "launch", "files", "total_bytes"}, "runtime manifest")
    require(type(runtime["schema_version"]) is int and runtime["schema_version"] == 1 and runtime["profile"] == "captured_cpu_python_v1", "runtime_profile", "actual captured CPU")
    require(type(runtime["packages"]) is dict and runtime["packages"].get("torch") == "2.8.0+cpu" and runtime["packages"].get("numpy") == "2.3.5", "runtime_packages", "fixed package identity")
    require(torch.__version__ == "2.8.0+cpu" and np.__version__ == "2.3.5" and torch.version.cuda is None, "actual_runtime", "CPU package observations")
    require(runtime["python_version"] == f"{sys.version_info.major}.{sys.version_info.minor}", "actual_runtime", "Python version")
    evidence = strict_json(evidence_raw)
    expected_slots = set(raw_inputs) | (set(TRAINED_INPUTS) if args.mode == "restore" else set())
    keys(evidence, expected_slots, "typed material input slots")
    shared_project = None
    for slot, raw in raw_inputs.items():
        entry = evidence[slot]
        keys(entry, {"project_id", "manifest_sha256", "reference_id", "path", "measured_sha256"}, "project data receipt " + slot)
        require(all(type(entry[name]) is str and entry[name] for name in entry), "input_evidence_type", slot)
        require(HASH.fullmatch(entry["manifest_sha256"]) is not None and entry["measured_sha256"] == sha256(raw), "input_evidence_hash", slot)
        relative = Path(entry["path"])
        require(not relative.is_absolute() and relative.parts and all(part not in (".", "..") for part in relative.parts), "input_evidence_path", slot)
        project = (entry["project_id"], entry["manifest_sha256"])
        if shared_project is None: shared_project = project
        require(project == shared_project, "input_evidence_project", "one declared project data snapshot")
    return source, source_raw, config, evidence


def validate_trained_custody(evidence, raw_trained, evidence_path):
    # Reuse the accepted owner's exact captured cold validator; never copy its
    # custody semantics or import an ambient checkout.
    path = evidence_path.with_name("recipe_process.py")
    raw = regular_bytes(path, 1024 * 1024)
    require(sha256(raw) == CUSTODY_VALIDATOR_SHA256, "custody_validator_hash", "accepted captured owner source")
    spec = importlib.util.spec_from_file_location("global_pressure_captured_custody", path)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = owner
    spec.loader.exec_module(owner)
    shared = None
    for slot, (artifact, role) in TRAINED_INPUTS.items():
        entry = evidence[slot]
        keys(entry, {"kind", "source"}, "trained material receipt " + slot)
        require(entry["kind"] == "run_artifact", "trained_origin", "typed completed Run required")
        try:
            source = owner.validate_artifact_source(entry["source"], sha256(raw_trained[slot]), len(raw_trained[slot]))
        except owner.ProcessError as error:
            raise TrainingContractError("trained_custody", str(error)) from error
        require(source["artifact_name"] == artifact and source["artifact_role"] == role, "trained_origin", "artifact declaration " + slot)
        origin = {name: source[name] for name in ("catalog_id", "run_id", "attempt_id", "request_sha256", "material_sha256", "custody_receipt_sha256", "custody_identity", "custody_receipt_base64")}
        if shared is None: shared = origin
        require(canonical(origin) == canonical(shared), "trained_origin", "one originating completed Run")
    return shared


def validate_training_receipt(receipt, closure, network, cases, raw_trained):
    expected_keys = {"schema", "status", "mode", "optimizer_updates", "trained_optimizer_updates", "architecture", "parameter_count", "weights_sha256", "weights_inventory", "predictions_sha256", "predictions_inventory", "witness_cases", "per_case_inference_s", "closure", "sample_order_sha256", "training_history", "validation", "elapsed_s", "runtime_observations", "outputs", "originating_training_run"}
    keys(receipt, expected_keys, "training receipt")
    require(receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "complete" and receipt["mode"] == "train", "training_receipt", "completed fixed training")
    require(type(receipt["optimizer_updates"]) is type(receipt["trained_optimizer_updates"]) is int and receipt["optimizer_updates"] == receipt["trained_optimizer_updates"] == 200, "training_updates", "exact 200 actual steps")
    require(canonical(receipt["architecture"]) == canonical(numerics.ARCHITECTURE) and type(receipt["parameter_count"]) is int and receipt["parameter_count"] == numerics.parameter_count(network), "weights_architecture", "fixed actual architecture")
    require(canonical(receipt["witness_cases"]) == canonical(cases), "witness_cases", "ordered fixed witness")
    require(receipt["weights_sha256"] == sha256(raw_trained["trained_weights"]) and receipt["predictions_sha256"] == sha256(raw_trained["trained_predictions"]), "trained_payload_hash", "exact original artifact bytes")
    compare_keys = set(closure) - {"input_evidence_sha256"}
    require(set(receipt["closure"]) == set(closure), "closure_keys", "scientific identity closure")
    for name in compare_keys:
        require(canonical(receipt["closure"][name]) == canonical(closure[name]), "restore_identity", name)
    require(receipt["closure"]["normalization_sha256"] == sha256(raw_trained["trained_normalization"]) and receipt["closure"]["splits_sha256"] == sha256(raw_trained["trained_splits"]), "trained_payload_hash", "scale/split artifact linkage")
    for slot, (artifact, _) in TRAINED_INPUTS.items():
        if artifact == "receipt.json": continue
        require(receipt["outputs"][artifact]["sha256"] == sha256(raw_trained[slot]) and type(receipt["outputs"][artifact]["bytes"]) is int and receipt["outputs"][artifact]["bytes"] == len(raw_trained[slot]), "trained_output_hash", artifact)


def output_directory(path: Path) -> None:
    require(path.exists() and path.is_dir() and not path.is_symlink() and path.resolve() == path.absolute(), "output_path", "owned canonical output directory")
    require(not any(path.iterdir()), "output_exists", "refusing partial output or overwrite")


def run(args):
    started = time.perf_counter()
    output_directory(args.output_dir)
    raw_inputs = {name: regular_bytes(getattr(args, name), 32 * 1024 * 1024) for name in ("train0", "train1", "validation", "splits", "origin")}
    config_raw = regular_bytes(args.config, 64 * 1024)
    runtime_raw = regular_bytes(args.runtime_manifest, 16 * 1024 * 1024)
    evidence_raw = regular_bytes(args.input_evidence, 1024 * 1024)
    source, source_raw, config, evidence = validate_source(args, raw_inputs, config_raw, runtime_raw, evidence_raw)
    if __package__:
        from .data import load_data
    else:
        from data import load_data
    dataset = load_data(args.train0, args.train1, args.validation, args.splits, args.origin, expected_native_residual_origin_sha256=source["native_residual_origin_sha256"])
    normalization = numerics.fit_normalization(dataset.train, dataset.train_rows)
    cases = witness_cases(dataset)
    normalization_raw = canonical(normalization)
    closure = {"source_witness_sha256": sha256(source_raw), "source_commit": source["source_commit"], "config_sha256": sha256(config_raw), "runtime_snapshot_sha256": sha256(runtime_raw), "input_evidence_sha256": sha256(evidence_raw), "input_sha256": {name: sha256(raw) for name, raw in raw_inputs.items()}, "splits_sha256": sha256(raw_inputs["splits"]), "normalization_sha256": sha256(normalization_raw), "witness_sha256": sha256(canonical(cases))}
    numerics.configure_cpu()
    observations = {"torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    require(observations["torch_threads"] == observations["torch_interop_threads"] == 1, "runtime_threads", "actual one-thread CPU")
    observations["peak_rss_bytes"] = observed_peak_rss_bytes()
    order = sample_order(dataset.train_rows)
    origin = None
    if args.mode == "train":
        torch.manual_seed(20261014)
        network = numerics.PressureCorrectionNet()
        initial_validation = evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        progress("validation", 0, 200, loss=initial_validation)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        history = []
        for update, indices in enumerate(order, start=1):
            loss, gradient = one_update(network, optimizer, dataset.train, indices, normalization)
            if update % 20 == 0:
                history.append({"update": update, "train_loss": loss, "gradient_norm_before_clip": gradient})
                progress("training", update, 200, loss=loss, gradient_norm=gradient, peak_rss_bytes=observed_peak_rss_bytes())
        final_validation = evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        progress("validation", 200, 200, loss=final_validation)
        validation = [{"update": 0, "loss": initial_validation}, {"update": 200, "loss": final_validation}]
        updates = 200
        raw_trained = None
    else:
        raw_trained = {slot: regular_bytes(getattr(args, slot), OUTPUTS[artifact][1]) for slot, (artifact, _) in TRAINED_INPUTS.items()}
        origin = validate_trained_custody(evidence, raw_trained, args.input_evidence)
        network = numerics.PressureCorrectionNet()
        trained = strict_json(raw_trained["trained_receipt"])
        validate_training_receipt(trained, closure, network, cases, raw_trained)
        require(raw_trained["trained_normalization"] == normalization_raw and raw_trained["trained_splits"] == raw_inputs["splits"], "restore_preprocessing", "exact train-derived scales/splits")
        load_weights(network, raw_trained["trained_weights"], trained["weights_inventory"])
        history = trained["training_history"]
        validation = trained["validation"]
        updates = 0  # No optimizer is constructed anywhere in this branch.
    arrays, clocks = make_predictions(network, dataset, normalization, cases)
    inventory = prediction_inventory(arrays)
    if args.mode == "restore":
        expected = read_npz(raw_trained["trained_predictions"], trained["predictions_inventory"], 8 * 1024 * 1024)
        require(set(expected) == set(arrays) and all(expected[name].dtype == arrays[name].dtype and expected[name].shape == arrays[name].shape and expected[name].tobytes() == arrays[name].tobytes() for name in arrays), "restore_predictions", "bitwise pressure/row witness parity")
    output_directory(args.output_dir)
    if args.mode == "train":
        weights_inventory = save_weights(network, args.output_dir / "weights.npz")
        copy_new(args.output_dir / "normalization.json", normalization_raw)
        copy_new(args.output_dir / "splits.json", raw_inputs["splits"])
        with (args.output_dir / "predictions.npz").open("xb") as stream: np.savez(stream, **arrays)
    else:
        weights_inventory = trained["weights_inventory"]
        for slot in ("trained_weights", "trained_normalization", "trained_splits", "trained_predictions"):
            artifact, _ = TRAINED_INPUTS[slot]
            copy_new(args.output_dir / artifact, raw_trained[slot])
    summary = {"format": "tuldok_result_summary_v1", "title": "Global pressure correction: " + args.mode, "metrics": [{"name": "Optimizer updates", "value": updates, "unit": "updates"}, {"name": "Trained optimizer updates", "value": 200, "unit": "updates"}, {"name": "Parameters", "value": numerics.parameter_count(network), "unit": "parameters"}, {"name": "Final validation loss", "value": validation[-1]["loss"], "unit": "ratio"}, {"name": "Witness cases", "value": len(cases), "unit": "cases"}], "scope": ["Smoke train/validation teacher states only; no final-test or rollout claim.", "Fixed eager float32 CPU model; float64 full finite-volume loss.", "Recipe-reported costs and loss; solver quality and acceleration require separate diagnostics.", "Exact trained-byte zero-update restore." if args.mode == "restore" else "Final fixed update 200; validation is observational."], "checkpoint": "weights.npz"}
    write_json_new(args.output_dir / "summary.json", summary)
    outputs = {name: {**identity(regular_bytes(args.output_dir / name, limit)), "role": role} for name, (role, limit) in OUTPUTS.items() if name != "receipt.json"}
    observations["peak_rss_bytes"] = observed_peak_rss_bytes()
    receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "mode": args.mode, "optimizer_updates": updates, "trained_optimizer_updates": 200, "architecture": numerics.ARCHITECTURE, "parameter_count": numerics.parameter_count(network), "weights_sha256": outputs["weights.npz"]["sha256"], "weights_inventory": weights_inventory, "predictions_sha256": outputs["predictions.npz"]["sha256"], "predictions_inventory": inventory, "witness_cases": cases, "per_case_inference_s": clocks, "closure": closure, "sample_order_sha256": sha256(order.tobytes()), "training_history": history, "validation": validation, "elapsed_s": time.perf_counter() - started, "runtime_observations": observations, "outputs": outputs, "originating_training_run": origin}
    require(sum(spec["bytes"] for spec in outputs.values()) + len(canonical(receipt)) <= 32 * 1024 * 1024, "output_bound", "aggregate outputs")
    write_json_new(args.output_dir / "receipt.json", receipt)
    progress("complete", updates, 200, validation_loss=validation[-1]["loss"], witness_cases=len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "restore"))
    for name in ("train0", "train1", "validation", "splits", "origin", "config", "runtime-manifest", "source-witness", "input-evidence", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for slot in TRAINED_INPUTS:
        parser.add_argument("--" + slot.replace("_", "-"), type=Path)
    args = parser.parse_args()
    try:
        require(args.mode != "restore" or all(getattr(args, slot) is not None for slot in TRAINED_INPUTS), "restore_inputs", "all five originating trained artifacts required")
        require(args.mode != "train" or all(getattr(args, slot) is None for slot in TRAINED_INPUTS), "train_inputs", "trained artifacts not training inputs")
        run(args)
    except (TrainingContractError, numerics.ModelContractError, OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "failed", "error": getattr(error, "code", "training_failure"), "detail": str(error)}, allow_nan=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
