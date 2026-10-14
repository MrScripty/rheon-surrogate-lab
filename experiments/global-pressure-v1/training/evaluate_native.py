#!/usr/bin/env python3
"""Isolated native build/residual/oracle and fixed teacher-state diagnostics.

This local diagnostic is outside the generic captured model runtime. It never
generates trajectories, trains a model, advances simulation state or claims speedup.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

LAB = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROBE = LAB / "probes/global-pressure-v1-training"
BASE = "2209012bc5f902693d0f695ec1c2d854126af317"
CAPTURE_SHA256 = "dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d"
SOURCE_FILES = ["probes/global-pressure-v1-training/" + name for name in
                ("Cargo.toml", "Cargo.lock", "build.rs", "src/main.rs")]
SOURCE_FILES += ["experiments/global-pressure-v1/training/evaluate_native.py",
                 "experiments/global-pressure-v1/prepare_dependency.py",
                 "experiments/global-pressure-v1/dependency/manifest.json",
                 "experiments/global-pressure-v1/dependency/source-files.json"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with regular(Path(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON field")
            out[key] = value
        return out
    def invalid(_):
        raise ValueError("nonfinite JSON value")
    def finite(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON value")
        return value
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def regular(path):
    require(not path.is_symlink() and path.is_file(), f"regular file required: {path}")
    require(path.resolve() == path.absolute(), f"symlink/traversal prohibited: {path}")
    return path


def helper():
    source = LAB / "experiments/global-pressure-v1/prepare_dependency.py"
    spec = importlib.util.spec_from_file_location("immutable_native_preparation", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes():
    contract, _, _, _ = helper().load_contract()
    files = SOURCE_FILES + ["experiments/global-pressure-v1/dependency/" + contract["patch_file"]]
    return {name: sha(LAB / name) for name in files}


def new_output(path):
    path = Path(path).absolute()
    require(path.resolve() == path and path.parent == Path("/tmp"),
            "native output must be a fresh direct child of /tmp")
    path.mkdir()
    return path


def build(args):
    require(not args.prepared_source.is_symlink(), "prepared source symlink")
    prepared = args.prepared_source.resolve()
    receipt = helper().verify_prepared(prepared)
    require(receipt["base_commit"] == BASE, "native source pin")
    output = new_output(args.output)
    configuration = f'patch."https://github.com/MrScripty/Rheon.git".rheon.path={json.dumps(str(prepared))}'
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CARGO_", "RHEON_", "RUSTFLAGS", "RUSTC", "RUSTDOC", "RUSTUP_TOOLCHAIN"))}
    for key in ("CARGO_HOME", "RUSTUP_HOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(CARGO_TARGET_DIR="/tmp/global-pressure-training-native-target",
               RAYON_NUM_THREADS="1", RUSTUP_TOOLCHAIN="1.92.0",
               RUSTC=str(args.cargo.parent / "rustc"), RUSTDOC=str(args.cargo.parent / "rustdoc"),
               RUSTC_WRAPPER="", RUSTC_WORKSPACE_WRAPPER="")
    cargo_version = subprocess.check_output([str(args.cargo), "--version"], env=env, text=True).strip()
    rustc_version = subprocess.check_output([env["RUSTC"], "-vV"], env=env, text=True).strip()
    require(cargo_version.startswith("cargo 1.92.0 ") and "release: 1.92.0" in rustc_version,
            "exact Cargo/Rust1.92.0 required")
    common = ["--manifest-path", str(PROBE / "Cargo.toml"), "--config", configuration]
    metadata = strict_json(subprocess.check_output(
        [str(args.cargo), "metadata", *common, "--format-version", "1", "--locked"], env=env, cwd=LAB))
    native = [p for p in metadata["packages"] if p["name"] == "rheon"]
    require(len(native) == 1 and native[0]["source"] is None
            and Path(native[0]["manifest_path"]).resolve() == prepared / "Cargo.toml",
            "actual Cargo graph must select prepared native source")
    require(Path(metadata["workspace_root"]).resolve() == PROBE, "Cargo workspace mismatch")
    metadata_bytes = canonical(metadata)
    (output / "cargo-metadata.json").write_bytes(metadata_bytes)
    command = [str(args.cargo), "build", *common, "--locked", "--release"]
    hashes = source_hashes()
    binding = {"schema": "rheon-global-pressure-training-build-v1", "prepared_source": str(prepared),
               "native_receipt": receipt, "resolved_native_manifest": native[0]["manifest_path"],
               "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
               "cargo_command": command, "source_sha256": hashes,
               "toolchain": {"cargo": cargo_version, "rustc_vV": rustc_version,
                             "rustc_path": env["RUSTC"], "rustdoc_path": env["RUSTDOC"]},
               "configuration": configuration}
    (output / "binding.json").write_bytes(canonical(binding))
    env.update(RHEON_BUILD_BINDING=str(output / "binding.json"),
               RHEON_BUILD_BINDING_SHA256=sha(output / "binding.json"))
    start = time.monotonic()
    subprocess.run(command, env=env, cwd=LAB, check=True, timeout=600)
    subprocess.run([str(args.cargo), "test", *common, "--locked", "--release"],
                   env=env, cwd=LAB, check=True, timeout=600)
    require(helper().verify_prepared(prepared) == receipt and source_hashes() == hashes,
            "source drift during build/tests")
    binary = output / "rheon-global-pressure-training"
    shutil.copyfile(Path(env["CARGO_TARGET_DIR"]) / "release/rheon-global-pressure-training", binary)
    binary.chmod(0o555)
    identity = strict_json(subprocess.check_output([str(binary), "--identity"], timeout=10))
    require(identity["binding"] == binding and identity["native_receipt"] == receipt,
            "binary identity mismatch")
    report = {"status": "qualified_build", "binary_path": str(binary), "binary_sha256": sha(binary),
              "identity": identity, "elapsed_s": time.monotonic() - start}
    (output / "build-receipt.json").write_bytes(canonical(report))
    print(json.dumps(report, sort_keys=True))


def admit_binary(binary):
    binary = Path(binary).absolute()
    receipt = strict_json(regular(binary.parent / "build-receipt.json").read_bytes())
    require(receipt["status"] == "qualified_build" and receipt["binary_path"] == str(binary)
            and sha(binary) == receipt["binary_sha256"], "native binary custody")
    identity = strict_json(subprocess.check_output([str(binary), "--identity"], timeout=10))
    require(identity == receipt["identity"], "native compiled identity")
    binding = identity["binding"]
    require(binding == strict_json(regular(binary.parent / "binding.json").read_bytes()), "build binding")
    require(sha(binary.parent / "cargo-metadata.json") == binding["metadata_sha256"], "build graph hash")
    require(helper().verify_prepared(Path(binding["prepared_source"])) == identity["native_receipt"],
            "prepared source changed")
    require(identity["native_threads"] == 1 and binding["source_sha256"] == source_hashes(),
            "native diagnostic source closure changed")
    return receipt


def native_call(binary, mode, request=None, timeout=60):
    encoded = b"" if request is None else canonical(request)
    require(len(encoded) <= 1024 * 1024, "native request exceeds1MiB")
    started = time.monotonic()
    result = subprocess.run([str(binary), mode], input=encoded, capture_output=True,
                            env={**os.environ, "RAYON_NUM_THREADS": "1"}, timeout=timeout)
    require(result.returncode == 0, "native diagnostic refused: " + result.stderr.decode(errors="replace")[:4096])
    require(len(result.stdout) <= 512 * 1024, "native output exceeds512KiB")
    return strict_json(result.stdout), time.monotonic() - started


def oracle(args):
    before = admit_binary(args.binary)
    report, wall = native_call(args.binary, "--oracle")
    require(admit_binary(args.binary) == before, "source drift during oracle")
    report.update(binary_sha256=before["binary_sha256"], enclosing_wall_s=wall)
    print(json.dumps(report, sort_keys=True))


def case_population():
    """Declare all32 teacher snapshots before training, independently of a witness."""
    trajectories = [f"smoke-train-{seed}" for seed in range(100000, 100012)]
    trajectories += [f"smoke-validation-{seed}" for seed in range(100100, 100104)]
    return [{"case_id": f"{tid}:frame-{frame:03d}", "trajectory_id": tid,
             "frame": frame, "row_index": index * 24 + frame,
             "split": "validation" if "-validation-" in tid else "train"}
            for index, tid in enumerate(trajectories) for frame in (0, 23)]


def read_witness(path, expected_sha256):
    import numpy as np
    path = regular(Path(path).absolute())
    limit = 8 * 1024 * 1024
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    require(0 < len(blob) <= limit and hashlib.sha256(blob).hexdigest() == expected_sha256,
            "trained witness byte identity/size")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
        require(len(members) == 2 and {m.filename for m in members} ==
                {"pressure.npy", "row_indices.npy"}, "numeric witness inventory")
        require(all(m.file_size <= limit for m in members)
                and sum(m.file_size for m in members) <= limit,
                "expanded witness bound")
        for member in members:
            require(member.compress_type == zipfile.ZIP_STORED and not member.flag_bits & 1,
                    "numeric witness compression/encryption policy")
            with archive.open(member) as stream:
                require(stream.read(8) == b"\x93NUMPY\x01\x00", "numeric witness NPY header version")
                length_bytes = stream.read(2)
                require(len(length_bytes) == 2, "numeric witness NPY header length")
                header_size = int.from_bytes(length_bytes, "little")
                require(0 < header_size <= 1024, "numeric witness NPY header bound")
                header_bytes = stream.read(header_size)
                require(len(header_bytes) == header_size, "numeric witness NPY header truncation")
                header = ast.literal_eval(header_bytes.decode("latin1"))
                require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"},
                        "numeric witness NPY header fields")
                shape = (32, 16, 16, 16) if member.filename == "pressure.npy" else (32,)
                types = ("<f8",) if member.filename == "pressure.npy" else ("<i4", "<i8")
                require(header["descr"] in types and header["fortran_order"] is False
                        and type(header["shape"]) is tuple and all(type(n) is int for n in header["shape"])
                        and header["shape"] == shape,
                        "numeric witness NPY header dtype/shape")
                size = math.prod(shape) * (4 if header["descr"] == "<i4" else 8)
                require(member.file_size == 10 + header_size + size, "numeric witness NPY exact payload bytes")
    # The same bounded immutable byte copy is decoded only after every header,
    # dtype, shape product and payload length has been admitted.
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        pressure = archive["pressure"]
        rows = archive["row_indices"]
    require(pressure.dtype.str == "<f8" and pressure.shape == (32, 16, 16, 16),
            "pressure witness dtype/shape")
    require(rows.dtype.str in ("<i4", "<i8") and rows.shape == (32,),
            "numeric row witness dtype/shape")
    require(rows.tolist() == [case["row_index"] for case in case_population()],
            "fixed32 witness population/order differs")
    return pressure


def capture_manifest(root):
    root = Path(root).absolute()
    require(root.resolve() == root, "capture root symlink/traversal")
    path = regular(root / "manifest.json")
    require(sha(path) == CAPTURE_SHA256, "immutable complete development capture identity")
    origin = strict_json(path.read_bytes())
    require(origin["status"] == "development_complete" and origin["fixture"] is False
            and origin["preset"] == "smoke" and len(origin["trajectories"]) == 16,
            "development capture eligibility")
    expected = list(dict.fromkeys(case["trajectory_id"] for case in case_population()))
    require([t["trajectory_id"] for t in origin["trajectories"]] == expected
            and all(len(t["frames"]) == 24 for t in origin["trajectories"]),
            "complete reserved development trajectories required")
    require(helper().verify_prepared(Path(origin["provenance"]["prepared_source"])) ==
            origin["provenance"]["native_receipt"], "capture prepared source identity")
    return origin


def capture_array(root, declaration):
    import numpy as np
    path = Path(root) / declaration["path"]
    require(Path(root) in path.absolute().parents and ".." not in path.parts,
            "capture array path escape")
    path = regular(path)
    require(path.stat().st_size == declaration["bytes"] and sha(path) == declaration["sha256"],
            "capture raw byte identity")
    array = np.fromfile(path, dtype=declaration["dtype"])
    require(array.size == math.prod(declaration["shape_xyz"])
            and np.isfinite(array).all(), "capture array shape/finiteness")
    return array


def evaluate(args):
    import numpy as np
    start = time.monotonic()
    before = admit_binary(args.binary)
    require(sha(args.receipt) == args.receipt_sha256, "actual trained receipt identity")
    training = strict_json(regular(args.receipt).read_bytes())
    require(training.get("schema") == "rheon-global-pressure-training-receipt-v1"
            and training.get("status") == "complete" and training.get("mode") in ("train", "restore"),
            "qualified scientific training/restore receipt required")
    require(sha(args.weights) == args.weights_sha256, "actual trained weight bytes")
    # The exact receipt field names are supplied by the one scientific trainer.
    require(training.get("weights_sha256") == args.weights_sha256
            and training.get("predictions_sha256") == args.predictions_sha256,
            "trained receipt does not bind weights/predictions")
    require(type(training.get("trained_optimizer_updates")) is int
            and training["trained_optimizer_updates"] == 200
            and type(training.get("optimizer_updates")) is int
            and training["optimizer_updates"] == (200 if training["mode"] == "train" else 0),
            "exact200 trained updates/zero-update restore required")
    pressure = read_witness(args.predictions, args.predictions_sha256)
    require(training.get("witness_cases") == case_population(), "receipt fixed32 population differs")
    require(training["predictions_inventory"]["pressure"]["sha256"] ==
            hashlib.sha256(pressure.tobytes(order="C")).hexdigest(), "trained raw pressure identity")
    inference = training["per_case_inference_s"]
    require(type(inference) is list and len(inference) == 32 and
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in inference),
            "complete bounded model inference timing diagnostics")
    manifest = capture_manifest(args.capture)
    trajectories = {t["trajectory_id"]: t for t in manifest["trajectories"]}
    output = new_output(args.output)
    (output / "population.json").write_bytes(canonical(case_population()))
    consumed = {}
    records = []
    for index, case in enumerate(case_population()):
        require(time.monotonic() - start < 600, "teacher diagnostics600s limit; preserve partial evidence")
        trajectory = trajectories[case["trajectory_id"]]
        frame = trajectory["frames"][case["frame"]]
        require(frame["index"] == case["frame"], "snapshot frame index")
        previous = trajectory["initial"]["pressure"] if case["frame"] == 0 else (
            trajectory["frames"][case["frame"] - 1]["arrays"]["pressure"])
        require(frame["previous_pressure"] == previous["path"], "previous accepted pressure lineage")
        declarations = {"original_rhs": frame["arrays"]["rhs"],
                        "reference_pressure": frame["arrays"]["pressure"],
                        "previous_pressure": previous}
        values = {name: capture_array(args.capture, entry).tolist()
                  for name, entry in declarations.items()}
        velocity = [capture_array(args.capture, frame["arrays"]["pre_" + axis]).tolist()
                    for axis in ("u", "v", "w")]
        for entry in [*declarations.values(), *(frame["arrays"]["pre_" + axis] for axis in ("u", "v", "w"))]:
            consumed[entry["path"]] = entry["sha256"]
        candidate = pressure[index].reshape(-1)
        candidate_error = None if np.isfinite(candidate).all() else "prediction_nonfinite"
        request = {"schema": "rheon-global-pressure-teacher-case-v1", "counts_xyz": [16] * 3,
                   "spacing_m": [h / 16 for h in trajectory["geometry"]["extents_m"]],
                   "density_kg_m3": trajectory["geometry"]["density_kg_m3"],
                   "dt_s": trajectory["reservation_entry"]["dt_s"], **values,
                   "pre_velocity": velocity,
                   "candidate_pressure": candidate.tolist() if candidate_error is None else None,
                   "candidate_error": candidate_error}
        result, wall = native_call(args.binary, "--evaluate", request,
                                   timeout=min(60, max(0.1, 600 - (time.monotonic() - start))))
        require(result["schema"] == "rheon-global-pressure-teacher-diagnostic-v1"
                and result["state_advanced"] is False, "native diagnostic result contract")
        records.append({**case, "split": trajectory["reservation_entry"]["split"],
                        "prediction_raw_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                        "recorded_model_inference_s": inference[index],
                        "enclosing_native_process_wall_s": wall, **result})
        with (output / "cases.jsonl").open("ab") as journal:
            journal.write(canonical(records[-1]) + b"\n")
            journal.flush()
            os.fsync(journal.fileno())
    require(len(records) == 32 and admit_binary(args.binary) == before, "complete diagnostic/native closure")
    require(sha(args.receipt) == args.receipt_sha256 and sha(args.weights) == args.weights_sha256
            and sha(args.predictions) == args.predictions_sha256
            and sha(args.capture / "manifest.json") == CAPTURE_SHA256,
            "trained/capture identity changed during diagnostic")
    for relative, expected in consumed.items():
        require(sha(args.capture / relative) == expected, "selected reference arrays changed")
    report = {"schema": "rheon-global-pressure-teacher-evaluation-v1", "status": "complete_diagnostic",
              "case_population": case_population(), "cases": records,
              "scope": "fixed32 teacher-state development snapshots; no autonomous rollout/generalization/acceleration qualification",
              "capture_manifest_sha256": CAPTURE_SHA256, "training_receipt_sha256": args.receipt_sha256,
              "weights_sha256": args.weights_sha256, "predictions_sha256": args.predictions_sha256,
              "build_receipt_sha256": sha(args.binary.parent / "build-receipt.json"),
              "native_identity": before["identity"], "consumed_raw_arrays": consumed,
              "elapsed_s": time.monotonic() - start,
              "model_inference_and_generic_custody_cost_included": False,
              "cost_scope": "native process startup/JSON exchange/solver/correction/fallback/physics enclosed; model inference independently recorded from trained receipt; generic custody excluded. Separate stage times do not prove an enclosing end-to-end speedup"}
    (output / "evaluation.json").write_bytes(canonical(report))
    print(json.dumps({"output": str(output / "evaluation.json"), "cases": len(records),
                      "raw_qualified": sum(c["raw_learned"]["qualified"] for c in records),
                      "corrected_qualified": sum(c["learned_jacobi"]["qualified"] for c in records),
                      "elapsed_s": report["elapsed_s"], "scope": report["scope"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ("cargo", "prepared-source", "output"):
        b.add_argument("--" + name, type=Path, required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--binary", type=Path, required=True)
    sub.add_parser("population")
    e = sub.add_parser("evaluate")
    for name in ("binary", "capture", "predictions", "receipt", "weights", "output"):
        e.add_argument("--" + name, type=Path, required=True)
    for name in ("predictions-sha256", "receipt-sha256", "weights-sha256"):
        e.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "population":
        print(json.dumps(case_population(), sort_keys=True))
    else:
        {"build": build, "oracle": oracle, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
