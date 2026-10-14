#!/usr/bin/env python3
"""Assemble immutable lab bytes for existing generic Tuldok project contracts.

This assembler never opens a catalog, grants material, submits a Run or trains.
The existing project CLI, catalog and Runs owners govern those actions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
TULDOK_PIN = "f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6"
EXECUTABLES = ("data.py", "model.py", "train.py")
DATA_NAMES = {"train0": "train-0.npz", "train1": "train-1.npz", "validation": "validation.npz", "splits": "splits.json", "origin": "origin.json"}
MIB = 1024 * 1024
OUTPUTS = (("weights.npz", "checkpoint", 8*MIB), ("splits.json", "training_splits", 256*1024), ("normalization.json", "preprocessing", 256*1024), ("receipt.json", "training_result", 512*1024), ("predictions.npz", "predictions", 8*MIB), ("summary.json", "result_summary", 65536))
TRAINED = {"trained_weights": ("weights.npz", 8*MIB), "trained_receipt": ("receipt.json", 512*1024), "trained_normalization": ("normalization.json", 256*1024), "trained_predictions": ("predictions.npz", 8*MIB), "trained_splits": ("splits.json", 256*1024)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path, cap):
    path = Path(path)
    require(path.absolute() == path.resolve() and not path.is_symlink(), "canonical regular input required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "regular input byte limit")
        raw = stream.read(cap+1)
        after = os.fstat(stream.fileno())
    require(len(raw) == before.st_size and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "input changed while reading")
    return raw


def source_commit():
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=LAB, text=True)
    require(not status, "commit qualified source before assembling its reproducible project")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LAB, text=True).strip()


def definition(command, sources, snapshot):
    inputs = {name: {"max_bytes": (256*1024 if name == "splits" else 32*MIB)} for name in DATA_NAMES}
    arguments = [{"literal": command}]
    for name in DATA_NAMES:
        arguments += [{"literal": "--"+name}, {"input": name}]
    for name in ("config", "runtime-manifest", "source-witness"):
        arguments += [{"literal": "--"+name}, {"source": name+".json"}]
    arguments += [{"literal": "--input-evidence"}, {"input_evidence": True}, {"literal": "--output-dir"}, {"output_dir": True}]
    if command == "restore":
        for name, (_artifact, cap) in TRAINED.items():
            inputs[name] = {"max_bytes": cap}
            arguments += [{"literal": "--"+name.replace("_", "-")}, {"input": name}]
    return {"schema_version": 1, "format": "external_recipe_v1", "id": "global-pressure-"+command,
            "name": "Global pressure CPU "+command, "version": 1, "runtime": "captured_cpu_python_v1",
            "runtime_snapshot_sha256": snapshot,
            "sources": [{"path": path, "sha256": sha(raw)} for path, raw in sorted(sources.items())],
            "entrypoint": "train.py", "arguments": arguments, "parameters": {}, "inputs": inputs,
            "outputs": [{"path": path, "role": ("evaluation_result" if command == "restore" and path == "receipt.json" else role), "max_bytes": cap, "required": True} for path, role, cap in OUTPUTS],
            "environment": {}, "resources": {"wall_seconds": 600, "max_output_bytes": 32*MIB}}


def assemble(args):
    from data import load_data, strict_json
    commit = source_commit()
    tuldok = args.tuldok_source.resolve(strict=True)
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tuldok, text=True).strip() == TULDOK_PIN, "exact generic Tuldok pin required")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=tuldok, text=True), "generic Tuldok source must be unchanged")
    sys.path.insert(0, str(tuldok))
    from recipe_contract import decode_definition as generic_definition
    from projects import manifest as generic_manifest
    packet = args.package.resolve(strict=True)
    receipt_raw = read(packet/"dataset-receipt.json", 2*MIB)
    receipt = strict_json(receipt_raw)
    require(receipt["status"] == "verified_lossless_development_packet" and receipt["packager_source_sha256"] == sha(read(HERE/"data.py", 8*MIB)), "qualified packet/source binding")
    require(receipt["train_rows"] == 288 and receipt["validation_rows"] == 96, "complete development row counts")
    inputs = {slot: read(packet/name, 256*1024 if slot == "splits" else 32*MIB) for slot, name in DATA_NAMES.items()}
    for slot, raw in inputs.items():
        spec = receipt["files"][DATA_NAMES[slot]]
        require(canonical(spec) == canonical({"sha256": sha(raw), "bytes": len(raw)}), "qualified input hash/bytes: "+slot)
    require(sum(map(len, inputs.values())) <= 256*MIB, "aggregate input bound")
    admitted = load_data(*(packet/DATA_NAMES[name] for name in DATA_NAMES), expected_native_residual_origin_sha256=receipt["native_residual_origin_sha256"])
    require(len(admitted.train_rows) == 288 and len(admitted.validation_rows) == 96, "fresh numeric reader coverage")
    runtime_raw = read(args.runtime_manifest, 8*MIB)
    runtime = strict_json(runtime_raw)
    require(runtime["profile"] == "captured_cpu_python_v1" and runtime["packages"]["torch"] == "2.8.0+cpu" and runtime["packages"]["numpy"] == "2.3.5", "qualified CPU snapshot packages")
    snapshot = sha(runtime_raw)
    config_raw = read(HERE/"config.json", 65536)
    qualification_raw = read(args.qualification_report, MIB)
    qualification = strict_json(qualification_raw)
    require(qualification["status"] == "passed", "pretraining qualification must pass")
    sources = {name: read(HERE/name, 8*MIB) for name in EXECUTABLES}
    executable_hashes = {name: sha(raw) for name, raw in sources.items()}
    require(qualification["executable_sources"] == executable_hashes, "qualification/current source binding")
    native = strict_json(read(args.native_build_receipt, MIB))
    require(native["status"] == "qualified_build", "qualified native input residual producer")
    native_receipt = native["identity"]["native_receipt"]
    require(native["binary_sha256"] == qualification["evidence"]["native_operator"]["operator_binary_sha256"], "qualified native model/input operator identity")
    sources.update({"config.json": config_raw, "runtime-manifest.json": runtime_raw, "model-qualification.json": qualification_raw})
    witness = {"format": "rheon-global-pressure-training-source-v1", "source_commit": commit,
               "tuldok_revision": TULDOK_PIN, "fixedmodel_format": "rheon-global-pressure-model-v1",
               "runtime_snapshot_sha256": snapshot, "executable_sources": executable_hashes,
               "config_sha256": sha(config_raw), "package_inputs": {slot: {"sha256": sha(raw), "bytes": len(raw)} for slot, raw in inputs.items()},
               "native_qualification": {"base_commit": native_receipt["base_commit"], "patch_sha256": native_receipt["patch_sha256"], "prepared_tree_sha256": native_receipt["prepared_tree_sha256"], "operator_binary_sha256": native["binary_sha256"]},
               "qualification_report_sha256": sha(qualification_raw), "native_residual_origin_sha256": receipt["native_residual_origin_sha256"]}
    sources["source-witness.json"] = canonical(witness)
    require(len(sources) <= 64 and all(len(raw) <= 8*MIB for raw in sources.values()), "source closure byte/count limits")
    require(sum(map(len, sources.values())) + sum(map(len, inputs.values())) + MIB <= 320*MIB, "complete capture bound with bootstrap headroom")
    output = args.output.absolute()
    namespace = LAB/"runs/global-pressure-v1-project"
    require(output == output.resolve() and output.parent == namespace and not output.exists(), "fresh canonical direct project namespace child required")
    files = dict(sources)
    files.update({"inputs/"+DATA_NAMES[slot]: raw for slot, raw in inputs.items()})
    files["evidence/dataset-receipt.json"] = receipt_raw
    files["evidence/model-qualification.json"] = qualification_raw
    recipes = []
    for command in ("train", "restore"):
        value = definition(command, sources, snapshot)
        raw = canonical(value)
        generic_definition(raw)
        path = "recipes/"+command+".json"
        files[path] = raw
        recipes.append({"id": command, "label": "Fixed CPU "+command, "path": path, "sha256": sha(raw)})
    manifest = {"schema_version": 1, "id": uuid.uuid4().hex, "name": "Global pressure correction CPU smoke",
                "data": [{"id": slot, "label": "Verified development "+slot, "path": "inputs/"+DATA_NAMES[slot], "sha256": sha(raw)} for slot, raw in inputs.items()],
                "recipes": recipes,
                "evidence": [{"id": "dataset", "label": "Lossless development packet", "path": "evidence/dataset-receipt.json", "sha256": sha(receipt_raw)}, {"id": "qualification", "label": "Pretraining model qualification", "path": "evidence/model-qualification.json", "sha256": sha(qualification_raw)}],
                "requirements": [{"id": "cpu-snapshot", "label": "Actual qualified captured CPU Python3.12, Torch2.8+cpu and NumPy2.3.5"}, {"id": "scientific-scope", "label": "Exactly200Adamupdates; development only; final reservations sealed; no solver gain claim"}]}
    generic_manifest(manifest)
    files["project.json"] = canonical(manifest)
    namespace.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # claim exclusively; a failed partial project cannot be reused
    for name, raw in files.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o400)
    result = {"status": "assembled_not_admitted", "project_id": manifest["id"], "manifest_sha256": sha(files["project.json"]), "source_commit": commit, "runtime_snapshot_sha256": snapshot,
              "source_bytes": sum(map(len, sources.values())), "input_bytes": sum(map(len, inputs.values())), "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}, "path": str(output)}
    print(json.dumps(result, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--runtime-manifest", type=Path, required=True)
    p.add_argument("--native-build-receipt", type=Path, required=True)
    p.add_argument("--qualification-report", type=Path, required=True)
    p.add_argument("--tuldok-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        assemble(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("project_assembly_failure: "+str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
