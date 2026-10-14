"""Package verified original pressure observations and diagnostic CNN recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_HASH = "ee384926b3e3326356aa6c9b033b53c3c3236117514c848c2ebfc3a413115c37"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def package(dataset, destination):
    """Native integrity/physics verification precedes captured-data packaging."""
    import numpy as np
    dataset, destination = Path(dataset), Path(destination)
    if destination.exists():
        raise ValueError("Dataset package destination already exists")
    binary = ROOT / "target/release/rheon-surrogate-lab"
    report = json.loads(subprocess.check_output([str(binary), "verify", str(dataset)], text=True))
    if (report.get("status") != "verified" or report.get("cases") != 18
            or report.get("tool", {}).get("source_dirty") is not False
            or report["tool"].get("rheon_commit") != "34235d33c886aa05b5bdf95d920b863d1fb2ad6f"):
        raise ValueError("Original native verification did not qualify this dataset")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["schema"] != "rheon-pressure-dataset-v1" or manifest["per_family"] != 4:
        raise ValueError("Use the original four-seed-per-family development dataset")
    arrays, cases = {}, []
    for split in ("train", "validation", "test"):
        selected = [case for case in manifest["cases"] if case["spec"]["split"] == split]
        for field in ("rhs", "pressure"):
            values = []
            for case in selected:
                spec = case["arrays"][field]
                source = dataset / spec["path"]
                if sha(source) != spec["sha256"] or spec["dtype"] != "<f8" or spec["shape_xyz"] != [8, 7, 6]:
                    raise ValueError("Verified source array changed before packaging")
                values.append(np.fromfile(source, dtype="<f8").reshape(6, 7, 8))
            arrays[split + "_" + field] = np.stack(values)
        cases.extend(case["spec"] for case in selected)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(destination / "dataset.npz", **arrays)
    write(destination / "splits.json", {"schema": "rheon-cnn-splits-v1", "shape_xyz": [8, 7, 6],
          "dtype": "<f8", "rheon_revision": manifest["rheon_commit"],
          "generator_revision": manifest["experiment_commit"], "manifest_sha256": sha(dataset / "manifest.json"),
          "dataset_sha256": sha(destination / "dataset.npz"), "cases": cases,
          "policy": "whole seeds and scenario families; standalone fields; checks excluded"})
    write(destination / "dataset-receipt.json", {"status": "verified", "source_manifest_sha256": sha(dataset / "manifest.json"),
          "captured_dataset_sha256": sha(destination / "dataset.npz"), "splits_sha256": sha(destination / "splits.json"),
          "native_binary_sha256": sha(binary), "verification": report,
          "scope": "Lossless RHS/pressure pairs from original native verification; check cases excluded from model data"})


def definition(command, sources, snapshot):
    inputs = {"dataset": {"max_bytes": 1024**2}, "splits": {"max_bytes": 65536}}
    arguments = [{"literal": command}, {"literal": "--dataset"}, {"input": "dataset"},
                 {"literal": "--splits"}, {"input": "splits"}, {"literal": "--output"}, {"output_dir": True}]
    parameters = {}
    if command == "train":
        parameters["steps"] = {"type": "integer", "min": 1, "max": 1000, "default": 200}
        arguments += [{"literal": "--steps"}, {"parameter": "steps"}]
    else:
        for name, maximum in (("checkpoint", 65536), ("receipt", 1024**2),
                              ("normalization", 65536), ("predictions", 1024**2)):
            inputs[name] = {"max_bytes": maximum}
            arguments += [{"literal": "--" + name}, {"input": name}]
    files = [("weights.npz", "checkpoint", 65536), ("splits.json", "training_splits", 65536),
             ("normalization.json", "preprocessing", 65536), ("predictions.npz", "predictions", 1024**2),
             ("receipt.json", "training_result" if command == "train" else "evaluation_result", 1024**2),
             ("summary.json", "result_summary", 65536)]
    return {"schema_version": 1, "format": "external_recipe_v1", "id": "lab-cnn-" + command,
            "name": "Captured CNN " + command, "version": 1, "runtime": "captured_cpu_python_v1",
            "runtime_snapshot_sha256": snapshot, "sources": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(sources.items())], "entrypoint": "cnn.py", "arguments": arguments,
            "parameters": parameters, "inputs": inputs,
            "outputs": [{"path": name, "role": role, "max_bytes": cap, "required": True} for name, role, cap in files],
            "environment": {}, "resources": {"wall_seconds": 600, "max_output_bytes": 4*1024**2}}


def assemble(package_root, destination, snapshot):
    if len(snapshot) != 64 or any(c not in "0123456789abcdef" for c in snapshot):
        raise ValueError("Supply actual captured CPU snapshot SHA256")
    if sha(ROOT / "scripts/train_cpu.py") != MODEL_HASH:
        raise ValueError("Original CNN source changed")
    sources = {"cnn.py": (ROOT / "tuldok/cnn.py").read_bytes(), "train_cpu.py": (ROOT / "scripts/train_cpu.py").read_bytes()}
    package_root, destination = Path(package_root), Path(destination)
    evidence = json.loads((package_root / "dataset-receipt.json").read_text())
    if (evidence.get("status") != "verified" or evidence["captured_dataset_sha256"] != sha(package_root / "dataset.npz")
            or evidence["splits_sha256"] != sha(package_root / "splits.json")):
        raise ValueError("Captured dataset evidence changed")
    files = dict(sources)
    refs = []
    for name in ("dataset.npz", "splits.json"):
        files["inputs/" + name] = (package_root / name).read_bytes()
        refs.append({"id": name.split(".")[0], "label": "Captured " + name, "path": "inputs/" + name,
                     "sha256": sha(package_root / name)})
    files["evidence/dataset-receipt.json"] = (package_root / "dataset-receipt.json").read_bytes()
    recipes = []
    for command in ("train", "restore"):
        path = "recipes/cnn-" + command + "-v1.json"
        files[path] = (json.dumps(definition(command, sources, snapshot), indent=2) + "\n").encode()
        recipes.append({"id": command, "label": "CNN " + command, "path": path,
                        "sha256": hashlib.sha256(files[path]).hexdigest()})
    manifest = {"schema_version": 1, "id": uuid.uuid4().hex, "name": "Original pressure CNN diagnostic",
                "data": refs, "recipes": recipes,
                "evidence": [{"id": "native-input-evidence", "label": "Original native data verification", "path": "evidence/dataset-receipt.json",
                              "sha256": sha(package_root / "dataset-receipt.json")}],
                "requirements": [{"id": "quality", "label": "Radius-one CNN diagnostic; no pressure quality or speedup qualification"}]}
    files["project.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o400)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("package")
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--destination", required=True)
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--package", required=True)
    assembly.add_argument("--destination", required=True)
    assembly.add_argument("--runtime-snapshot-sha256", required=True)
    args = parser.parse_args()
    if args.command == "package":
        package(args.dataset, args.destination)
    else:
        print(json.dumps(assemble(args.package, args.destination, args.runtime_snapshot_sha256)))


if __name__ == "__main__":
    main()
