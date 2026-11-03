"""Freeze one real water-trainer update and CPU restore using current owners.

No optimizer or inference is executed here. Original source, packet, weights,
normalization and recorded results remain untouched. Only the bounded update
schedule and current provenance/runtime/custody bindings are freshly produced.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile

from dependency_config import canonical, decode, digest, load, require


def capture_closure(paths):
    return {str(Path(path).resolve()): digest(Path(path)) for path in paths}


def verify_closure(expected):
    require(capture_closure(expected) == expected, "captured training code/input closure changed")


def training_packet(inputs):
    """Repackage original .npy members without decoding or converting any value."""
    manifest = decode(inputs["splits"][1])
    selected = [(i, parent) for i, parent in enumerate(manifest["parents"]) if parent["entry"]["split"] == "train"]
    require(bool(selected), "original training population required")
    old_arrays = {}
    for slot in ("dataset", "dataset_extra"):
        with zipfile.ZipFile(io.BytesIO(inputs[slot][1])) as archive:
            for name in archive.namelist():
                require(name.endswith(".npy") and name[:-4] not in old_arrays, "unique original array member")
                old_arrays[name[:-4]] = archive.read(name)
    arrays, members, parent_mapping = {}, [], []
    for new_index, (old_index, parent) in enumerate(selected):
        parent_mapping.append({"original_index": old_index, "current_index": new_index,
                               "trajectory_id": parent["entry"]["trajectory_id"]})
        for field in ("base", "reference", "rhs", "native_input_residual"):
            old_key, key = f"p{old_index:04d}_{field}", f"p{new_index:04d}_{field}"
            arrays[key] = manifest["arrays"][old_key]
            members.append((key, old_arrays[old_key]))
    manifest["parents"] = [parent for _, parent in selected]
    manifest["arrays"] = arrays
    manifest["reservation"]["selection"] = [{"source": p["source"], "trajectory_id": p["scene"]["trajectory_id"],
        "frames": [f["index"] for f in p["frames"]]} for p in manifest["parents"]]
    manifest["reservation_sha256"] = hashlib.sha256(canonical(manifest["reservation"])).hexdigest()
    manifest["shards"] = []
    for index, slot in enumerate(("dataset", "dataset_extra")):
        chosen = members[index::2]
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as archive:
            for key, raw in chosen:
                archive.writestr(zipfile.ZipInfo(key + ".npy"), raw)
        raw = stream.getvalue()
        inputs[slot] = (inputs[slot][0], raw)
        manifest["shards"].append({"keys": [key for key, _ in chosen], "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    inputs["splits"] = (inputs["splits"][0], canonical(manifest))
    return manifest, parent_mapping


def assemble(selection, output):
    require(Path(__file__).resolve().is_relative_to(selection.source_root("lab")), "selected current material producer required")
    origin = selection.path_for("control_project")
    old_witness_path = selection.artifact("control_source_witness")
    require(old_witness_path == origin / "source-witness.json", "control source witness location")
    old_witness = decode(old_witness_path.read_bytes())
    closure_paths = [Path(__file__), Path(__file__).with_name("dependency_training_entry.py"),
                     Path(__file__).with_name("dependency_config.py"), old_witness_path]
    closure_paths += [origin / name for name in old_witness["sources"]]
    for name, expected in old_witness["sources"].items():
        require(Path(name).name == name and digest(origin / name) == expected,
                "retained trainer source mismatch: " + name)
    runtime = selection.runtime("cpu")
    closure_paths += [Path(runtime["manifest"]), selection.source_root("tuldok") / "torch_runtime.py",
                      selection.source_root("tuldok") / "recipe_process.py",
                      selection.source_root("tuldok") / "recipe_contract.py",
                      selection.source_root("tuldok") / "projects.py"]
    root = selection.path_for("artifact_root")
    output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "fresh contained project output required")
    prefix = output.relative_to(root).as_posix() + "/"
    config = decode((origin / "config.json").read_bytes())
    original_config = copy.deepcopy(config)
    config["id"] = "water-dependency-migration-one-update"
    config["training"].update(updates=1, checkpoints=[0, 1], schedule=[])
    sources = {name: (origin / name).read_bytes() for name in old_witness["sources"]}
    sources["config.json"] = canonical(config)
    sources["runtime-manifest.json"] = Path(runtime["manifest"]).read_bytes()
    sources["custody-validator.py"] = (selection.source_root("tuldok") / "recipe_process.py").read_bytes()
    sources["qualification.json"] = canonical({"status": "fresh_execution_smoke_material",
        "optimizer_updates_planned": 1, "original_numerical_model_and_loss": True,
        "scientific_acceptance_or_model_promotion": False})
    original_recipes = {mode: decode((origin / (mode + "-recipe.json")).read_bytes())
                        for mode in ("train", "restore")}
    input_slots = original_recipes["train"]["inputs"]
    old_manifest = decode((origin / "project.json").read_bytes())
    old_refs = {entry["id"]: entry for entry in old_manifest["data"]}
    closure_paths += [origin / "project.json", origin / "train-recipe.json", origin / "restore-recipe.json"]
    closure_paths += [root / old_refs[slot]["path"] for slot in input_slots]
    captured_closure = capture_closure(closure_paths)
    inputs = {}
    for slot in input_slots:
        path = root / old_refs[slot]["path"]
        raw = path.read_bytes()
        require({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == old_witness["inputs"][slot],
                "retained training input mismatch: " + slot)
        inputs[slot] = (Path(old_refs[slot]["path"]).name, raw)
    require(not any("reserved" in slot or slot.startswith("validation") for slot in inputs),
            "execution smoke must not import heldout inputs")
    packet, parent_mapping = training_packet(inputs)
    direction = decode(inputs["direction_manifest"][1])
    direction["packet_splits_sha256"] = hashlib.sha256(inputs["splits"][1]).hexdigest()
    inputs["direction_manifest"] = (inputs["direction_manifest"][0], canonical(direction))
    config["direction_feedback"]["manifest_sha256"] = hashlib.sha256(inputs["direction_manifest"][1]).hexdigest()
    sources["config.json"] = canonical(config)
    sources["smoke_train.py"] = Path(__file__).with_name("dependency_training_entry.py").read_bytes()
    witness = copy.deepcopy(old_witness)
    witness.update(source_commit=selection.value["sources"]["lab"]["revision"],
                   experiment_sha256=hashlib.sha256(sources["config.json"]).hexdigest(),
                   runtime_manifest_sha256=hashlib.sha256(sources["runtime-manifest.json"]).hexdigest(),
                   custody_validator_sha256=hashlib.sha256(sources["custody-validator.py"]).hexdigest(),
                   inputs={slot: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for slot, (_, raw) in inputs.items()},
                   packet_experiment_sha256=hashlib.sha256(canonical(packet["reservation"])).hexdigest(),
                   sources={name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()
                            if name != "runtime-manifest.json"})
    witness["architecture"]["direction_training_feedback"] = copy.deepcopy(config["direction_feedback"])
    witness["migration_provenance"] = {"original_source_witness_sha256": digest(old_witness_path),
        "original_source_commit": old_witness["source_commit"],
        "selection_sha256": digest(selection.path), "producer_sha256": digest(Path(__file__)),
        "lab_source_content_sha256": selection.value["sources"]["lab"].get("content_sha256"),
        "heldout_parent_count": 0, "diagnostic_population": "train_only"}
    sources["source-witness.json"] = canonical(witness)
    recipes = {}
    for mode, original in original_recipes.items():
        recipe = copy.deepcopy(original)
        recipe.update(id=config["id"] + "-" + mode, name="Water migration " + mode,
                      runtime="managed_torch_python_v1", runtime_device="cuda" if mode == "train" else "cpu",
                      runtime_snapshot_sha256=runtime["manifest_sha256"], entrypoint=prefix + "smoke_train.py",
                      sources=[{"path": prefix + name, "sha256": hashlib.sha256(raw).hexdigest()}
                               for name, raw in sorted(sources.items())])
        outputs = [entry for entry in recipe["outputs"] if not entry["path"].startswith("checkpoint-")]
        template = next(entry for entry in recipe["outputs"] if entry["path"] == "checkpoint-000.npz")
        recipe["outputs"] = outputs + [{**template, "path": f"checkpoint-{update:03d}.npz"} for update in (0, 1)]
        recipe["inputs"] = copy.deepcopy(input_slots)
        arguments = [{"literal": mode}]
        if mode == "restore":
            recipe["inputs"].update({"trained_" + entry["path"].replace("-", "_").replace(".", "_"):
                {"max_bytes": entry["max_bytes"]} for entry in recipe["outputs"] if entry["path"] != "summary.json"})
        for slot in recipe["inputs"]:
            arguments.extend([{"literal": "--" + slot.replace("_", "-")}, {"input": slot}])
        for name in ("config", "runtime-manifest", "source-witness"):
            arguments.extend([{"literal": "--" + name}, {"source": prefix + name + ".json"}])
        arguments.extend([{"literal": "--input-evidence"}, {"input_evidence": True},
                          {"literal": "--output-dir"}, {"output_dir": True}])
        recipe["arguments"] = arguments
        recipes[mode] = recipe
    sys.path.insert(0, str(selection.source_root("tuldok")))
    from recipe_contract import decode_definition
    from projects import manifest as validate_project
    for recipe in recipes.values():
        decode_definition(canonical(recipe))
    manifest = {"schema_version": 2, "id": uuid.uuid4().hex, "name": "Water migration one-update execution qualification",
                "data": [{"id": slot, "label": "Original training " + slot, "path": prefix + "inputs/" + name,
                          "sha256": hashlib.sha256(raw).hexdigest()} for slot, (name, raw) in inputs.items()],
                "recipes": [{"id": mode, "label": "Water " + mode, "path": prefix + mode + "-recipe.json",
                             "sha256": hashlib.sha256(canonical(recipe)).hexdigest()} for mode, recipe in recipes.items()],
                "evidence": [], "requirements": [], "extensions": []}
    validate_project(manifest)
    # Only source actually copied/interpreted here invalidates this material.
    # The complete checkout witness remains independently retained provenance.
    selection.source_root("lab")
    selection.source_root("tuldok")
    verify_closure(captured_closure)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for name, raw in sources.items():
        (output / name).write_bytes(raw)
    for name, raw in inputs.values():
        (output / "inputs" / name).write_bytes(raw)
    for mode, recipe in recipes.items():
        (output / (mode + "-recipe.json")).write_bytes(canonical(recipe))
    (output / "project.json").write_bytes(canonical(manifest))
    restore_bindings = {slot: slot for slot in inputs}
    for entry in recipes["restore"]["outputs"]:
        if entry["path"] != "summary.json":
            slot = "trained_" + entry["path"].replace("-", "_").replace(".", "_")
            restore_bindings[slot] = {"stage": "train", "artifact": entry["path"]}
    stages = [{"name": "train", "recipe_id": "train", "inputs": {slot: slot for slot in inputs}},
              {"name": "restore", "recipe_id": "restore", "inputs": restore_bindings}]
    (output / "stages.json").write_bytes(canonical(stages))
    receipt = {"schema": "rheon-migration-training-material-v1", "status": "prepared_without_execution",
               "project": str(output), "root": str(root), "manifest_path": (output / "project.json").relative_to(root).as_posix(),
               "project_sha256": digest(output / "project.json"), "original_source_witness_sha256": digest(old_witness_path),
               "current_source_witness_sha256": digest(output / "source-witness.json"),
               "runtime": runtime, "original_training": original_config["training"], "current_training": config["training"],
               "normalization_sha256": witness["normalization_sha256"], "input_slots": list(inputs),
               "heldout_inputs_imported": False, "model_source_unchanged": digest(output / "model.py") == digest(origin / "model.py"),
               "trainer_source_unchanged": digest(output / "train.py") == digest(origin / "train.py"),
               "diagnostic_population": "train_only", "heldout_parent_count": 0, "train_parent_mapping": parent_mapping,
               "packet_numeric_values": "original .npy members copied byte-for-byte with new parent keys",
               "diagnostic_entrypoint_sha256": digest(output / "smoke_train.py")}
    receipt["captured_input_closure"] = captured_closure
    receipt["captured_input_closure_sha256"] = hashlib.sha256(canonical(captured_closure)).hexdigest()
    receipt["closure_pre_post_identical"] = True
    (output / "migration-material.json").write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selection = load(args.dependency_config)
    print(json.dumps(assemble(selection, args.output or selection.path_for("migration") / "training-project")))


if __name__ == "__main__":
    main()
