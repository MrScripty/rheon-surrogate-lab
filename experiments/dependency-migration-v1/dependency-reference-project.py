"""Freeze measured scientific material into existing generic Tuldok recipes."""
import io
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

import numpy as np
import torch

from contract import HERE, LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import data
import model_profile
import train

TULDOK_PIN = "f7543fb78f6f1cb52e45dd315efa17a4b2fe6fa6"


def output_projection(config, weight_bytes, splits_bytes):
    checkpoints = 1 + config["training"]["updates"] // config["training"]["checkpoint_interval"]
    projected = (checkpoints + 1) * weight_bytes + 2 * MIB + splits_bytes + MIB + 2 * MIB
    require(weight_bytes <= 2 * MIB and projected <= config["resources"]["max_output_bytes"], "output_projection", "complete configured history and publication reserve")
    return projected


def definition(config, sources, snapshot, mode):
    inputs = {slot: {"max_bytes": 4 * MIB if slot in ("splits", "origin") else 32 * MIB} for slot in train.DATA}
    arguments = [{"literal": mode}]
    for slot in inputs:
        arguments += [{"literal": "--" + slot}, {"input": slot}]
    for name in ("config", "runtime-manifest", "source-witness"):
        arguments += [{"literal": "--" + name}, {"source": name + ".json"}]
    arguments += [{"literal": "--input-evidence"}, {"input_evidence": True}, {"literal": "--output-dir"}, {"output_dir": True}]
    if mode == "restore":
        for slot, (_name, _role, cap) in train.trained_inputs(config).items():
            inputs[slot] = {"max_bytes": cap}
            arguments += [{"literal": "--" + slot.replace("_", "-")}, {"input": slot}]
    return {"schema_version": 1, "format": "external_recipe_v1", "id": config["id"] + "-" + mode, "name": "Configured CPU " + mode, "version": 1, "runtime": "captured_cpu_python_v1", "runtime_snapshot_sha256": snapshot, "sources": [{"path": name, "sha256": sha(raw)} for name, raw in sorted(sources.items())], "entrypoint": "train.py", "arguments": arguments, "parameters": {}, "inputs": inputs, "outputs": [{"path": name, "role": "evaluation_result" if mode == "restore" and name == "receipt.json" else role, "max_bytes": cap, "required": True} for name, (role, cap) in train.outputs(config).items()], "environment": {}, "resources": config["resources"]}


def assemble(packet, runtime_manifest, tuldok, output):
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=LAB), "uncommitted_source", "commit qualified source before freezing the project")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tuldok, text=True).strip() == TULDOK_PIN and not subprocess.check_output(["git", "status", "--porcelain"], cwd=tuldok), "tuldok_source", "clean pinned generic owner")
    dataset = data.load(packet)
    config = dataset.config
    packet_receipt = decode(read(packet / "dataset-receipt.json", 2 * MIB))
    require(packet_receipt["status"] == "verified_lossless_development" and packet_receipt["reader_source_sha256"] == sha(read(HERE / "data.py", MIB)), "packet_reader", "qualified fresh reader")
    inputs = {slot: read(packet / name, 4 * MIB if slot in ("splits", "origin") else 32 * MIB) for slot, name in train.DATA.items()}
    for slot, raw in inputs.items():
        require(packet_receipt["files"][train.DATA[slot]] == {"sha256": sha(raw), "bytes": len(raw)}, "packet_identity", slot)
    runtime_raw = read(runtime_manifest, 16 * MIB)
    runtime = decode(runtime_raw)
    require(runtime["profile"] == "captured_cpu_python_v1" and runtime["packages"]["torch"] == torch.__version__ == "2.8.0+cpu" and runtime["packages"]["numpy"] == np.__version__ == "2.3.5" and torch.version.cuda is None, "runtime_profile", "actual qualified CPU packages")
    model, helpers = model_profile.load()
    model.configure_cpu()
    torch.manual_seed(config["training"]["seed"])
    with patch.object(torch.optim.Optimizer, "__init__", side_effect=RuntimeError("qualification cannot construct an optimizer")):
        network = model.PressureCorrectionNet()
        normalization = model.fit_normalization(dataset.train, dataset.train_rows)
        sample = helpers.tensors(dataset.train, [1], labels=False)
        with torch.no_grad():
            pressure = helpers.prediction(network, sample, normalization)
            zero = model.predict_pressure(network, torch.zeros_like(sample["native_input_residual"]), sample["base"], sample["spacing"], sample["rho"], sample["dt"], normalization)
        require(torch.isfinite(pressure).all().item() and torch.equal(zero, sample["base"]), "model_qualification", "finite fresh pressure and exact zero-residual bypass")
        state = io.BytesIO()
        np.savez(state, **{name: value.detach().numpy() for name, value in network.state_dict().items()})
        weight_bytes = len(state.getvalue())
    projected = output_projection(config, weight_bytes, len(inputs["splits"]))
    qualification = {"schema": "rheon-local-pressure-qualification-v1", "status": "passed", "optimizer_constructed": False, "optimizer_updates": 0, "normalization_sha256": sha(canonical(normalization)), "train_parent_ids": normalization["training_provenance"]["trajectory_ids"], "zero_residual_exact": True, "initialized_weight_bytes": weight_bytes, "conservative_output_projection_bytes": projected, "candidate_update": config["training"]["updates"], "model_arithmetic": "unchanged groupnorm-v1; pretraining numerical suite required by bootstrap"}
    sources = {name: read(HERE / name, 2 * MIB) for name in ("contract.py", "data.py", "model_profile.py", "train.py")}
    sources["gn_model.py"] = read(LAB / "experiments/global-pressure-v1/groupnorm-v1/model.py", MIB)
    for name in ("model", "train", "data"):
        sources["original_" + name + ".py"] = read(LAB / "experiments/global-pressure-v1/training" / (name + ".py"), MIB)
    sources["historical-reservation.json"] = read(LAB / "docs/plans/global-pressure-correction-v1/reservation.json", MIB)
    sources["config.json"] = canonical(config)
    sources["runtime-manifest.json"] = runtime_raw
    sources["qualification.json"] = canonical(qualification)
    witness = {"schema": "rheon-local-pressure-source-v1", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LAB, text=True).strip(), "tuldok_revision": TULDOK_PIN, "experiment_sha256": sha(sources["config.json"]), "runtime_manifest_sha256": sha(runtime_raw), "sources": {name: sha(raw) for name, raw in sources.items() if name != "runtime-manifest.json"}, "inputs": {slot: {"sha256": sha(raw), "bytes": len(raw)} for slot, raw in inputs.items()}, "normalization_sha256": sha(canonical(normalization)), "native_residual_origin_sha256": packet_receipt["native_residual_origin_sha256"], "custody_validator_sha256": sha(read(tuldok / "recipe_process.py", MIB)), "qualification_sha256": sha(canonical(qualification))}
    sources["source-witness.json"] = canonical(witness)
    require(sum(map(len, sources.values())) + sum(map(len, inputs.values())) + MIB <= 320 * MIB, "capture_budget", "complete generic source and input closure")
    sys.path.insert(0, str(tuldok))
    from recipe_contract import decode_definition
    from projects import manifest as project_manifest
    output = fresh(output)
    for name, raw in sources.items():
        (output / name).write_bytes(raw)
    (output / "inputs").mkdir()
    for slot, raw in inputs.items():
        (output / "inputs" / train.DATA[slot]).write_bytes(raw)
    recipes = []
    for mode in ("train", "restore"):
        recipe = canonical(definition(config, sources, sha(runtime_raw), mode))
        decode_definition(recipe)
        (output / (mode + "-recipe.json")).write_bytes(recipe)
        recipes.append({"id": mode, "label": "Configured CPU " + mode, "path": mode + "-recipe.json", "sha256": sha(recipe)})
    manifest = {"schema_version": 1, "id": uuid.uuid4().hex, "name": config["id"], "data": [{"id": slot, "label": "Verified development " + slot, "path": "inputs/" + train.DATA[slot], "sha256": sha(raw)} for slot, raw in inputs.items()], "recipes": recipes, "evidence": [], "requirements": []}
    project_manifest(manifest)
    write(output / "project.json", manifest)
    stages = [{"name": "train", "recipe_id": "train", "inputs": {slot: slot for slot in train.DATA}}, {"name": "restore", "recipe_id": "restore", "inputs": {**{slot: slot for slot in train.DATA}, **{slot: {"stage": "train", "artifact": name} for slot, (name, _role, _cap) in train.trained_inputs(config).items()}}}]
    write(output / "stages.json", stages)
    return {"project": str(output), "source_witness_sha256": sha(sources["source-witness.json"]), "runtime_snapshot_sha256": sha(runtime_raw), "qualification": qualification}
