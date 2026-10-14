#!/usr/bin/env python3
"""Configured managed CPU training and exact zero-update trained-state restore."""
import argparse
import io
import platform
import re
from pathlib import Path
import sys
import time

import numpy as np
import torch

from contract import HERE, MIB, canonical, decode, experiment, load_module, read, require, sha, write
import data
import model_profile

DATA = {"train0": "train-0.npz", "train1": "train-1.npz", "validation": "validation.npz", "splits": "splits.json", "origin": "origin.json"}


def outputs(config):
    spec = {"weights.npz": ("checkpoint", 2 * MIB), "predictions.npz": ("predictions", 2 * MIB), "normalization.json": ("preprocessing", 256 * 1024), "splits.json": ("training_splits", 2 * MIB), "history.json": ("training_diagnostics", MIB), "receipt.json": ("training_result", MIB), "summary.json": ("result_summary", 64 * 1024)}
    for update in range(0, config["training"]["updates"] + 1, config["training"]["checkpoint_interval"]):
        spec[f"checkpoint-{update:03d}.npz"] = ("checkpoint", 2 * MIB)
    return spec


def trained_inputs(config):
    return {"trained_" + name.replace("-", "_").replace(".", "_"): (name, role, cap) for name, (role, cap) in outputs(config).items() if name != "summary.json"}


def order(rows, config):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["trajectory_id"], []).append(index)
    parents = list(groups.values())
    rng = np.random.Generator(np.random.PCG64(config["training"]["seed"] + 1))
    result = []
    for _ in range(config["training"]["updates"]):
        batch = []
        for _ in range(config["training"]["batch_size"]):
            parent = parents[int(rng.integers(len(parents)))]
            batch.append(parent[int(rng.integers(len(parent)))])
        result.append(batch)
    return np.asarray(result, dtype="<i8")


def cases(dataset):
    return [{"case_id": row["row_id"], "trajectory_id": row["trajectory_id"], "frame": row["frame"], "split": row["split"], "row_index": index} for index, row in enumerate(dataset.train_rows + dataset.validation_rows) if row["frame"] in (0, 23)]


def predictions(network, dataset, normalization, helpers, population):
    network.eval()
    pressures, clocks = [], []
    with torch.no_grad():
        for case in population:
            index = case["row_index"]
            arrays = dataset.train if index < len(dataset.train_rows) else dataset.validation
            local = index if index < len(dataset.train_rows) else index - len(dataset.train_rows)
            started = time.perf_counter()
            value = helpers.prediction(network, helpers.tensors(arrays, [local], labels=False), normalization)
            require(bool(torch.isfinite(value).all()), "prediction_nonfinite", case["case_id"])
            pressures.append(value[0, 0].numpy().copy())
            clocks.append(time.perf_counter() - started)
    return {"pressure": np.asarray(pressures, dtype="<f8"), "row_indices": np.asarray([c["row_index"] for c in population], dtype="<i8")}, clocks


def inventory(arrays):
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape), "bytes": value.nbytes, "sha256": sha(value.tobytes())} for name, value in arrays.items()}


def custody(evidence, raw_trained, declared, path, expected_validator):
    require(sha(read(path, MIB)) == expected_validator, "custody_source", "exact selected Tuldok validator")
    owner = load_module("local_pressure_cold_custody", path)
    shared = None
    for slot, (name, role, _cap) in declared.items():
        entry = evidence[slot]
        require(set(entry) == {"kind", "source"} and entry["kind"] == "run_artifact", "trained_origin", "typed managed artifact")
        source = owner.validate_artifact_source(entry["source"], sha(raw_trained[slot]), len(raw_trained[slot]))
        require(source["artifact_name"] == name and source["artifact_role"] == role, "artifact_role", name)
        origin = {key: source[key] for key in ("catalog_id", "run_id", "attempt_id", "request_sha256", "material_sha256", "custody_receipt_sha256", "custody_identity", "custody_receipt_base64")}
        require(shared is None or origin == shared, "mixed_origin", "one completed/drained training Run")
        shared = origin
    return shared


def project_input(slot, raw, declaration, entry):
    require(declaration == {"sha256": sha(raw), "bytes": len(raw)}, "input_identity", slot)
    require(type(entry) is dict and set(entry) == {"project_id", "manifest_sha256", "reference_id", "path", "measured_sha256"}
            and all(type(value) is str and value for value in entry.values()), "input_evidence_schema", slot)
    require(re.fullmatch(r"[a-f0-9]{64}", entry["manifest_sha256"]) and entry["measured_sha256"] == sha(raw)
            and entry["reference_id"] == slot, "input_evidence_identity", slot)
    relative = Path(entry["path"])
    require(not relative.is_absolute() and relative.parts and relative.as_posix() == entry["path"]
            and all(part not in (".", "..") for part in relative.parts), "input_evidence_path", slot)
    return entry["project_id"], entry["manifest_sha256"]


def run(args):
    started = time.perf_counter()
    config_raw = read(args.config, MIB)
    config = experiment(decode(config_raw))
    source_raw = read(args.source_witness, 2 * MIB)
    source = decode(source_raw)
    require(source["schema"] == "rheon-local-pressure-source-v1" and source["experiment_sha256"] == sha(config_raw), "source_config", "frozen experiment")
    for name, expected in source["sources"].items():
        require(Path(name).name == name and sha(read(HERE / name, 4 * MIB)) == expected, "source_identity", name)
    runtime_raw = read(args.runtime_manifest, 16 * MIB)
    runtime = decode(runtime_raw)
    require(sha(runtime_raw) == source["runtime_manifest_sha256"] and runtime["profile"] == "captured_cpu_python_v1" and runtime["packages"]["torch"] == torch.__version__ == "2.8.0+cpu" and runtime["packages"]["numpy"] == np.__version__ == "2.3.5" and torch.version.cuda is None and runtime["python_version"] == f"{sys.version_info.major}.{sys.version_info.minor}", "runtime_identity", "actual captured CPU packages")
    raw_inputs = {slot: read(getattr(args, slot), 4 * MIB if slot in ("splits", "origin") else 32 * MIB) for slot in DATA}
    evidence = decode(read(args.input_evidence, MIB))
    declared = trained_inputs(config) if args.mode == "restore" else {}
    require(set(evidence) == set(DATA) | set(declared), "input_evidence", "complete typed slots")
    project = None
    for slot, raw in raw_inputs.items():
        entry = evidence[slot]
        owner = project_input(slot, raw, source["inputs"][slot], entry)
        require(project is None or owner == project, "mixed_project", "one source project")
        project = owner
    dataset = data.load(blobs={DATA[slot]: raw for slot, raw in raw_inputs.items()}, expected_residual=source["native_residual_origin_sha256"])
    require(canonical(dataset.config) == config_raw, "packet_config", "same selected experiment")
    model, helpers = model_profile.load()
    model.configure_cpu()
    torch.manual_seed(config["training"]["seed"])
    network = model.PressureCorrectionNet()
    normalization = model.fit_normalization(dataset.train, dataset.train_rows)
    normalization_raw = canonical(normalization)
    require(sha(normalization_raw) == source["normalization_sha256"], "normalization_identity", "measured train-only preprocessing")
    sample_order = order(dataset.train_rows, config)
    population = cases(dataset)
    closure = {"source_witness_sha256": sha(source_raw), "experiment_sha256": sha(config_raw), "runtime_manifest_sha256": sha(runtime_raw), "inputs": source["inputs"], "normalization_sha256": sha(normalization_raw), "sample_order_sha256": sha(sample_order.tobytes())}
    output = args.output_dir.absolute()
    require(output == output.resolve() and output.is_dir() and not any(output.iterdir()), "output_exists", "empty owned output")
    history, checkpoint_specs, origin = [], {}, None
    def checkpoint(update):
        name = f"checkpoint-{update:03d}.npz"
        tensors = helpers.save_weights(network, output / name)
        raw = read(output / name)
        checkpoint_specs[name] = {"update": update, "sha256": sha(raw), "bytes": len(raw), "inventory": tensors}
    if args.mode == "train":
        checkpoint(0)
        initial_loss = helpers.evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        optimizer = torch.optim.Adam(network.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        for update, indices in enumerate(sample_order, 1):
            require(time.perf_counter() - started < config["resources"]["wall_seconds"] - 10, "training_deadline", "stop before publication reserve")
            helpers.observed_peak_rss_bytes()
            loss, gradient = helpers.one_update(network, optimizer, dataset.train, indices, normalization)
            if update % config["training"]["checkpoint_interval"] == 0:
                checkpoint(update)
                history.append({"update": update, "loss": loss, "gradient_norm_before_clip": gradient})
                helpers.progress("training", update, config["training"]["updates"], loss=loss)
        validation = {"initial": initial_loss, "final": helpers.evaluate(network, dataset.validation, dataset.validation_rows, normalization)}
        write(output / "history.json", {"history": history, "checkpoints": checkpoint_specs, "validation": validation, "scope": "observational states; final update alone is candidate; no optimizer resume"})
        final_name = f'checkpoint-{config["training"]["updates"]:03d}.npz'
        (output / "weights.npz").write_bytes(read(output / final_name))
        predicted, clocks = predictions(network, dataset, normalization, helpers, population)
        with (output / "predictions.npz").open("xb") as stream:
            np.savez(stream, **predicted)
        prediction_spec = inventory(predicted)
        write(output / "normalization.json", normalization)
        (output / "splits.json").write_bytes(raw_inputs["splits"])
    else:
        raw_trained = {slot: read(getattr(args, slot), cap) for slot, (_name, _role, cap) in declared.items()}
        origin = custody(evidence, raw_trained, declared, args.input_evidence.with_name("recipe_process.py"), source["custody_validator_sha256"])
        trained = decode(raw_trained["trained_receipt_json"])
        require(trained["schema"] == "rheon-local-pressure-training-receipt-v1" and trained["status"] == "complete" and trained["mode"] == "train" and type(trained["optimizer_updates"]) is int and trained["optimizer_updates"] == config["training"]["updates"] and trained["closure"] == closure and trained["cases"] == population, "restore_identity", "same trained experiment and candidate")
        for slot, (name, role, _cap) in declared.items():
            if name != "receipt.json":
                require(trained["outputs"][name] == {"sha256": sha(raw_trained[slot]), "bytes": len(raw_trained[slot]), "role": role}, "trained_hash", name)
        require(raw_trained["trained_normalization_json"] == normalization_raw and raw_trained["trained_splits_json"] == raw_inputs["splits"], "restore_preprocessing", "exact train-only scales and splits")
        observed = decode(raw_trained["trained_history_json"])
        checkpoint_specs = observed["checkpoints"]
        expected_updates = list(range(0, config["training"]["updates"] + 1, config["training"]["checkpoint_interval"]))
        require(set(checkpoint_specs) == {f"checkpoint-{u:03d}.npz" for u in expected_updates} and [x["update"] for x in observed["history"]] == expected_updates[1:], "restore_history", "all observed states")
        initial_inventory = helpers.state_inventory(network)
        for update in expected_updates:
            name = f"checkpoint-{update:03d}.npz"
            raw = raw_trained["trained_" + name.replace("-", "_").replace(".", "_")]
            spec = checkpoint_specs[name]
            require(spec["update"] == update and spec["sha256"] == sha(raw) and spec["bytes"] == len(raw) and set(spec["inventory"]) == set(initial_inventory), "restore_checkpoint", name)
            for key in initial_inventory:
                require(spec["inventory"][key]["shape"] == initial_inventory[key]["shape"] and spec["inventory"][key]["dtype"] == "<f4", "restore_shape", key)
            helpers.read_npz(raw, spec["inventory"], 2 * MIB)
        require(checkpoint_specs["checkpoint-000.npz"]["inventory"] == initial_inventory, "restore_initialization", "seeded initial state")
        final_name = f'checkpoint-{config["training"]["updates"]:03d}.npz'
        require(raw_trained["trained_weights_npz"] == raw_trained["trained_" + final_name.replace("-", "_").replace(".", "_")], "restore_candidate", "final state only")
        helpers.load_weights(network, getattr(args, "trained_weights_npz"), checkpoint_specs[final_name]["inventory"])
        predicted, clocks = predictions(network, dataset, normalization, helpers, population)
        prediction_spec = inventory(predicted)
        require(prediction_spec == trained["prediction_inventory"], "restore_prediction", "fresh exact numeric parity")
        helpers.read_npz(raw_trained["trained_predictions_npz"], prediction_spec, 2 * MIB)
        for slot, (name, _role, _cap) in declared.items():
            if name != "receipt.json":
                (output / name).write_bytes(raw_trained[slot])
        history, validation = observed["history"], observed["validation"]
    write(output / "summary.json", {"mode": args.mode, "optimizer_updates": config["training"]["updates"] if args.mode == "train" else 0, "candidate_update": config["training"]["updates"], "validation": validation, "solver_acceleration_claim": False})
    artifact_specs = {name: {"sha256": sha(read(output / name, cap)), "bytes": len(read(output / name, cap)), "role": role} for name, (role, cap) in outputs(config).items() if name != "receipt.json"}
    receipt = {"schema": "rheon-local-pressure-training-receipt-v1", "status": "complete", "mode": args.mode, "optimizer_updates": config["training"]["updates"] if args.mode == "train" else 0, "candidate_update": config["training"]["updates"], "closure": closure, "cases": population, "prediction_inventory": prediction_spec, "per_case_inference_s": clocks, "outputs": artifact_specs, "originating_training_run": origin, "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "peak_rss_bytes": helpers.observed_peak_rss_bytes()}, "elapsed_s": time.perf_counter() - started}
    require(sum(x["bytes"] for x in artifact_specs.values()) + len(canonical(receipt)) <= config["resources"]["max_output_bytes"], "output_budget", "aggregate complete outputs")
    write(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "restore"))
    for name in (*DATA, "config", "runtime-manifest", "source-witness", "input-evidence", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    # The option vocabulary is fixed; config selects which declared checkpoints
    # are required. No originating artifact is guessed from a filesystem path.
    for name in ("weights.npz", "predictions.npz", "normalization.json", "splits.json", "history.json", "receipt.json", *(f"checkpoint-{u:03d}.npz" for u in range(201))):
        slot = "trained_" + name.replace("-", "_").replace(".", "_")
        parser.add_argument("--" + slot.replace("_", "-"), dest=slot, type=Path)
    args = parser.parse_args()
    config = experiment(decode(read(args.config, MIB)))
    declared = trained_inputs(config)
    require(args.mode != "restore" or all(getattr(args, slot) is not None for slot in declared), "restore_inputs", "all configured originating artifacts required")
    if args.mode == "train":
        require(all(value is None for key, value in vars(args).items() if key.startswith("trained_")), "train_inputs", "fresh training consumes no trained state")
    run(args)


if __name__ == "__main__":
    main()
