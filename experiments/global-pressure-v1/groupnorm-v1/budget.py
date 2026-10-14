#!/usr/bin/env python3
"""Pretraining output envelope; initialized serializations are never trained outputs.

The trainer owns the flat output inventory. Fixed-shape, uncompressed numeric
files are measured with its real serializer. Variable JSON outputs reserve their
entire declared caps, so aggregate admission does not depend on optimistic logs.
No optimizer, grant, Run, checkpoint selection or native evaluation occurs here.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import time
import types


MIB = 1024 * 1024
AGGREGATE_LIMIT = 32 * MIB
CHECKPOINT_UPDATES = tuple(range(0, 201, 20))
NATIVE_ANCHOR = "26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8"
SPLITS_SHA256 = "fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a"
RESERVATION_SHA256 = "1e9885d4af7b0dd5d5fa762b467f9b15cfb9d6bb2d1637dccec6a195e05063f5"
BASE_OUTPUTS = {"weights.npz", "predictions.npz", "splits.json", "normalization.json",
                "receipt.json", "summary.json", "diagnostics.json"}
EXPECTED_OUTPUTS = BASE_OUTPUTS | {f"checkpoint-{u:03d}.npz" for u in CHECKPOINT_UPDATES}
VARIABLE_OUTPUTS = {"receipt.json", "summary.json", "diagnostics.json"}
HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]


class BudgetError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition, code, detail):
    if not condition:
        raise BudgetError(code, detail)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def regular_bytes(path, maximum=32 * MIB):
    path = Path(path).absolute()
    require(path == path.resolve() and stat.S_ISREG(path.lstat().st_mode),
            "unsafe_path", str(path))
    size = path.stat().st_size
    require(0 < size <= maximum, "file_size", str(path))
    with path.open("rb") as stream:
        blob = stream.read(maximum + 1)
    require(0 < len(blob) <= maximum, "file_size", str(path))
    return blob


def admit_inventory(outputs):
    require(type(outputs) is dict and set(outputs) == EXPECTED_OUTPUTS,
            "inventory", "all eighteen declared flat artifacts, no omissions or extras")
    for name, specification in outputs.items():
        require(type(name) is str and Path(name).name == name and name not in (".", ".."),
                "inventory", "flat output name")
        require(type(specification) in (tuple, list) and len(specification) == 2
                and type(specification[0]) is str and specification[0]
                and type(specification[1]) is int and 0 < specification[1] <= AGGREGATE_LIMIT,
                "inventory_type", name)
    for update in CHECKPOINT_UPDATES:
        require(outputs[f"checkpoint-{update:03d}.npz"] == ("checkpoint", 2 * MIB),
                "checkpoint_cap", "fixed two-MiB numeric checkpoint policy")
    return outputs


def check_sizes(outputs, measured_bytes, projected_bytes):
    """Fail closed on every artifact, then the complete conservative aggregate."""
    admit_inventory(outputs)
    require(type(measured_bytes) is dict and type(projected_bytes) is dict
            and set(measured_bytes) == set(projected_bytes) == set(outputs),
            "inventory", "complete measured and projected inventories")
    for name, (_, maximum) in outputs.items():
        measured, projected = measured_bytes[name], projected_bytes[name]
        require(type(measured) is int and type(projected) is int
                and 0 < measured <= projected <= maximum,
                "artifact_bound", name)
    total = sum(projected_bytes.values())
    require(total <= AGGREGATE_LIMIT, "aggregate_bound", "entire conservative output envelope")
    return total


def load_trainer():
    package_name = "rheon_groupnorm_budget_sources"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(HERE)]
        sys.modules[package_name] = package
    name = package_name + ".train"
    spec = importlib.util.spec_from_file_location(name, HERE / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preservation_snapshot(packet):
    paths = list(sorted(packet.iterdir()))
    require({p.name for p in paths} == {
        "train-0.npz", "train-1.npz", "validation.npz", "splits.json", "origin.json",
        "dataset-receipt.json"}, "packet_inventory", "identical frozen packet2")
    paths += list(sorted((LAB / "experiments/global-pressure-v1/training").glob("*.py")))
    paths += list(sorted((LAB / "experiments/global-pressure-v1/training").glob("*.json")))
    paths += list(sorted((LAB / "docs/plans/global-pressure-diagnosis-v1").rglob("*.json")))
    paths += [LAB / "docs/plans/global-pressure-correction-v1/reservation.json"]
    return {str(path): {"bytes": len(blob), "sha256": digest(blob)}
            for path in paths for blob in [regular_bytes(path)]}


def write_new(path, blob):
    with path.open("xb") as stream:
        stream.write(blob)


def numeric_text_envelope(value):
    """Conservative JSON spelling for the bounded diagnostic numeric fields.

Fixed protocol indices/counts are at most six digits; existing larger immutable
byte counts stay intact. Finite binary64 JSON float spellings need at most24
characters in the pinned Python runtime. Optional cosine None also reserves a
float spelling. These are byte fixtures, never claimed physical observations.
"""
    if type(value) is dict:
        return {key: numeric_text_envelope(item) for key, item in value.items()}
    if type(value) is list:
        return [numeric_text_envelope(item) for item in value]
    if type(value) is float or value is None:
        return -sys.float_info.max
    if type(value) is int:
        return max(value, 999999)
    return value


def measure(packet, work_dir):
    started = time.perf_counter()
    import numpy as np
    import torch

    packet = Path(packet).absolute()
    work_dir = Path(work_dir).absolute()
    require(packet == packet.resolve() and packet.is_dir(), "unsafe_path", "packet2 directory")
    require(work_dir == work_dir.resolve()
            and LAB / "runs/global-pressure-groupnorm-v1" in work_dir.parents,
            "unsafe_path", "task-owned qualification work directory")
    before = preservation_snapshot(packet)
    trainer = load_trainer()
    outputs = admit_inventory(trainer.OUTPUTS)
    require(tuple(trainer.CHECKPOINT_UPDATES) == CHECKPOINT_UPDATES
            and trainer.FIXED_CONFIG["max_output_bytes"] == AGGREGATE_LIMIT,
            "protocol", "fixed checkpoint sequence and aggregate policy")
    sources = trainer.executable_sources()
    model = trainer.model
    reader = model.load_original("data", model.ORIGINAL_DATA_SHA256)
    receipt_raw = regular_bytes(packet / "dataset-receipt.json")
    receipt = trainer.strict_json(receipt_raw)
    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
            and receipt["status"] == "verified_lossless_development_packet"
            and receipt["packager_source_sha256"] == model.ORIGINAL_DATA_SHA256
            and receipt["origin_sha256"] == reader.ORIGIN_SHA256
            and receipt["capture_source_commit"] == reader.SOURCE_COMMIT
            and type(receipt["train_rows"]) is int and receipt["train_rows"] == 288
            and type(receipt["validation_rows"]) is int and receipt["validation_rows"] == 96
            and receipt["native_residual_origin_sha256"] == NATIVE_ANCHOR
            and digest(regular_bytes(packet / "splits.json")) == SPLITS_SHA256,
            "packet_identity", "unchanged native origin and original split")
    require(digest(regular_bytes(LAB / "docs/plans/global-pressure-correction-v1/reservation.json"))
            == RESERVATION_SHA256, "reservation_identity", "untouched final reservation")
    dataset = reader.load_data(*(packet / name for name in ("train-0.npz", "train-1.npz", "validation.npz")),
                               packet / "splits.json", packet / "origin.json",
                               expected_native_residual_origin_sha256=NATIVE_ANCHOR)
    require(len(dataset.train_rows) == 288 and len(dataset.validation_rows) == 96,
            "packet_identity", "all original development rows")
    for filename, recorded in receipt["files"].items():
        actual = before[str(packet / filename)]
        require(actual == recorded, "packet_identity", "frozen packet receipt: " + filename)
    model.configure_cpu()
    require(torch.__version__ == "2.8.0+cpu" and torch.version.cuda is None
            and np.__version__ == "2.3.5", "runtime", "existing pinned CPU numerical runtime")
    torch.manual_seed(20261014)
    network = model.PressureCorrectionNet()
    proof = model.initialization_proof(network)
    normalization = model.fit_normalization(dataset.train, dataset.train_rows)
    normalization_raw = trainer.canonical(normalization)
    require(normalization_raw == trainer.base.canonical(model.original.fit_normalization(
        dataset.train, dataset.train_rows)), "normalization", "original train-only normalization bytes")
    initial_inventory = trainer.state_inventory(network)
    order = trainer.sample_order(dataset.train_rows)
    observation = trainer.observe_batch(network, dataset.train, order[0], normalization)
    population = trainer.observe_population(network, dataset, normalization)
    validation_observations = [
        {"row_index": 288 + index, "row_id": row["row_id"],
         "observations": trainer.observe_batch(network, dataset.validation, [index], normalization)}
        for index, row in enumerate(dataset.validation_rows) if row["frame"] in (0, 23)
    ]
    require(len(population) == 384 and len(validation_observations) == 8,
            "diagnostic_inventory", "complete initialized development diagnostics")
    require(trainer.state_inventory(network) == initial_inventory
            and all(parameter.grad is None for parameter in network.parameters()),
            "no_updates", "autograd observation must not mutate state or parameter gradients")
    cases = trainer.witness_cases(dataset)
    predictions, _ = trainer.make_predictions(network, dataset, normalization, cases)
    trainer.prediction_inventory(predictions)
    require(trainer.state_inventory(network) == initial_inventory,
            "no_updates", "initialized witness forward must preserve weights")
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        for update in CHECKPOINT_UPDATES:
            trainer.save_weights(network, work_dir / f"checkpoint-{update:03d}.npz")
        trainer.save_weights(network, work_dir / "weights.npz")
        with (work_dir / "predictions.npz").open("xb") as stream:
            np.savez(stream, **predictions)
        write_new(work_dir / "splits.json", regular_bytes(packet / "splits.json"))
        write_new(work_dir / "normalization.json", normalization_raw)
        sampled_rows = trainer.sampled_rows(order, dataset.train_rows)
        diagnostic = {
            "schema": trainer.DIAGNOSTICS_SCHEMA,
            "candidate_update": 200,
            "initialization_mapping": proof,
            "checkpoint_hashes": {
                f"checkpoint-{update:03d}.npz": {
                    "bytes": (work_dir / f"checkpoint-{update:03d}.npz").stat().st_size,
                    "sha256": digest(regular_bytes(work_dir / f"checkpoint-{update:03d}.npz")),
                } for update in CHECKPOINT_UPDATES
            },
            "sampled_rows": sampled_rows,
            "logs": [
                {"update": update, "sampled_rows": sampled_rows[update - 1]["rows"],
                 "before_update": observation, "after_update": observation,
                 "checkpoint_sha256": digest(regular_bytes(work_dir / f"checkpoint-{update:03d}.npz"))}
                for update in CHECKPOINT_UPDATES[1:]
            ],
            "population_final": population,
            "validation_gradient_cases": validation_observations,
            "scope": "Qualification-only INITIALIZED observations and synthetic future-log size slots; ZERO updates, not trained diagnostics.",
            "qualification_only": True,
            "projection_update_labels_not_executed": True,
        }
        diagnostic_raw = trainer.canonical(diagnostic)
        diagnostic_numeric_envelope_bytes = len(trainer.canonical(numeric_text_envelope(diagnostic)))
        require(diagnostic_numeric_envelope_bytes <= outputs["diagnostics.json"][1],
                "diagnostic_bound", "full required layout with conservative finite-number spelling")
        write_new(work_dir / "diagnostics.json", diagnostic_raw)
        write_new(work_dir / "summary.json", canonical({
            "scope": "initialized serialization qualification; no scientific Run or trained outputs",
            "optimizer_updates": 0, "parameter_count": proof["parameter_count"],
        }))
        write_new(work_dir / "receipt.json", canonical({
            "status": "qualification_only_not_a_training_receipt", "optimizer_updates": 0,
            "initialized_state_inventory": initial_inventory,
            "initialized_prediction_inventory": trainer.prediction_inventory(predictions),
        }))
        require({p.name for p in work_dir.iterdir()} == set(outputs),
                "inventory", "every representative artifact explicitly declared")
        measured = {name: len(regular_bytes(work_dir / name, spec[1])) for name, spec in outputs.items()}
        projected = {name: spec[1] if name in VARIABLE_OUTPUTS else measured[name]
                     for name, spec in outputs.items()}
        total = check_sizes(outputs, measured, projected)
        after = preservation_snapshot(packet)
        require(after == before and trainer.executable_sources() == sources,
                "preservation", "original packet/source/reservation or candidate source drift")
        require(trainer.state_inventory(network) == initial_inventory,
                "no_updates", "unchanged initialized state after all serialization")
        return {
            "format": "rheon-global-pressure-groupnorm-output-budget-v1",
            "status": "passed_pretraining_conservative_projection",
            "scope": "initialized serialization/forward/autograd qualification only; no trained outputs",
            "optimizer_updates": 0, "optimizer_constructed": False,
            "checkpoint_labels_are_projection_slots": True,
            "candidate_update_if_later_admitted": 200,
            "checkpoint_updates": list(CHECKPOINT_UPDATES), "artifact_count": len(outputs),
            "architecture": model.ARCHITECTURE, "parameter_count": proof["parameter_count"],
            "raw_weight_bytes": proof["raw_weight_bytes"],
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                        "device": "cpu"},
            "executable_sources": sources, "budget_source_sha256": digest(regular_bytes(Path(__file__))),
            "initialization_mapping_sha256": digest(canonical(proof)),
            "initial_state_sha256": digest(canonical(initial_inventory)),
            "sampled_row_ids": [dataset.train_rows[int(i)]["row_id"] for i in order[0]],
            "initialized_diagnostic_inventory": {
                "population_rows": len(population), "validation_gradient_cases": len(validation_observations),
                "actual_train_batch_observations": 1, "synthetic_before_after_log_slots": 20,
                "complete_layout_bytes": len(diagnostic_raw),
                "conservative_numeric_text_bytes": diagnostic_numeric_envelope_bytes,
                "limit_bytes": outputs["diagnostics.json"][1],
                "optimizer_updates": 0,
            },
            "artifacts": {name: {
                "role": spec[0], "limit_bytes": spec[1], "measured_initialized_bytes": measured[name],
                "conservative_projection_bytes": projected[name],
                "measured_initialized_sha256": digest(regular_bytes(work_dir / name, spec[1])),
                "projection_basis": "entire declared JSON cap reserved" if name in VARIABLE_OUTPUTS else (
                    "exact retained preprocessing bytes" if name in ("splits.json", "normalization.json")
                    else "fixed-shape uncompressed numeric serialization, value-independent size"),
            } for name, spec in outputs.items()},
            "aggregate": {"limit_bytes": AGGREGATE_LIMIT,
                          "measured_initialized_bytes": sum(measured.values()),
                          "conservative_projection_bytes": total,
                          "headroom_bytes": AGGREGATE_LIMIT - total,
                          "declared_individual_caps_sum_bytes": sum(spec[1] for spec in outputs.values()),
                          "per_artifact_passed": True, "aggregate_passed": True},
            "preservation": {"before_after_equal": True, "files": before,
                             "native_residual_origin_sha256": NATIVE_ANCHOR,
                             "splits_sha256": SPLITS_SHA256, "reservation_sha256": RESERVATION_SHA256,
                             "train_rows": 288, "validation_rows": 96,
                             "normalization_sha256": digest(normalization_raw)},
            "elapsed_s": time.perf_counter() - started,
            "observed_peak_rss_bytes": trainer.observed_peak_rss_bytes(),
            "limitations": [
                "Checkpoint labels 020..200 contain initialized-state serialization samples, not trained states.",
                "Only final200 may become a candidate after separately authorized completed training.",
                "Future trainer must enforce every file cap and actual aggregate; unused declared room is not hidden output permission.",
                "JSON/history sizes use the full caps, not invented trained values or projected optimizer success.",
            ],
        }
    except BaseException:
        shutil.rmtree(work_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = measure(args.packet, args.work_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_new(args.report, json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        print(json.dumps({"status": report["status"], "report": str(args.report),
                          "aggregate": report["aggregate"]}, sort_keys=True))
    except (BudgetError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "code": getattr(error, "code", "qualification"),
                          "detail": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
