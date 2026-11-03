"""Current pressure evidence glue over the captured finite-update water trainer.

Import is inert. Execution belongs exclusively to the managed recipe/Run owner.
CUDA training and ordinary CPU restore keep the original numerical implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


def require(condition, message):
    if not condition:
        raise ValueError("Pressure evidence: " + message)


def mean_free_relative_l2(prediction, reference):
    """Same-request paired pressure metric, with an explicit undefined boundary."""
    import numpy as np
    require(prediction.shape == reference.shape and prediction.dtype.str == reference.dtype.str == "<f8",
            "exact float64 paired pressure required")
    require(np.isfinite(prediction).all() and np.isfinite(reference).all(), "nonfinite paired pressure")
    error = prediction - reference
    error = error - error.mean()
    centered = reference - reference.mean()
    denominator = float(np.linalg.norm(centered.reshape(-1)))
    numerator = float(np.linalg.norm(error.reshape(-1)))
    require(denominator > 0 or numerator == 0, "relative pressure error undefined for constant reference")
    return numerator / denominator if denominator else 0.0


def restore_artifact_names(declared):
    """Only the SDK/profile collision changes a published filename, never bytes."""
    return {slot: ("training-profile.json" if name == "profile.json" else name, role, maximum)
            for slot, (name, role, maximum) in declared.items()}


def trainer_input_receipts(evidence, runtime_path, witness_path):
    """Keep the manifest's current Project custody while omitting it from fit slots."""
    result = dict(evidence)
    receipt = result.pop("runtime_manifest")
    require(type(receipt) is dict and set(receipt) == {
        "project_id", "manifest_sha256", "reference_id", "path", "measured_sha256"}
        and receipt["reference_id"] == "runtime_manifest", "current runtime manifest Project receipt required")
    actual = hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest()
    witness = json.loads(Path(witness_path).read_bytes())
    require(receipt["measured_sha256"] == actual == witness["runtime_manifest_sha256"],
            "supplemental runtime manifest differs from exact captured witness")
    require((receipt["project_id"], receipt["manifest_sha256"]) ==
            (result["dataset"]["project_id"], result["dataset"]["manifest_sha256"]),
            "runtime and fitting inputs belong to different snapshots")
    return result


def emit_evidence(output, context_raw, probe_path):
    """Evaluate the actual final weights on one already-consumed validation frame."""
    import numpy as np
    import train
    from tuldok_extension_sdk.evidence import Evidence

    started = time.monotonic()
    output = Path(output)
    training = json.loads((output / "receipt.json").read_bytes())
    require(training["mode"] == "train" and 1 <= training["optimizer_updates"] <= 5000
            and training["status"] == "complete", "completed bounded CUDA training required")
    updates = training["optimizer_updates"]
    checkpoint = f"checkpoint-{updates:03d}.npz"
    history = json.loads((output / "history.json").read_bytes())
    normalization = json.loads((output / "normalization.json").read_bytes())
    # Preserve the original training profile and receipt exactly. The SDK owns
    # profile.json in this prospective typed recipe's declared output inventory.
    (output / "profile.json").rename(output / "training-profile.json")
    writer = Evidence(output, context_raw)
    require(writer.static["profile"]["capture"] == {
        "clock": "optimizer_update", "interval": updates, "retain_selected": True, "max_snapshots": 1},
        "only the finite final update is indexed")
    members = writer.static["probe"]["members"]
    require(len(members) == 1 and members[0]["frozen_reference"]["split"] == "validation",
            "one predeclared consumed validation frame required")
    policy = writer.policy
    specification = policy["model_specification"]
    context = specification["physical_context"]
    require(context["density_kg_m3"] == 1000 and context["dt_s"] == 1 / 60
            and context["viscosity_pa_s"] == 0.001 and context["origin_xyz_m"] == [0, 0, 0],
            "frozen pressure physical context required")
    require(hashlib.sha256(Path(probe_path).read_bytes()).hexdigest() == writer.static["probe"]["data_sha256"],
            "admitted consumed validation probe bytes changed")
    with np.load(probe_path, allow_pickle=False) as probe:
        require(set(probe.files) == {"initial_residual", "previous_pressure", "native_pressure"},
                "exact residual/base/native probe arrays required")
        arrays = {name: probe[name].copy() for name in probe.files}
    field = policy["fields"][0]
    shape = field["contract"]["output"]["shape_zyx"]
    require(all(value.dtype.str == "<f8" and list(value.shape) == shape and np.isfinite(value).all()
                for value in arrays.values()), "unchanged finite float64 ZYX probe fields required")
    geometry = field["geometry"]
    spacing = geometry["spacing_xyz"]
    require(spacing[0] == spacing[1] == spacing[2] and spacing[0] > 0,
            "one positive cubic cell size required")
    network = train.model.PressureCorrectionNet().cpu()
    train.data.helpers().load_weights(network, output / checkpoint,
                                     history["checkpoints"][checkpoint]["inventory"])
    network.eval()
    sample = {"base": train.torch.from_numpy(arrays["previous_pressure"]).reshape(1, 1, *shape),
              "native_input_residual": train.torch.from_numpy(arrays["initial_residual"]).reshape(1, 1, *shape),
              "rho": train.torch.tensor([1000.0], dtype=train.torch.float64),
              "dt": train.torch.tensor([1 / 60], dtype=train.torch.float64),
              "cell_size_m": train.torch.tensor([spacing[0]], dtype=train.torch.float64)}
    with train.torch.no_grad():
        predicted = train.model.predict_pressure(network, sample, normalization).reshape(*shape).numpy()
    require(predicted.dtype.str == "<f8" and np.isfinite(predicted).all(), "finite reconstructed pressure required")
    require(arrays["previous_pressure"].flat[0] == 0 and arrays["native_pressure"].flat[0] == 0
            and predicted.flat[0] == 0, "declared cell-zero pressure gauge differs")
    member_id = members[0]["frozen_reference"]["member_id"]
    for row in policy["probe_inputs"]:
        require(row["member_id"] == member_id, "admitted probe input member differs")
        for value in row["fields"]:
            with (output / value["artifact"]).open("xb") as stream:
                np.save(stream, arrays[value["field_id"]], allow_pickle=False)
    bindings = policy["field_outputs"]
    require(len(bindings) == 1 and bindings[0]["clock"] == updates and bindings[0]["member_id"] == member_id,
            "one admitted full-pressure field pair required")
    for name, values in ((bindings[0]["prediction"], predicted),
                         (bindings[0]["reference"], arrays["native_pressure"])):
        with (output / name).open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
    value = mean_free_relative_l2(predicted, arrays["native_pressure"])
    require(len(policy["metric_definitions"]) == 1, "one exact pressure diagnostic metric required")
    writer.transform("normalization.json")
    writer.checkpoint(updates, {"optimizer_update": updates, "epoch": None, "fit_pass": None, "physical_time": None,
                         "wall_elapsed_ms": int(training["elapsed_s"] * 1000 + (time.monotonic() - started) * 1000)},
                      checkpoint, {policy["metric_definitions"][0]["id"]: value})
    writer.finish()
    writer.write("pressure-diagnostic.json", {"format": "rheon_pressure_diagnostic_v1",
        "population": "already_consumed_validation", "optimizer_updates_executed": updates,
        "mean_free_same_request_pressure_relative_l2": value, "trajectory_metric": None,
        "scientific_acceptance": "not_asserted", "model_promotion": False,
        "training_profile_artifact": "training-profile.json", "original_training_receipt": "receipt.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--pressure-probe", type=Path)
    parser.add_argument("--model-evidence-context", type=Path)
    args, training_arguments = parser.parse_known_args()
    require(training_arguments and training_arguments[0] in ("train", "restore"), "explicit train/restore mode required")
    import train
    import training_diagnostic
    output_index = training_arguments.index("--output-dir") + 1
    output = Path(training_arguments[output_index])
    evidence_index = training_arguments.index("--input-evidence") + 1
    evidence = json.loads(Path(training_arguments[evidence_index]).read_bytes())
    evidence = trainer_input_receipts(evidence,
        training_arguments[training_arguments.index("--runtime-manifest") + 1],
        training_arguments[training_arguments.index("--source-witness") + 1])
    if training_arguments[0] == "train":
        require(args.pressure_probe is not None and args.model_evidence_context is not None,
                "bound training requires captured validation probe and exact host context")
        train.evaluate = training_diagnostic.diagnostic
        train.write = training_diagnostic.write
        config = json.loads(Path(training_arguments[training_arguments.index("--config") + 1]).read_bytes())
        slots = set(train.input_slots(config))
        require(set(evidence) == slots | {"pressure_probe"}, "only the declared probe may supplement training inputs")
        evidence = {slot: evidence[slot] for slot in slots}
    else:
        require(args.pressure_probe is None and args.model_evidence_context is None,
                "ordinary CPU restore does not claim the bound field profile")
        original_custody = train.custody
        def custody(evidence, trained, declared, validator_path, validator_sha):
            return original_custody(evidence, trained, restore_artifact_names(declared), validator_path, validator_sha)
        train.custody = custody
    with tempfile.TemporaryDirectory(prefix="pressure-trainer-input-") as temporary:
        selected = Path(temporary) / "input-evidence.json"
        selected.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False))
        training_arguments[evidence_index] = str(selected)
        sys.argv[:] = [sys.argv[0], *training_arguments]
        train.main()
    if training_arguments[0] == "train":
        emit_evidence(output, args.model_evidence_context.read_bytes(), args.pressure_probe)


if __name__ == "__main__":
    main()
