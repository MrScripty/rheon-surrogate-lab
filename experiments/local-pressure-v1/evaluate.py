"""Fresh teacher-state native evaluation with retained physics and fallback costs."""
import os
from pathlib import Path
import time

import numpy as np

from contract import LAB, MIB, canonical, decode, fresh, load_module, read, require, sha, write
import capture
import data
import model_profile
import native

METHODS = ("learned_jacobi", "temporal_full_budget", "temporal_jacobi", "zero_jacobi")


def completed(attempt):
    error = attempt["solver_error"]
    if error is not None and not error.startswith(("pressure iteration limit ", "pressure solve cancelled after ")):
        return None
    return attempt["iterations"]


def summary(records):
    result = {}
    for method in METHODS:
        attempts, wasted = [], []
        for record in records:
            value = record[method]
            if method in ("learned_jacobi", "temporal_jacobi"):
                attempts += [a for a in (value["warm"], value["reset"]) if a is not None]
                if value["reset"] is not None and value["warm"] is not None:
                    wasted.append(value["warm"])
            else:
                attempts.append(value)
        counts = [completed(a) for a in attempts]
        wasted_counts = [completed(a) for a in wasted]
        result[method] = {"qualified": sum(r[method]["qualified"] for r in records), "completed_iterations_known": sum(n for n in counts if n is not None), "unknown_completed_iteration_attempts": sum(n is None for n in counts), "wasted_warm_iterations_known": sum(n for n in wasted_counts if n is not None), "unknown_wasted_warm_attempts": sum(n is None for n in wasted_counts), "fallbacks": sum(r[method].get("reset_fallback", False) for r in records)}
    return result


def run(capture_root, packet, binary, training, restored, output):
    start = time.monotonic()
    verified = capture.verify(capture_root)
    require(verified["status"] == "verified_development", "evaluation_population", "complete development only")
    dataset = data.load(packet)
    training_receipt = decode(read(training / "receipt.json", MIB))
    restore_receipt = decode(read(restored / "receipt.json", MIB))
    require(training_receipt["schema"] == restore_receipt["schema"] == "rheon-local-pressure-training-receipt-v1" and training_receipt["status"] == restore_receipt["status"] == "complete" and training_receipt["mode"] == "train" and restore_receipt["mode"] == "restore" and restore_receipt["optimizer_updates"] == 0 and training_receipt["optimizer_updates"] == dataset.config["training"]["updates"] and training_receipt["closure"] == restore_receipt["closure"] and restore_receipt["originating_training_run"] is not None, "trained_restore", "independent managed restore of selected final candidate")
    require(training_receipt["closure"]["experiment_sha256"] == sha(canonical(dataset.config)) and dataset.splits["capture_manifest_sha256"] == verified["manifest_sha256"], "evaluation_lineage", "same fresh capture/packet/experiment")
    selected = {}
    for name, declaration in restore_receipt["outputs"].items():
        raw = read(restored / name, 2 * MIB)
        require(declaration["sha256"] == sha(raw) and declaration["bytes"] == len(raw), "restore_artifact", name)
        if name != "summary.json":
            require(raw == read(training / name, 2 * MIB), "restore_parity", name)
        selected[str(restored / name)] = sha(raw)
    _model, helpers = model_profile.load()
    arrays = helpers.read_npz(read(restored / "predictions.npz", 2 * MIB), restore_receipt["prediction_inventory"], 2 * MIB)
    require(training_receipt["cases"] == restore_receipt["cases"] and len(restore_receipt["cases"]) == 2 * len(dataset.config["trajectories"]), "case_population", "frames zero/final for every whole parent")
    receipt = native.admit(binary)
    require(receipt == dataset.splits["native_residual_origin"]["native_build_receipt"], "residual_operator", "same qualified full residual/native operator")
    policy = load_module("local_pressure_native_policy", LAB / "experiments/global-pressure-v1/groupnorm-native-dev-v1/admission.py")
    require(np.array_equal(arrays["row_indices"], [c["row_index"] for c in restore_receipt["cases"]]), "prediction_mapping", "complete fixed case mapping")
    output = fresh(output)
    write(output / "population.json", restore_receipt["cases"])
    records = []
    try:
        for index, case in enumerate(restore_receipt["cases"]):
            require(time.monotonic() - start < 590, "evaluation_deadline", "bounded local diagnostic")
            row_index = case["row_index"]
            arrays_source = dataset.train if row_index < len(dataset.train_rows) else dataset.validation
            local = row_index if row_index < len(dataset.train_rows) else row_index - len(dataset.train_rows)
            trajectory = next(t for t in dataset.origin["trajectories"] if t["trajectory_id"] == case["trajectory_id"])
            frame = trajectory["frames"][case["frame"]]
            velocity = [capture.oracle.load_array(capture_root, frame["arrays"]["pre_" + axis], capture.oracle.shape(d), "<f4", "m/s") for d, axis in enumerate(("u", "v", "w"))]
            base = arrays_source["base"][local].reshape(-1)
            residual = arrays_source["native_input_residual"][local].reshape(-1)
            candidate = arrays["pressure"][index].reshape(-1)
            zero = policy.zero_disposition(candidate, base, residual)
            l2, maximum = policy.residual_metrics(residual)
            request = {"schema": "rheon-global-pressure-teacher-case-v1", "counts_xyz": [16] * 3, "spacing_m": arrays_source["spacing"][local].tolist(), "density_kg_m3": float(arrays_source["rho"][local]), "dt_s": float(arrays_source["dt"][local]), "original_rhs": arrays_source["rhs"][local].reshape(-1).tolist(), "reference_pressure": arrays_source["reference"][local].reshape(-1).tolist(), "previous_pressure": base.tolist(), "pre_velocity": velocity, "candidate_pressure": candidate.tolist(), "candidate_error": None}
            response, measurement = native.call(binary, "--evaluate", request, min(60, 590 - (time.monotonic() - start)))
            record = {**case, **response, "native_base_residual_l2": l2, "native_base_residual_max": maximum, "zero_residual_base_retained": zero, "process": measurement, "earlier_restore_inference_s": restore_receipt["per_case_inference_s"][index]}
            with (output / "cases.jsonl").open("ab") as journal:
                journal.write(canonical(record) + b"\n")
                journal.flush()
                os.fsync(journal.fileno())
            policy.validate_result(request, response, record)
            require(all(response[method]["qualified"] for method in METHODS), "native_terminal_failure", case["case_id"])
            records.append(record)
        require(native.admit(binary) == receipt, "native_drift", "post-evaluation identity")
        for path, expected in selected.items():
            require(sha(read(Path(path), 2 * MIB)) == expected, "artifact_drift", path)
        require(sha(read(capture_root / "manifest.json", 4 * MIB)) == verified["manifest_sha256"], "capture_drift", "post-evaluation capture")
        report = {"schema": "rheon-local-pressure-native-evaluation-v1", "status": "complete_diagnostic", "cases": len(records), "raw_qualified": sum(r["raw_learned"]["qualified"] for r in records), "methods": summary(records), "native_elapsed_sum_s": sum(r["native_elapsed_s"] for r in records), "workflow_elapsed_s": time.monotonic() - start, "restore_inference_sum_s": sum(restore_receipt["per_case_inference_s"]), "native_build_receipt_sha256": sha(read(binary.parent / "build-receipt.json", 2 * MIB)), "capture_manifest_sha256": verified["manifest_sha256"], "training_receipt_sha256": sha(read(training / "receipt.json", MIB)), "restore_receipt_sha256": sha(read(restored / "receipt.json", MIB)), "scope": "teacher-state development snapshots; separate historical/new lineages; no autonomous rollout, final generalization, GPU or acceleration claim", "cost_scope": "native methods include failed warm work and fallback; enclosing evaluation includes verification and process lifetime; earlier managed train/restore and inference clocks are separate overlapping scopes, not an end-to-end speedup"}
        write(output / "evaluation.json", report)
        return report
    except BaseException as error:
        write(output / "failure.json", {"status": "failed", "completed_cases": len(records), "error": str(error), "elapsed_s": time.monotonic() - start})
        raise
