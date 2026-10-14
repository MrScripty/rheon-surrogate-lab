"""Future fixed GroupNorm experiment and exact history-preserving restore.

Execution requires a new admitted source/config/runtime/material closure. This
module's pretraining qualification never creates an optimizer or invokes run().
Numeric intermediate states are observations; only final200 is a candidate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from . import model
else:
    import model

base = model.load_original("train", model.ORIGINAL_TRAIN_SHA256)
FIXED_CONFIG = copy.deepcopy(base.FIXED_CONFIG)
FIXED_CONFIG.update(schema="rheon-global-pressure-groupnorm-training-config-v1",
                    model_format=model.ARCHITECTURE["schema"],
                    checkpoint_updates=list(range(0, 201, 20)),
                    candidate_update=200,
                    block_normalization=model.ARCHITECTURE["block_normalization"])
SOURCE_SCHEMA = "rheon-global-pressure-groupnorm-source-v1"
RECEIPT_SCHEMA = "rheon-global-pressure-groupnorm-training-receipt-v1"
DIAGNOSTICS_SCHEMA = "rheon-global-pressure-groupnorm-diagnostics-v1"
CHECKPOINT_UPDATES = tuple(range(0, 201, 20))
OUTPUTS = {**base.OUTPUTS,
           **{f"checkpoint-{update:03d}.npz": ("checkpoint", 2 * 1024**2) for update in CHECKPOINT_UPDATES},
           "diagnostics.json": ("training_diagnostics", 2 * 1024**2)}
TRAINED_INPUTS = {**base.TRAINED_INPUTS,
                  **{f"trained_checkpoint_{update:03d}": (f"checkpoint-{update:03d}.npz", "checkpoint") for update in CHECKPOINT_UPDATES},
                  "trained_diagnostics": ("diagnostics.json", "training_diagnostics")}
# Adapt only the private frozen helper's artifact-role inventory at composition
# root; its accepted cold custody validator/algorithm and source remain unchanged.
base.TRAINED_INPUTS = TRAINED_INPUTS
NATIVE_BASE = base.NATIVE_BASE
TULDOK_REVISION = base.TULDOK_REVISION
CUSTODY_VALIDATOR_SHA256 = base.CUSTODY_VALIDATOR_SHA256
HASH = base.HASH
TrainingContractError = base.TrainingContractError
require, keys, strict_json = base.require, base.keys, base.strict_json
canonical, sha256, regular_bytes, identity = base.canonical, base.sha256, base.regular_bytes, base.identity
tensors, prediction, losses, evaluate = base.tensors, base.prediction, base.losses, base.evaluate
sample_order, state_inventory, save_weights, load_weights, read_npz = base.sample_order, base.state_inventory, base.save_weights, base.load_weights, base.read_npz
witness_cases, make_predictions, prediction_inventory = base.witness_cases, base.make_predictions, base.prediction_inventory
copy_new, write_json_new, progress = base.copy_new, base.write_json_new, base.progress
observed_peak_rss_bytes, output_directory = base.observed_peak_rss_bytes, base.output_directory
validate_trained_custody = base.validate_trained_custody

SOURCE_NAMES = {"model.py", "train.py", "original_data.py", "original_model.py", "original_train.py"}
PINNED_DEPENDENCIES = {"original_model.py": model.ORIGINAL_MODEL_SHA256,
                       "original_train.py": model.ORIGINAL_TRAIN_SHA256,
                       "original_data.py": model.ORIGINAL_DATA_SHA256}
PACKET_SHA256 = {"train0": "6b3d0c4a57df0198b4fe13f7e887b8fa238fbf3fd34877f9dc598c157d2b55dc",
                 "train1": "d6d7980a51c17b79a6567c870aca9001108779798cee490101429e00814a9f02",
                 "validation": "17a9faa84723208d1bd9d6a3a8b1f5921806e93b85d8983d2041404fb295855d",
                 "splits": "fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a",
                 "origin": "dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d"}
NORMALIZATION_SHA256 = "8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f"
QUALIFICATION_GATES = ["initialization_mapping", "full_stencil", "gauge_loss_gradient", "model_parameter_gradient",
                       "global_dependence", "shapes_16_32", "group_statistics", "batch_neighbor_invariance",
                       "train_eval_invariance", "zero_residual_bypass", "near_constant_large_amplitude",
                       "forcing_amplitude_sensitivity", "numeric_checkpoint_roundtrip", "strict_restore_identity", "no_optimizer_steps", "output_budget"]
QUALIFICATION_METRICS = {"initialization_tensor_count", "groupnorm_module_count", "group_mean_max_abs", "group_variance_max",
                         "batch_neighbor_max_abs_error", "train_eval_max_abs_error", "zero_residual_max_abs_error",
                         "gauge_finite_difference_max_abs_error", "groupnorm_finite_difference_max_abs_error",
                         "model_parameter_gradient_abs", "model_parameter_fd_abs_error", "distant_input_derivative_abs",
                         "global_pool_gradient_l1", "native_operator_max_abs_error", "amplitude_response_relative_error",
                         "amplitude_derivative_abs", "dry_forward_s_16", "dry_forward_s_32", "dry_autograd_s",
                         "observed_peak_rss_bytes"}


def source_bytes(name):
    local = Path(__file__).with_name(name)
    try:
        local.lstat()
    except FileNotFoundError:
        pass
    else:
        return regular_bytes(local, 1024**2)
    original_name = name.removeprefix("original_")
    require(name in PINNED_DEPENDENCIES, "source_missing", name)
    return regular_bytes(Path(__file__).parent.parent / "training" / original_name, 1024**2)


def executable_sources():
    result = {name: sha256(source_bytes(name)) for name in sorted(SOURCE_NAMES)}
    require(all(result[name] == expected for name, expected in PINNED_DEPENDENCIES.items()), "original_source_hash", "immutable original dependencies")
    return result


def loss_components(estimated, batch, normalization):
    scale = model._scale_values(normalization)
    count = estimated[0].numel()
    dp = batch["reference"].flatten(1).square().sum(1).clamp_min((1e-6 * math.sqrt(count) * scale["pressure"])**2)
    db = batch["rhs"].flatten(1).square().sum(1).clamp_min((1e-6 * math.sqrt(count) * scale["rhs"])**2)
    pressure = .5 * (estimated - batch["reference"]).flatten(1).square().sum(1) / dp
    residual = .5 * (model.full_operator(estimated, batch["spacing"], batch["rho"]) - batch["rhs"]).flatten(1).square().sum(1) / db
    require(torch.equal(pressure + residual, model.pressure_loss(estimated, batch["reference"], batch["rhs"], batch["spacing"], batch["rho"], normalization)), "loss_components", "accepted physical loss")
    return pressure, residual


def magnitude(value):
    v = value.detach().double()
    require(bool(torch.isfinite(v).all()), "diagnostic_nonfinite", "activation/gradient")
    flat = v.flatten(1) if v.ndim > 1 else v.reshape(1, -1)
    return {"min": float(v.min()), "max": float(v.max()), "rms": float(v.square().mean().sqrt()),
            "zero_fraction": float((v == 0).double().mean()),
            "spatial_range_max": float((flat.max(1).values - flat.min(1).values).max())}


def gradient_summary(values):
    v = torch.cat([value.detach().double().flatten() for value in values])
    require(bool(torch.isfinite(v).all()), "diagnostic_nonfinite", "component gradients")
    return {"l2": float(v.norm()), "max_abs": float(v.abs().max()), "nonzero_count": int((v != 0).sum())}


class ActivationHooks:
    def __init__(self, network, keep_graph=False):
        self.values = {}
        self.handles = []
        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv3d, nn.GroupNorm, nn.SiLU, nn.AdaptiveAvgPool3d)):
                self.handles.append(module.register_forward_pre_hook(self.hook(name + ":input", keep_graph)))
                self.handles.append(module.register_forward_hook(self.hook(name + ":output", keep_graph, output=True)))

    def hook(self, name, keep_graph, output=False):
        def callback(_module, args, value=None):
            v = value if output else args[0]
            self.values[name] = v if keep_graph else v.detach()
        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()


def observe_batch(network, arrays, indices, normalization):
    """Read-only before/after observation; autograd.grad never writes .grad."""
    hooks = ActivationHooks(network, keep_graph=True)
    try:
        batch = tensors(arrays, indices)
        x = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization).requires_grad_()
        active = (batch["native_input_residual"].flatten(1) != 0).any(1)
        require(bool(active.all()), "diagnostic_bypass", "log population must have nonzero native residual")
        raw = network(x)
        correction = model.gauge_correction(raw, model._scale_values(normalization)["pressure"])
        estimated = batch["base"] + correction
        lp, lb = loss_components(estimated, batch, normalization)
        parameters = list(network.parameters())
        gp = torch.autograd.grad(lp.mean(), parameters, retain_graph=True)
        gb = torch.autograd.grad(lb.mean(), parameters, retain_graph=True)
        globals_value = hooks.values["global_pool:output"]
        raw_gradient, input_gradient, global_gradient = torch.autograd.grad((lp + lb).mean(), [raw, x, globals_value])
        p = torch.cat([g.detach().double().flatten() for g in gp])
        b = torch.cat([g.detach().double().flatten() for g in gb])
        pnorm, bnorm = float(p.norm()), float(b.norm())
        return {"pressure_loss": float(lp.mean().detach()), "residual_loss": float(lb.mean().detach()),
                "total_loss": float((lp + lb).mean().detach()), "raw_head": magnitude(raw), "correction": magnitude(correction),
                "component_gradients": {"pressure": gradient_summary(gp), "residual": gradient_summary(gb),
                                        "cosine": float((p * b).sum()) / (pnorm * bnorm) if pnorm and bnorm else None},
                "raw_output_gradient": gradient_summary([raw_gradient]), "global_gradient": gradient_summary([global_gradient]),
                "input_gradients": {name: gradient_summary([input_gradient[:, index]]) for index, name in enumerate(model.CHANNELS)},
                "activations": {name: magnitude(v.flatten(0, 1) if v.ndim == 5 else v) for name, v in hooks.values.items()}}
    finally:
        hooks.close()


def observe_population(network, dataset, normalization):
    """Final384 existing development teacher states, no checkpoint selection."""
    result = []
    for split, arrays, rows in (("train", dataset.train, dataset.train_rows), ("validation", dataset.validation, dataset.validation_rows)):
        hooks = ActivationHooks(network, keep_graph=True)
        try:
            for first in range(0, len(rows), 2):
                batch = tensors(arrays, [first, first + 1])
                x = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization).requires_grad_()
                raw = network(x)
                correction = model.gauge_correction(raw, model._scale_values(normalization)["pressure"])
                lp, lb = loss_components(batch["base"] + correction, batch, normalization)
                # Sum independent per-case losses, so each selected gradient is
                # that row's derivative with no batch-mean scaling factor.
                gr, gx, gg = torch.autograd.grad((lp + lb).sum(), [raw, x, hooks.values["global_pool:output"]])
                for offset in range(2):
                    result.append({"row_index": len(result), "row_id": rows[first + offset]["row_id"], "split": split,
                                   "pressure_loss": float(lp[offset].detach()), "residual_loss": float(lb[offset].detach()),
                                   "raw_head": magnitude(raw[offset]), "correction": magnitude(correction[offset]),
                                   "last_decoder_pre_silu": magnitude(hooks.values["decoders.3.layers.2:input"][offset]),
                                   "last_decoder_pre_groupnorm": magnitude(hooks.values["decoders.3.layers.1:input"][offset]),
                                   "head_features": magnitude(hooks.values["head:input"][offset]),
                                   "raw_output_gradient": gradient_summary([gr[offset]]),
                                   "global_gradient": gradient_summary([gg[offset]]),
                                   "input_gradients": {name: gradient_summary([gx[offset, channel]]) for channel, name in enumerate(model.CHANNELS)}})
                hooks.values.clear()
            observed_peak_rss_bytes()
        finally:
            hooks.close()
    require(len(result) == 384, "diagnostic_population", "all development rows")
    return result


def validate_qualification(report, source):
    keys(report, {"format", "status", "executable_sources", "runtime", "parameter_count", "raw_weight_bytes", "tests", "evidence"}, "GroupNorm qualification")
    require(report["format"] == "rheon-global-pressure-groupnorm-qualification-v1" and report["status"] == "passed", "qualification_status", "passed independent pretraining qualification")
    require(canonical(report["executable_sources"]) == canonical(source["executable_sources"]), "qualification_source", "exact new source closure")
    require(canonical(report["runtime"]) == canonical({"torch": "2.8.0+cpu", "numpy": "2.3.5", "threads": 1, "interop_threads": 1}), "qualification_runtime", "qualified CPU")
    require(type(report["parameter_count"]) is type(report["raw_weight_bytes"]) is int and report["parameter_count"] == 372377 and report["raw_weight_bytes"] == 1489508, "qualification_model", "fixed inventory")
    keys(report["tests"], {"suite_sha256", "passed_count"}, "qualification suite")
    require(type(report["tests"]["suite_sha256"]) is str and HASH.fullmatch(report["tests"]["suite_sha256"]) and type(report["tests"]["passed_count"]) is int and report["tests"]["passed_count"] > 0, "qualification_tests", "passing source-bound suite")
    keys(report["evidence"], {"gates", "metrics", "initialization_mapping_sha256", "no_optimizer_constructed", "native_operator", "output_budget_sha256"}, "qualification evidence")
    require(canonical(report["evidence"]["gates"]) == canonical(QUALIFICATION_GATES) and report["evidence"]["no_optimizer_constructed"] is True, "qualification_gates", "all no-training gates")
    require(type(report["evidence"]["initialization_mapping_sha256"]) is str and HASH.fullmatch(report["evidence"]["initialization_mapping_sha256"]), "qualification_mapping", "exact original parameter mapping")
    native = report["evidence"]["native_operator"]
    keys(native, set(source["native_qualification"]) | {"oracle_action_sha256"}, "qualification native operator")
    require(all(native[name] == value for name, value in source["native_qualification"].items()) and type(native["oracle_action_sha256"]) is str and HASH.fullmatch(native["oracle_action_sha256"]), "qualification_native", "source-bound inherited native oracle")
    require(type(report["evidence"]["output_budget_sha256"]) is str and HASH.fullmatch(report["evidence"]["output_budget_sha256"]), "qualification_budget", "qualified new output budget")
    metrics = report["evidence"]["metrics"]
    keys(metrics, QUALIFICATION_METRICS, "qualification metric inventory")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in metrics.values()), "qualification_metrics", "finite arithmetic evidence")
    require(type(metrics["initialization_tensor_count"]) is type(metrics["groupnorm_module_count"]) is int and metrics["initialization_tensor_count"] == 38 and metrics["groupnorm_module_count"] == 18, "qualification_metrics", "complete model inventory")
    require(metrics["group_mean_max_abs"] <= 2e-6 and metrics["group_variance_max"] <= 1.0001 and metrics["batch_neighbor_max_abs_error"] == metrics["train_eval_max_abs_error"] == metrics["zero_residual_max_abs_error"] == 0, "qualification_metrics", "fixed GroupNorm/invariance arithmetic gates")
    require(metrics["gauge_finite_difference_max_abs_error"] <= 1e-6 and metrics["groupnorm_finite_difference_max_abs_error"] <= 1e-6 and metrics["model_parameter_gradient_abs"] > 1e-6 and metrics["model_parameter_fd_abs_error"] <= max(1e-4, .05 * metrics["model_parameter_gradient_abs"]), "qualification_metrics", "meaningful finite differences")
    require(metrics["distant_input_derivative_abs"] > 0 and metrics["global_pool_gradient_l1"] > 0 and metrics["amplitude_derivative_abs"] > 0 and metrics["native_operator_max_abs_error"] <= 1e-12, "qualification_metrics", "actual global/forcing/native sensitivity")
    require(type(metrics["observed_peak_rss_bytes"]) is int and 0 < metrics["observed_peak_rss_bytes"] <= 4 * 1024**3, "qualification_metrics", "observed CPU memory budget")


def validate_source(args, raw_inputs, config_raw, runtime_raw, evidence_raw):
    source_raw = regular_bytes(args.source_witness, 256 * 1024)
    source = strict_json(source_raw)
    keys(source, {"format", "source_commit", "tuldok_revision", "fixedmodel_format", "runtime_snapshot_sha256", "executable_sources", "config_sha256", "package_inputs", "native_qualification", "qualification_report_sha256", "native_residual_origin_sha256"}, "new source witness")
    require(source["format"] == SOURCE_SCHEMA and source["fixedmodel_format"] == model.ARCHITECTURE["schema"], "source_format", "new GroupNorm identity")
    require(type(source["source_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", source["source_commit"]), "source_commit", "committed new closure")
    require(source["tuldok_revision"] == TULDOK_REVISION, "tuldok_revision", "accepted generic owner")
    require(source["native_residual_origin_sha256"] == "26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8", "residual_anchor", "unchanged packet2 native anchor")
    require(canonical(source["executable_sources"]) == canonical(executable_sources()), "source_hash", "new source and pinned dependency closure")
    require(source["config_sha256"] == sha256(config_raw) and canonical(strict_json(config_raw)) == canonical(FIXED_CONFIG), "fixed_config", "fixed new scientific config")
    require(source["runtime_snapshot_sha256"] == sha256(runtime_raw), "runtime_snapshot_hash", "qualified captured runtime")
    keys(source["package_inputs"], raw_inputs, "package inputs")
    for name, raw in raw_inputs.items():
        require(canonical(source["package_inputs"][name]) == canonical(identity(raw)), "input_hash", name)
        require(sha256(raw) == PACKET_SHA256[name], "original_packet_hash", "unchanged exact packet2 " + name)
    keys(source["native_qualification"], {"base_commit", "patch_sha256", "prepared_tree_sha256", "operator_binary_sha256"}, "native identity")
    require(source["native_qualification"]["base_commit"] == NATIVE_BASE and all(type(v) is str and HASH.fullmatch(v) for n, v in source["native_qualification"].items() if n != "base_commit"), "native_identity", "unchanged qualified operator")
    raw_qualification = regular_bytes(args.source_witness.with_name("model-qualification.json"), 1024**2)
    require(sha256(raw_qualification) == source["qualification_report_sha256"], "qualification_hash", "new pretraining report")
    qualification = strict_json(raw_qualification)
    validate_qualification(qualification, source)
    budget = regular_bytes(args.source_witness.with_name("output-budget.json"), 1024**2)
    require(sha256(budget) == qualification["evidence"]["output_budget_sha256"], "qualification_budget_hash", "exact independently projected output bytes")
    runtime = strict_json(runtime_raw)
    keys(runtime, {"schema_version", "profile", "python_version", "packages", "launch", "files", "total_bytes"}, "runtime manifest")
    require(type(runtime["schema_version"]) is int and runtime["schema_version"] == 1 and runtime["profile"] == "captured_cpu_python_v1", "runtime_profile", "captured CPU")
    require(type(runtime["packages"]) is dict and runtime["packages"].get("torch") == "2.8.0+cpu" and runtime["packages"].get("numpy") == "2.3.5", "runtime_packages", "fixed packages")
    require(torch.__version__ == "2.8.0+cpu" and np.__version__ == "2.3.5" and torch.version.cuda is None and runtime["python_version"] == f"{sys.version_info.major}.{sys.version_info.minor}", "actual_runtime", "actual CPU libraries")
    evidence = strict_json(evidence_raw)
    keys(evidence, set(raw_inputs) | (set(TRAINED_INPUTS) if args.mode == "restore" else set()), "typed material slots")
    project = None
    for name, raw in raw_inputs.items():
        entry = evidence[name]
        keys(entry, {"project_id", "manifest_sha256", "reference_id", "path", "measured_sha256"}, "project data " + name)
        require(all(type(v) is str and v for v in entry.values()) and HASH.fullmatch(entry["manifest_sha256"]) and entry["measured_sha256"] == sha256(raw), "input_evidence", name)
        relative = Path(entry["path"])
        require(not relative.is_absolute() and relative.parts and all(v not in (".", "..") for v in relative.parts), "input_evidence_path", name)
        current = (entry["project_id"], entry["manifest_sha256"])
        if project is None: project = current
        require(project == current, "input_evidence_project", "one captured project")
    return source, source_raw, evidence


def sampled_rows(order, rows):
    return [{"update": update, "rows": [{"row_index": int(index), "row_id": rows[index]["row_id"],
                                         "trajectory_id": rows[index]["trajectory_id"], "frame": rows[index]["frame"]}
                                        for index in indices]} for update, indices in enumerate(order, 1)]


def publish_checkpoint(network, directory, update):
    require(type(update) is int and update in CHECKPOINT_UPDATES, "checkpoint_update", "fixed retained states")
    name = f"checkpoint-{update:03d}.npz"
    inventory = save_weights(network, directory / name)
    raw = regular_bytes(directory / name, OUTPUTS[name][1])
    return {"update": update, **identity(raw), "inventory": inventory}


def finite_number(value, nonnegative=False):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def validate_magnitude(value):
    keys(value, {"min", "max", "rms", "zero_fraction", "spatial_range_max"}, "activation magnitude")
    require(all(finite_number(v) for v in value.values()) and value["min"] <= value["max"] and value["rms"] >= 0 and value["spatial_range_max"] >= 0 and 0 <= value["zero_fraction"] <= 1, "diagnostic_types", "finite typed activation statistics")


def validate_gradient(value):
    keys(value, {"l2", "max_abs", "nonzero_count"}, "gradient magnitude")
    require(finite_number(value["l2"], True) and finite_number(value["max_abs"], True) and type(value["nonzero_count"]) is int and value["nonzero_count"] >= 0, "diagnostic_types", "finite typed gradient statistics")


def validate_input_gradients(values):
    keys(values, model.CHANNELS, "ten input gradients")
    for value in values.values():
        validate_gradient(value)


def validate_observation(value, network):
    keys(value, {"pressure_loss", "residual_loss", "total_loss", "raw_head", "correction", "component_gradients", "raw_output_gradient", "global_gradient", "input_gradients", "activations"}, "batch observation")
    require(all(finite_number(value[n], True) for n in ("pressure_loss", "residual_loss", "total_loss")), "diagnostic_types", "component losses")
    for name in ("raw_head", "correction"):
        validate_magnitude(value[name])
    for name in ("raw_output_gradient", "global_gradient"):
        validate_gradient(value[name])
    validate_input_gradients(value["input_gradients"])
    component = value["component_gradients"]
    keys(component, {"pressure", "residual", "cosine"}, "component gradients")
    validate_gradient(component["pressure"])
    validate_gradient(component["residual"])
    require(component["cosine"] is None or finite_number(component["cosine"]) and abs(component["cosine"]) <= 1.0001, "diagnostic_types", "gradient angle")
    expected = {name + suffix for name, module in network.named_modules() if isinstance(module, (nn.Conv3d, nn.GroupNorm, nn.SiLU, nn.AdaptiveAvgPool3d)) for suffix in (":input", ":output")}
    keys(value["activations"], expected, "complete block/head/global observations")
    for item in value["activations"].values():
        validate_magnitude(item)


def validate_history(history, validation):
    require(type(history) is list and len(history) == 10, "restore_history", "ten retained logs")
    for update, value in zip(CHECKPOINT_UPDATES[1:], history):
        keys(value, {"update", "train_loss", "gradient_norm_before_clip"}, "retained training history")
        require(type(value["update"]) is int and value["update"] == update and all(finite_number(value[n], True) for n in ("train_loss", "gradient_norm_before_clip")), "restore_history", "typed update/loss/gradient")
    require(type(validation) is list and len(validation) == 2, "restore_validation", "only observational0/200")
    for update, value in zip((0, 200), validation):
        keys(value, {"update", "loss"}, "retained validation")
        require(type(value["update"]) is int and value["update"] == update and finite_number(value["loss"], True), "restore_validation", "typed fixed observations")


def validate_prediction_parity(arrays, retained):
    require(set(retained) == set(arrays) and all(retained[name].dtype == arrays[name].dtype and retained[name].shape == arrays[name].shape and retained[name].tobytes() == arrays[name].tobytes() for name in arrays), "restore_predictions", "closed keys/dtype/shape and bitwise final200 inference")


def validate_restore(receipt, closure, network, dataset, normalization, cases, raw_trained):
    expected = {"schema", "status", "mode", "optimizer_updates", "trained_optimizer_updates", "architecture", "parameter_count",
                "weights_sha256", "weights_inventory", "predictions_sha256", "predictions_inventory", "witness_cases", "per_case_inference_s",
                "closure", "sample_order_sha256", "training_history", "validation", "elapsed_s", "runtime_observations", "outputs",
                "originating_training_run", "checkpoints", "diagnostics_sha256", "initialization_mapping_sha256"}
    keys(receipt, expected, "GroupNorm trained receipt")
    require(receipt["schema"] == RECEIPT_SCHEMA and receipt["status"] == "complete" and receipt["mode"] == "train", "restore_identity", "completed new GroupNorm experiment only")
    validate_history(receipt["training_history"], receipt["validation"])
    require(finite_number(receipt["elapsed_s"], True) and receipt["originating_training_run"] is None, "restore_metadata", "original training elapsed/origin")
    runtime = receipt["runtime_observations"]
    keys(runtime, {"torch_version", "numpy_version", "python_version", "torch_threads", "torch_interop_threads", "device", "deterministic_algorithms", "peak_rss_bytes"}, "original runtime observations")
    require(canonical({n: runtime[n] for n in runtime if n != "peak_rss_bytes"}) == canonical({"torch_version": "2.8.0+cpu", "numpy_version": "2.3.5", "python_version": platform.python_version(), "torch_threads": 1, "torch_interop_threads": 1, "device": "cpu", "deterministic_algorithms": True}) and type(runtime["peak_rss_bytes"]) is int and 0 < runtime["peak_rss_bytes"] <= 4 * 1024**3, "restore_runtime", "actual typed CPU observations")
    require(type(receipt["optimizer_updates"]) is type(receipt["trained_optimizer_updates"]) is int and receipt["optimizer_updates"] == receipt["trained_optimizer_updates"] == 200, "restore_updates", "exact fixed200")
    require(canonical(receipt["architecture"]) == canonical(model.ARCHITECTURE) and type(receipt["parameter_count"]) is int and receipt["parameter_count"] == model.parameter_count(network), "restore_architecture", "new GroupNorm state identity")
    keys(receipt["closure"], closure, "restore closure")
    for name in set(closure) - {"input_evidence_sha256"}:
        require(canonical(receipt["closure"][name]) == canonical(closure[name]), "restore_closure", name)
    require(canonical(receipt["witness_cases"]) == canonical(cases), "restore_witness", "fixed32 cases")
    require(type(receipt["per_case_inference_s"]) is list and len(receipt["per_case_inference_s"]) == 32 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in receipt["per_case_inference_s"]), "restore_clocks", "finite32 observations")
    keys(receipt["outputs"], set(OUTPUTS) - {"receipt.json"}, "restore outputs")
    for artifact, spec in receipt["outputs"].items():
        keys(spec, {"bytes", "sha256", "role"}, artifact)
        require(type(spec["bytes"]) is int and 0 < spec["bytes"] <= OUTPUTS[artifact][1] and type(spec["sha256"]) is str and HASH.fullmatch(spec["sha256"]) and spec["role"] == OUTPUTS[artifact][0], "restore_outputs", artifact)
    for slot, (artifact, role) in TRAINED_INPUTS.items():
        if artifact == "receipt.json":
            continue
        keys(receipt["outputs"][artifact], {"bytes", "sha256", "role"}, artifact)
        require(canonical(receipt["outputs"][artifact]) == canonical({**identity(raw_trained[slot]), "role": role}), "restore_artifact_hash", artifact)
    require(receipt["weights_sha256"] == sha256(raw_trained["trained_weights"]) and receipt["predictions_sha256"] == sha256(raw_trained["trained_predictions"]) and receipt["diagnostics_sha256"] == sha256(raw_trained["trained_diagnostics"]), "restore_payload", "trained output linkage")
    require(raw_trained["trained_normalization"] == canonical(normalization), "restore_normalization", "original exact persisted RMS bytes")
    order = sample_order(dataset.train_rows)
    require(receipt["sample_order_sha256"] == sha256(order.tobytes()), "restore_sample_order", "unchanged fixed train-only sample order")
    checkpoint_names = {f"checkpoint-{update:03d}.npz" for update in CHECKPOINT_UPDATES}
    keys(receipt["checkpoints"], checkpoint_names, "eleven retained states")
    for update in CHECKPOINT_UPDATES:
        name = f"checkpoint-{update:03d}.npz"
        spec = receipt["checkpoints"][name]
        keys(spec, {"update", "bytes", "sha256", "inventory"}, name)
        require(type(spec["update"]) is int and spec["update"] == update and type(spec["bytes"]) is int, "restore_checkpoint", name)
        raw = raw_trained[f"trained_checkpoint_{update:03d}"]
        require(canonical({k: spec[k] for k in ("bytes", "sha256")}) == canonical(identity(raw)), "restore_checkpoint_hash", name)
        # Header/raw tensor checks bind every numeric intermediate state, without
        # loading it into a candidate model or searching its validation loss.
        require(set(spec["inventory"]) == set(network.state_dict()), "restore_checkpoint_inventory", name)
        for tensor_name, tensor in network.state_dict().items():
            require(spec["inventory"][tensor_name]["shape"] == list(tensor.shape) and spec["inventory"][tensor_name]["dtype"] == "<f4", "restore_checkpoint_inventory", tensor_name)
        read_npz(raw, spec["inventory"], OUTPUTS[name][1])
    require(raw_trained["trained_checkpoint_200"] == raw_trained["trained_weights"], "restore_candidate", "final200 only, exact trained byte copy")
    initial = model.initialization_proof(network)  # network is seeded initial here
    require(receipt["initialization_mapping_sha256"] == sha256(canonical(initial)), "restore_initialization", "byte-identical original seeded convolutions/head")
    require(canonical(receipt["checkpoints"]["checkpoint-000.npz"]["inventory"]) == canonical(state_inventory(network)), "restore_initialization", "checkpoint000 exactseed")
    diagnostics = strict_json(raw_trained["trained_diagnostics"])
    keys(diagnostics, {"schema", "candidate_update", "initialization_mapping", "checkpoint_hashes", "sampled_rows", "logs", "population_final", "validation_gradient_cases", "scope"}, "inherited diagnostics")
    require(diagnostics["schema"] == DIAGNOSTICS_SCHEMA and type(diagnostics["candidate_update"]) is int and diagnostics["candidate_update"] == 200, "restore_diagnostics", "same final-only candidate")
    require(canonical(diagnostics["initialization_mapping"]) == canonical(initial) and canonical(diagnostics["sampled_rows"]) == canonical(sampled_rows(order, dataset.train_rows)), "restore_diagnostics", "original initialization/full sampled-row identities")
    expected_hashes = {name: {key: spec[key] for key in ("sha256", "bytes")} for name, spec in receipt["checkpoints"].items()}
    require(canonical(diagnostics["checkpoint_hashes"]) == canonical(expected_hashes), "restore_diagnostics", "every retained state link")
    require(type(diagnostics["logs"]) is list and [entry["update"] for entry in diagnostics["logs"]] == list(CHECKPOINT_UPDATES[1:]) and all(type(entry["update"]) is int for entry in diagnostics["logs"]), "restore_diagnostics", "ten fixed before/after logs")
    require(type(diagnostics["population_final"]) is list and len(diagnostics["population_final"]) == 384 and type(diagnostics["validation_gradient_cases"]) is list and len(diagnostics["validation_gradient_cases"]) == 8, "restore_diagnostics", "fixed development observations")
    for update, entry in zip(CHECKPOINT_UPDATES[1:], diagnostics["logs"]):
        keys(entry, {"update", "sampled_rows", "before_update", "after_update", "checkpoint_sha256"}, "before/after retained log")
        require(canonical(entry["sampled_rows"]) == canonical(sampled_rows(order, dataset.train_rows)[update - 1]["rows"]) and entry["checkpoint_sha256"] == receipt["checkpoints"][f"checkpoint-{update:03d}.npz"]["sha256"], "restore_diagnostics", "sampled log/checkpoint linkage")
        validate_observation(entry["before_update"], network)
        validate_observation(entry["after_update"], network)
    for index, (row, value) in enumerate(zip(dataset.train_rows + dataset.validation_rows, diagnostics["population_final"])):
        keys(value, {"row_index", "row_id", "split", "pressure_loss", "residual_loss", "raw_head", "correction", "last_decoder_pre_silu", "last_decoder_pre_groupnorm", "head_features", "raw_output_gradient", "global_gradient", "input_gradients"}, "population observation")
        require(type(value["row_index"]) is int and value["row_index"] == index and value["row_id"] == row["row_id"] and value["split"] == ("train" if index < 288 else "validation") and all(finite_number(value[n], True) for n in ("pressure_loss", "residual_loss")), "restore_population", "typed original development row")
        for name in ("raw_head", "correction", "last_decoder_pre_silu", "last_decoder_pre_groupnorm", "head_features"):
            validate_magnitude(value[name])
        for name in ("raw_output_gradient", "global_gradient"):
            validate_gradient(value[name])
        validate_input_gradients(value["input_gradients"])
    expected_validation = [(288 + local, row["row_id"]) for local, row in enumerate(dataset.validation_rows) if row["frame"] in (0, 23)]
    for (index, row_id), value in zip(expected_validation, diagnostics["validation_gradient_cases"]):
        keys(value, {"row_index", "row_id", "observations"}, "validation gradient case")
        require(type(value["row_index"]) is int and value["row_index"] == index and value["row_id"] == row_id, "restore_gradient_cases", "fixed eight cases")
        validate_observation(value["observations"], network)
    require(type(diagnostics["scope"]) is str, "restore_diagnostics", "textual observation scope")
    return diagnostics


def run(args):
    started = time.perf_counter()
    output_directory(args.output_dir)
    observed_peak_rss_bytes()
    raw_inputs = {name: regular_bytes(getattr(args, name), 32 * 1024**2) for name in ("train0", "train1", "validation", "splits", "origin")}
    config_raw = regular_bytes(args.config, 64 * 1024)
    runtime_raw = regular_bytes(args.runtime_manifest, 16 * 1024**2)
    evidence_raw = regular_bytes(args.input_evidence, 1024**2)
    source, source_raw, evidence = validate_source(args, raw_inputs, config_raw, runtime_raw, evidence_raw)
    reader = model.load_original("data", model.ORIGINAL_DATA_SHA256)
    dataset = reader.load_data(args.train0, args.train1, args.validation, args.splits, args.origin, expected_native_residual_origin_sha256=source["native_residual_origin_sha256"])
    observed_peak_rss_bytes()
    normalization = model.fit_normalization(dataset.train, dataset.train_rows)
    normalization_raw = canonical(normalization)
    require(sha256(normalization_raw) == NORMALIZATION_SHA256, "original_normalization_hash", "unchanged train-only physical RMS bytes")
    cases = witness_cases(dataset)
    model.configure_cpu()
    torch.manual_seed(20261014)
    network = model.PressureCorrectionNet()
    initialization = model.initialization_proof(network)
    observed_peak_rss_bytes()
    initialization_sha = sha256(canonical(initialization))
    closure = {"source_witness_sha256": sha256(source_raw), "source_commit": source["source_commit"], "config_sha256": sha256(config_raw),
               "runtime_snapshot_sha256": sha256(runtime_raw), "input_evidence_sha256": sha256(evidence_raw),
               "input_sha256": {name: sha256(raw) for name, raw in raw_inputs.items()}, "splits_sha256": sha256(raw_inputs["splits"]),
               "normalization_sha256": sha256(normalization_raw), "witness_sha256": sha256(canonical(cases))}
    order = sample_order(dataset.train_rows)
    origin = None
    if args.mode == "train":
        checkpoints = {"checkpoint-000.npz": publish_checkpoint(network, args.output_dir, 0)}
        initial_validation = evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        progress("validation", 0, 200, loss=initial_validation)
        observed_peak_rss_bytes()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        history, logs = [], []
        for update, indices in enumerate(order, 1):
            observed_peak_rss_bytes()
            before = observe_batch(network, dataset.train, indices, normalization) if update % 20 == 0 else None
            observed_peak_rss_bytes()
            loss, gradient = base.one_update(network, optimizer, dataset.train, indices, normalization)
            observed_peak_rss_bytes()
            if update % 20 == 0:
                after = observe_batch(network, dataset.train, indices, normalization)
                name = f"checkpoint-{update:03d}.npz"
                checkpoints[name] = publish_checkpoint(network, args.output_dir, update)
                history.append({"update": update, "train_loss": loss, "gradient_norm_before_clip": gradient})
                logs.append({"update": update, "sampled_rows": sampled_rows(order, dataset.train_rows)[update - 1]["rows"],
                             "before_update": before, "after_update": after, "checkpoint_sha256": checkpoints[name]["sha256"]})
                progress("training", update, 200, loss=loss, gradient_norm=gradient, peak_rss_bytes=observed_peak_rss_bytes())
        final_validation = evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        validation = [{"update": 0, "loss": initial_validation}, {"update": 200, "loss": final_validation}]
        gradient_cases = []
        for local, row in enumerate(dataset.validation_rows):
            if row["frame"] in (0, 23):
                gradient_cases.append({"row_index": 288 + local, "row_id": row["row_id"],
                                       "observations": observe_batch(network, dataset.validation, [local], normalization)})
        diagnostics = {"schema": DIAGNOSTICS_SCHEMA, "candidate_update": 200, "initialization_mapping": initialization,
                       "checkpoint_hashes": {name: {key: spec[key] for key in ("sha256", "bytes")} for name, spec in checkpoints.items()},
                       "sampled_rows": sampled_rows(order, dataset.train_rows), "logs": logs,
                       "population_final": observe_population(network, dataset, normalization), "validation_gradient_cases": gradient_cases,
                       "scope": "Retained initial/every20 states are observations, never candidates/resume/selection. Final200 alone; development teacher states only."}
        write_json_new(args.output_dir / "diagnostics.json", diagnostics)
        regular_bytes(args.output_dir / "diagnostics.json", OUTPUTS["diagnostics.json"][1])
        copy_new(args.output_dir / "weights.npz", regular_bytes(args.output_dir / "checkpoint-200.npz", OUTPUTS["weights.npz"][1]))
        weights_inventory = checkpoints["checkpoint-200.npz"]["inventory"]
        copy_new(args.output_dir / "normalization.json", normalization_raw)
        copy_new(args.output_dir / "splits.json", raw_inputs["splits"])
        updates = 200
    else:
        raw_trained = {slot: regular_bytes(getattr(args, slot), OUTPUTS[artifact][1]) for slot, (artifact, _) in TRAINED_INPUTS.items()}
        origin = validate_trained_custody(evidence, raw_trained, args.input_evidence)
        trained = strict_json(raw_trained["trained_receipt"])
        diagnostics = validate_restore(trained, closure, network, dataset, normalization, cases, raw_trained)
        require(raw_trained["trained_splits"] == raw_inputs["splits"], "restore_splits", "exact original288/96 histories")
        load_weights(network, raw_trained["trained_weights"], trained["weights_inventory"])
        for slot, (artifact, _) in TRAINED_INPUTS.items():
            if artifact != "receipt.json":
                copy_new(args.output_dir / artifact, raw_trained[slot])
        checkpoints, history, validation, weights_inventory = trained["checkpoints"], trained["training_history"], trained["validation"], trained["weights_inventory"]
        updates = 0
    arrays, clocks = make_predictions(network, dataset, normalization, cases)
    predictions_inventory = prediction_inventory(arrays)
    if args.mode == "train":
        with (args.output_dir / "predictions.npz").open("xb") as stream:
            np.savez(stream, **arrays)
    else:
        retained = read_npz(raw_trained["trained_predictions"], trained["predictions_inventory"], OUTPUTS["predictions.npz"][1])
        validate_prediction_parity(arrays, retained)
    summary = {"format": "tuldok_result_summary_v1", "title": "GroupNorm pressure correction: " + args.mode,
               "metrics": [{"name": "Optimizer updates", "value": updates, "unit": "updates"}, {"name": "Trained optimizer updates", "value": 200, "unit": "updates"},
                           {"name": "Parameters", "value": model.parameter_count(network), "unit": "parameters"},
                           {"name": "Final validation loss", "value": validation[-1]["loss"], "unit": "ratio"},
                           {"name": "Retained checkpoints", "value": 11, "unit": "checkpoints"}],
               "scope": ["Fixed non-affine GroupNorm intervention; original loss/data/optimizer policy unchanged.",
                         "Final200 alone is the candidate; checkpoints are observations, never selection/resume.",
                         "No final-test/rollout/generalization or solver acceleration qualification.", "Exact final/history byte restore, zero new optimizer updates." if updates == 0 else "New fixed200 experiment; validation observational."], "checkpoint": "weights.npz"}
    write_json_new(args.output_dir / "summary.json", summary)
    outputs = {name: {**identity(regular_bytes(args.output_dir / name, limit)), "role": role} for name, (role, limit) in OUTPUTS.items() if name != "receipt.json"}
    require(outputs["weights.npz"]["sha256"] == outputs["checkpoint-200.npz"]["sha256"], "candidate_checkpoint", "exact final200 bytes")
    observations = {"torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "device": "cpu", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "peak_rss_bytes": observed_peak_rss_bytes()}
    receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "mode": args.mode, "optimizer_updates": updates, "trained_optimizer_updates": 200,
               "architecture": model.ARCHITECTURE, "parameter_count": model.parameter_count(network), "weights_sha256": outputs["weights.npz"]["sha256"],
               "weights_inventory": weights_inventory, "predictions_sha256": outputs["predictions.npz"]["sha256"], "predictions_inventory": predictions_inventory,
               "witness_cases": cases, "per_case_inference_s": clocks, "closure": closure, "sample_order_sha256": sha256(order.tobytes()),
               "training_history": history, "validation": validation, "elapsed_s": time.perf_counter() - started, "runtime_observations": observations,
               "outputs": outputs, "originating_training_run": origin, "checkpoints": checkpoints, "diagnostics_sha256": outputs["diagnostics.json"]["sha256"],
               "initialization_mapping_sha256": initialization_sha}
    require(sum(spec["bytes"] for spec in outputs.values()) + len(canonical(receipt)) <= FIXED_CONFIG["max_output_bytes"], "output_bound", "all18 flat outputs")
    write_json_new(args.output_dir / "receipt.json", receipt)
    regular_bytes(args.output_dir / "receipt.json", OUTPUTS["receipt.json"][1])
    progress("complete", updates, 200, validation_loss=validation[-1]["loss"], retained_checkpoints=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "restore"))
    for name in ("train0", "train1", "validation", "splits", "origin", "config", "runtime-manifest", "source-witness", "input-evidence", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for slot in TRAINED_INPUTS:
        parser.add_argument("--" + slot.replace("_", "-"), type=Path)
    args = parser.parse_args()
    try:
        require(args.mode != "restore" or all(getattr(args, slot) is not None for slot in TRAINED_INPUTS), "restore_inputs", "all17 originating artifacts")
        require(args.mode != "train" or all(getattr(args, slot) is None for slot in TRAINED_INPUTS), "train_inputs", "no trained states as new training inputs")
        run(args)
    except (TrainingContractError, model.ModelContractError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(__import__("json").dumps({"status": "failed", "error": getattr(error, "code", "groupnorm_failure"), "detail": str(error)}, allow_nan=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
