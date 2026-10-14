"""Read-only, fixed-population diagnosis of the accepted frozen pressure model.

No optimizer or training loop is instantiated. Initial weights are reconstructed
from the original seed, not recovered from a retained initial checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "experiments/global-pressure-v1/training"
sys.path.insert(0, str(TRAINING))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import data
import model
import train

SOURCES = {
    "data.py": "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd",
    "model.py": "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb",
    "train.py": "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d",
    "test_model.py": "bf9351c5bfb86f4803c160d262740abaa09e252bce10eb5d70f6be2a07c41b7e",
}
PACKET = {
    "train-0.npz": "6b3d0c4a57df0198b4fe13f7e887b8fa238fbf3fd34877f9dc598c157d2b55dc",
    "train-1.npz": "d6d7980a51c17b79a6567c870aca9001108779798cee490101429e00814a9f02",
    "validation.npz": "17a9faa84723208d1bd9d6a3a8b1f5921806e93b85d8983d2041404fb295855d",
    "splits.json": "fbb9edddfa08877e0834fcdf6e0d10e8f4769c5883fc6597b829325c2784650a",
    "origin.json": "dee7811a95b85a4c9fd2a228163ab39288d85bdeee804d09804c24eb7c92663d",
}
ARTIFACTS = {
    "weights.npz": "f46aec5e0de59bca9a2a7a701e85c4904782c11b7ac6c113cfd3f7625c8391fc",
    "receipt.json": "5d69507cf12aad14fd5b614a7b7b7ed4b6c644efed8c0f18781dfc72bf56d8b5",
    "normalization.json": "8a68445ce25b46645b50939b13f67b974d60f1c44d8e69ce0a078f24dc81458f",
    "predictions.npz": "1413b35302eb8ee3c26ddb78bf2b95c9e914456f3bed25bf7f091f1a552e7267",
}
RESIDUAL_ORIGIN = "26c33fac5df17b66fadba036368f2008ec172a860231ff75822cd2ef9dba11b8"
WALL_LIMIT_S = 180
RSS_LIMIT_BYTES = 4 * 1024**3
STARTED = 0.0


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state_sha(network):
    return sha(model.canonical(train.state_inventory(network)))


def guard():
    check(time.perf_counter() - STARTED < WALL_LIMIT_S, "180s work budget exceeded")
    check(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RSS_LIMIT_BYTES,
          "4GiB observed RSS budget exceeded")


def deadline(_signal, _frame):
    raise RuntimeError("180s wall deadline reached; no successful report")


def bind(directory, inventory, label):
    result = {}
    for name, expected in inventory.items():
        path = directory / name
        check(path.is_file() and not path.is_symlink(), label + " regular file: " + name)
        raw = path.read_bytes()
        check(sha(raw) == expected, label + " frozen SHA drift: " + name)
        result[name] = {"sha256": expected, "bytes": len(raw)}
    return result


def stats(value):
    """Spatial variation within each channel, distinct from between-channel range."""
    v = value.detach().to(torch.float64)
    check(bool(torch.isfinite(v).all()), "nonfinite measured tensor")
    flat = v.flatten()
    spatial = v.reshape(v.shape[0], -1) if v.ndim >= 2 else v.reshape(1, -1)
    ranges = spatial.amax(1) - spatial.amin(1)
    return {"min": float(flat.min()), "max": float(flat.max()),
            "mean": float(flat.mean()), "rms": float(flat.square().mean().sqrt()),
            "max_abs": float(flat.abs().max()), "zero_count": int((flat == 0).sum()),
            "elements": flat.numel(), "spatial_range_max": float(ranges.max()),
            "spatial_range_mean": float(ranges.mean())}


def gradient_stats(value):
    v = value.detach().to(torch.float64)
    check(bool(torch.isfinite(v).all()), "nonfinite derivative")
    return {"l2": float(v.square().sum().sqrt()), "l1": float(v.abs().sum()),
            "max_abs": float(v.abs().max()), "sum": float(v.sum()),
            "nonzero_count": int((v != 0).sum()), "elements": v.numel()}


def components(prediction, batch, normalization):
    scale = model._scale_values(normalization)
    cells = prediction[0].numel()
    dp = batch["reference"].flatten(1).square().sum(1).clamp_min((1e-6 * math.sqrt(cells) * scale["pressure"])**2)
    db = batch["rhs"].flatten(1).square().sum(1).clamp_min((1e-6 * math.sqrt(cells) * scale["rhs"])**2)
    lp = 0.5 * (prediction - batch["reference"]).flatten(1).square().sum(1) / dp
    lb = 0.5 * (model.full_operator(prediction, batch["spacing"], batch["rho"]) - batch["rhs"]).flatten(1).square().sum(1) / db
    check(torch.equal(lp + lb, model.pressure_loss(prediction, batch["reference"], batch["rhs"], batch["spacing"], batch["rho"], normalization)),
          "diagnostic components differ from original loss")
    return lp, lb


class Stages:
    """Scoped observational hooks; all handles removed before derivative probes."""
    def __init__(self, network):
        self.values = {}
        self.handles = []
        for name, module in network.named_modules():
            if isinstance(module, nn.Conv3d):
                self.handles.append(module.register_forward_pre_hook(self.pre(name)))
                self.handles.append(module.register_forward_hook(self.post(name)))
            elif isinstance(module, (nn.SiLU, nn.AdaptiveAvgPool3d)):
                self.handles.append(module.register_forward_hook(self.post(name)))

    def pre(self, name):
        def hook(_module, args):
            self.values[name + ":pre_conv"] = args[0].detach()
        return hook

    def post(self, name):
        def hook(module, args, output):
            self.values[name + ":output"] = output.detach()
            if isinstance(module, nn.SiLU):
                # Mathematical derivative observed at the actual input, using f64
                # for tiny tails; this is not a claim about f32 autograd behavior.
                v = args[0].detach().double()
                s = torch.sigmoid(v)
                self.values[name + ":silu_derivative_f64"] = s * (1 + v * (1 - s))
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()


def copied_forward(network, inputs):
    """Ephemeral arithmetic probe, same stages; deliberately outside f32 API.

    The production model remains untouched. The explicit copied-stage f32
    equivalence is checked before f64 observations are interpreted.
    """
    skips = []
    v = inputs
    for encoder in network.encoders:
        v = encoder(v)
        skips.append(v)
        v = network.pool(v)
    v = network.bottleneck(v)
    v = v + network.global_pool(v).expand_as(v)
    for decoder, skip in zip(network.decoders, reversed(skips)):
        v = F.interpolate(v, size=skip.shape[2:], mode="nearest")
        v = decoder(torch.cat((v, skip), dim=1))
    return network.head(v)


def population(network, dataset, normalization, label):
    records = []
    hooks = Stages(network)
    try:
        with torch.no_grad():
            for split, arrays, rows in (("train", dataset.train, dataset.train_rows),
                                        ("validation", dataset.validation, dataset.validation_rows)):
                for first in range(0, len(rows), 2):
                    guard()
                    batch = train.tensors(arrays, np.arange(first, first + 2))
                    x = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization)
                    raw = network(x)
                    active = (batch["native_input_residual"].flatten(1) != 0).any(1)
                    check(bool(active.all()), "population contains bypass row; diagnose explicitly before proceeding")
                    correction = model.gauge_correction(raw, model._scale_values(normalization)["pressure"])
                    prediction = batch["base"] + correction
                    lp, lb = components(prediction, batch, normalization)
                    for offset in range(2):
                        row = rows[first + offset]
                        records.append({"row_index": len(records), "trajectory_id": row["trajectory_id"],
                                        "frame": row["frame"], "split": split,
                                        "witness": row["frame"] in (0, 23),
                                        "pressure_loss": float(lp[offset]), "residual_loss": float(lb[offset]),
                                        "total_loss": float((lp + lb)[offset]),
                                        "raw_head": stats(raw[offset]), "correction": stats(correction[offset]),
                                        "reference_delta": stats((batch["reference"] - batch["base"])[offset]),
                                        "input_channels": {name: stats(x[offset, channel:channel + 1]) for channel, name in enumerate(model.CHANNELS)},
                                        "stages": {name: stats(v[offset]) for name, v in hooks.values.items()}})
                    hooks.values.clear()
    finally:
        hooks.close()
    aggregates = {}
    for split in ("train", "validation"):
        selected = [row for row in records if row["split"] == split]
        aggregates[split] = {name: math.fsum(row[name] for row in selected) / len(selected)
                             for name in ("pressure_loss", "residual_loss", "total_loss")}
        aggregates[split].update(rows=len(selected), zero_correction_rows=sum(row["correction"]["max_abs"] == 0 for row in selected),
                                 constant_raw_head_rows=sum(row["raw_head"]["spatial_range_max"] == 0 for row in selected))
    print(json.dumps({"phase": "population", "model": label, "rows": len(records), "aggregate": aggregates}), flush=True)
    return {"aggregates": aggregates, "rows": records}


def directions(shape, dtype):
    v = torch.cos(torch.arange(math.prod(shape), dtype=torch.float64) * 0.37).reshape(shape)
    v = v / v.square().sum().sqrt()
    return v.to(dtype)


def difference(objective, value, direction, steps):
    # All additions create temporary tensor values; authoritative weights are
    # never assigned, nor is any optimizer state constructed.
    results = []
    for epsilon in steps:
        plus = objective(value + epsilon * direction)
        minus = objective(value - epsilon * direction)
        results.append({"epsilon": epsilon, "central_difference": float((plus - minus) / (2 * epsilon)),
                        "plus_loss": float(plus), "minus_loss": float(minus)})
    return results


def derivative_cases(network, dataset, normalization, label):
    results = []
    scale = model._scale_values(normalization)["pressure"]
    # Cast a separate model copy once; source f32 weights are exact representable
    # f64 inputs. This is observation, not another candidate trained model.
    precise = model.PressureCorrectionNet().double().eval()
    precise.load_state_dict({name: value.detach().double().clone() for name, value in network.state_dict().items()})
    for local, row in enumerate(dataset.validation_rows):
        if row["frame"] not in (0, 23):
            continue
        guard()
        batch = train.tensors(dataset.validation, [local])
        x = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization).requires_grad_()
        retained_stages = {}
        handles = []
        for name, module in (("deepest_silu", network.bottleneck.layers[3]),
                             ("global_pool", network.global_pool),
                             ("head_features", network.decoders[-1].layers[3])):
            handles.append(module.register_forward_hook(lambda _m, _args, out, n=name: retained_stages.update({n: out})))
        raw = network(x)
        for handle in handles:
            handle.remove()
        correction = model.gauge_correction(raw, scale)
        prediction = batch["base"] + correction
        objective = lambda candidate: model.pressure_loss(batch["base"] + model.gauge_correction(candidate, scale), batch["reference"], batch["rhs"], batch["spacing"], batch["rho"], normalization).mean()
        lp, lb = components(prediction, batch, normalization)
        loss = (lp + lb).mean()
        parameters = list(network.named_parameters())
        pressure_grads = torch.autograd.grad(lp.mean(), [raw, *[p for _, p in parameters]], retain_graph=True)
        residual_grads = torch.autograd.grad(lb.mean(), [raw, *[p for _, p in parameters]], retain_graph=True)
        gradients = torch.autograd.grad(loss, [raw, correction, x, *retained_stages.values(), *[p for _, p in parameters]])
        raw_grad, correction_grad, input_grad, *tail = gradients
        stage_grads, parameter_grads = tail[:len(retained_stages)], tail[len(retained_stages):]
        gp = torch.cat([g.detach().double().flatten() for g in pressure_grads[1:]])
        gb = torch.cat([g.detach().double().flatten() for g in residual_grads[1:]])
        npg, nbg = float(gp.norm()), float(gb.norm())
        result = {"row_index": len(dataset.train_rows) + local, "trajectory_id": row["trajectory_id"], "frame": row["frame"],
                  "loss": float(loss.detach()), "raw_head": stats(raw[0]), "correction": stats(correction[0]),
                  "raw_head_gradient": gradient_stats(raw_grad), "gauge_correction_gradient": gradient_stats(correction_grad),
                  "stage_gradients": {name: gradient_stats(g) for name, g in zip(retained_stages, stage_grads)},
                  "loss_gradient_components": {"pressure_loss": float(lp.detach()), "residual_loss": float(lb.detach()),
                                               "pressure_parameter_l2": npg, "residual_parameter_l2": nbg,
                                               "pressure_over_residual_l2": npg / nbg if nbg else None,
                                               "parameter_cosine": float((gp * gb).sum()) / (npg * nbg) if npg and nbg else None,
                                               "pressure_raw_gradient": gradient_stats(pressure_grads[0]),
                                               "residual_raw_gradient": gradient_stats(residual_grads[0])},
                  "input_channel_gradients": {name: gradient_stats(input_grad[:, channel]) for channel, name in enumerate(model.CHANNELS)},
                  "parameter_gradients": {name: gradient_stats(g) for (name, _), g in zip(parameters, parameter_grads)}}
        # Raw-output FD in f64 prevents perturbations disappearing at raw f32 ULP.
        raw64 = raw.detach().double()
        output_direction = directions(raw.shape, torch.float64)
        direct_gradient = torch.autograd.grad(objective(raw64.requires_grad_()), raw64)[0]
        result["raw_direction"] = {"normalization": "unit_l2_cosine_0.37", "autograd_f64": float((direct_gradient * output_direction).sum()),
                                   "fd": difference(objective, raw64.detach(), output_direction, [1e-4, 1e-6]),
                                   "constant_gauge_fd": difference(objective, raw64.detach(), torch.ones_like(raw64), [1e-4])}
        with torch.no_grad():
            check(torch.equal(copied_forward(network, x.detach()), raw.detach()), "copied f32 forward differs from exact production API")
            head_features = retained_stages["head_features"].detach()
            unbiased32 = F.conv3d(head_features, network.head.weight.detach(), bias=None)
            unbiased64 = F.conv3d(head_features.double(), network.head.weight.detach().double(), bias=None)
            biased64 = unbiased64 + network.head.bias.detach().double().reshape(1, 1, 1, 1, 1)
            bias = network.head.bias.detach().numpy()[0]
            bias_ulp = float(abs(np.nextafter(bias, np.float32(np.inf)) - bias))
            result["same_f32_features_head_arithmetic"] = {
                "bias_f32": float(bias), "bias_ulp_toward_positive": bias_ulp,
                "unbiased_head_f32": stats(unbiased32[0]), "unbiased_head_f64": stats(unbiased64[0]),
                "biased_head_f64": stats(biased64[0]),
                "unbiased_head_f64_gauge_correction": stats(model.gauge_correction(unbiased64, scale)[0]),
                "unbiased_spatial_range_over_bias_ulp": stats(unbiased64[0])["spatial_range_max"] / bias_ulp,
                "scope": "same frozen f32 decoder features and weights; only head arithmetic/gauge measured; no model candidate or state assignment"}
        copied_raw = copied_forward(precise, x.detach().double())
        copied_loss = objective(copied_raw)
        copied_gradient = torch.autograd.grad(copied_loss, precise.head.weight)[0]
        head_direction = directions(precise.head.weight.shape, torch.float64)
        # Stateless head substitution by algebra, retaining all precise backbone
        # activations. Head is affine so this is an exact copied-head direction.
        head_inputs = []
        handle = precise.head.register_forward_pre_hook(lambda _m, args: head_inputs.append(args[0].detach()))
        with torch.no_grad():
            copied_raw = copied_forward(precise, x.detach().double())
        handle.remove()
        variation = F.conv3d(head_inputs[0], head_direction)
        unbiased_all64 = F.conv3d(head_inputs[0], precise.head.weight.detach(), bias=None)
        result["precision_probe"] = {"kind": "ephemeral copied stages f64, exact f32 weight values; normalized inputs already f32",
                                     "raw_head_f64": stats(copied_raw[0]),
                                     "correction_f64": stats(model.gauge_correction(copied_raw, scale)[0]),
                                     "raw_f32_vs_f64_max_abs": float((raw.detach().double() - copied_raw).abs().max()),
                                     "head_weight_gradient_f64": gradient_stats(copied_gradient),
                                     "unbiased_all_f64_head": stats(unbiased_all64[0]),
                                     "unbiased_all_f64_gauge_correction": stats(model.gauge_correction(unbiased_all64, scale)[0]),
                                     "head_direction_autograd_f64": float((copied_gradient * head_direction).sum()),
                                     "head_direction_fd": [{"epsilon": epsilon,
                                                            "central_difference": float((objective(copied_raw + epsilon * variation) - objective(copied_raw - epsilon * variation)) / (2 * epsilon))}
                                                           for epsilon in (1e-4, 1e-6)]}
        results.append(result)
    print(json.dumps({"phase": "derivatives", "model": label, "cases": len(results)}), flush=True)
    return results


def run(args):
    global STARTED
    STARTED = time.perf_counter()
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT_S)
    permitted_report = ROOT / "docs/plans/global-pressure-diagnosis-v1/reports/diagnosis.json"
    diagnosis_directory = Path(__file__).parent
    resolved_output = args.output.resolve()
    check(resolved_output == permitted_report or resolved_output.is_relative_to(diagnosis_directory),
          "output must be the owned report or a new diagnosis-directory artifact")
    check(not any(parent.is_symlink() for parent in (args.output, *args.output.parents)), "symlink output path")
    check(not args.output.exists(), "immutable report exists")
    script_before = sha(Path(__file__).read_bytes())
    source_before = bind(TRAINING, SOURCES, "source")
    packet_before = bind(args.packet, PACKET, "packet2")
    artifact_before = bind(args.artifacts, ARTIFACTS, "trained artifact")
    check(torch.__version__ == "2.8.0+cpu" and np.__version__ == "2.3.5" and torch.version.cuda is None, "runtime drift")
    model.configure_cpu()
    receipt = train.strict_json((args.artifacts / "receipt.json").read_bytes())
    normalization = train.strict_json((args.artifacts / "normalization.json").read_bytes())
    check(receipt["mode"] == "train" and receipt["optimizer_updates"] == 200, "original receipt mode")
    dataset = data.load_data(*(args.packet / name for name in PACKET), expected_native_residual_origin_sha256=RESIDUAL_ORIGIN)
    guard()
    check(len(dataset.train_rows) == 288 and len(dataset.validation_rows) == 96, "development coverage")
    check(model.canonical(model.fit_normalization(dataset.train, dataset.train_rows)) == model.canonical(normalization), "train-only normalization reproduction")
    torch.manual_seed(20261014)
    initial = model.PressureCorrectionNet().eval()
    final = model.PressureCorrectionNet().eval()
    train.load_weights(final, args.artifacts / "weights.npz", receipt["weights_inventory"])
    states_before = {"initial": state_sha(initial), "final": state_sha(final)}
    result = {"schema": "rheon-global-pressure-frozen-diagnosis-v1", "status": "complete",
              "scope": "all384 existing development rows, no final/test access; no optimizer constructed or updates",
              "initial_identity": "seed20261014 reconstruction, NOT a retained initial checkpoint",
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
                          "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "device": "cpu"},
              "before": {"sources": source_before, "packet2": packet_before, "trained_artifacts": artifact_before, "states": states_before},
              "retained_history": receipt["training_history"],
              "history_limit": "Only ten every20-update logs exist; no intermediate weights/Adam states inspected or reconstructed."}
    for label, network in (("initial", initial), ("final", final)):
        result[label] = population(network, dataset, normalization, label)
        result[label]["derivatives"] = derivative_cases(network, dataset, normalization, label)
    validation = []
    for index, (label, network) in enumerate((("initial", initial), ("final", final))):
        guard()
        observed = train.evaluate(network, dataset.validation, dataset.validation_rows, normalization)
        retained = receipt["validation"][index]["loss"]
        check(observed == retained, "original batch2 validation loss reproduction failed: " + label)
        validation.append({"model": label, "observed": observed, "retained": retained, "bitwise_float_equal": observed == retained})
    cases = train.witness_cases(dataset)
    check(cases == receipt["witness_cases"], "witness identity drift")
    witnessed, _clocks = train.make_predictions(final, dataset, normalization, cases)
    retained = train.read_npz(args.artifacts / "predictions.npz", receipt["predictions_inventory"], 8 * 1024**2)
    check(all(witnessed[name].tobytes() == retained[name].tobytes() for name in witnessed), "32 retained witness bytes differ")
    result["reproduction"] = {"validation": validation, "witness_cases": cases, "all32_prediction_bytes_equal": True,
                              "witness_zero_corrections": sum(result["final"]["rows"][case["row_index"]]["correction"]["max_abs"] == 0 for case in cases)}
    result["after"] = {"sources": bind(TRAINING, SOURCES, "source after"), "packet2": bind(args.packet, PACKET, "packet after"),
                       "trained_artifacts": bind(args.artifacts, ARTIFACTS, "artifact after"), "states": {"initial": state_sha(initial), "final": state_sha(final)}}
    check(result["before"] == result["after"], "source/input/checkpoint mutation")
    result["cost"] = {"elapsed_s": time.perf_counter() - STARTED, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "wall_budget_s": WALL_LIMIT_S, "observed_rss_budget_bytes": RSS_LIMIT_BYTES,
                      "measurement": "single local invocation, file verification/data load/forward/hooks/autograd/precision probes/witness recheck included; RSS observation, not adversarial aggregate containment"}
    result["diagnostic_script_sha256"] = sha(Path(__file__).read_bytes())
    check(result["diagnostic_script_sha256"] == script_before, "diagnostic source mutation")
    guard()
    raw = model.canonical(result) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": "complete", "report_sha256": sha(raw), "bytes": len(raw), "cost": result["cost"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=ROOT / "runs/global-pressure-v1-training/development-packet2")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/global-pressure-v1-training-runtime/train-stage/artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/plans/global-pressure-diagnosis-v1/reports/diagnosis.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
