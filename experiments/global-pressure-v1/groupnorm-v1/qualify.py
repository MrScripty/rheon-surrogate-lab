"""Fixed arithmetic qualification; no optimizer construction or updates."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

if __package__:
    from . import model, train
else:
    import model
    import train


def fixture(count=16):
    values = torch.arange(count**3, dtype=torch.float64).reshape(1, 1, count, count, count)
    base = torch.sin(values * .013) * .01
    base = base - base[..., :1, :1, :1]
    reference = base + (torch.sin(values * .027) - torch.sin(values[..., :1, :1, :1] * .027)) * .03
    spacing = torch.tensor([[.1, .2, .3]], dtype=torch.float64)
    rho = torch.tensor([1.], dtype=torch.float64)
    rhs = model.full_operator(reference, spacing, rho)
    residual = rhs - model.full_operator(base, spacing, rho)
    arrays = {"base": base[:, 0].numpy(), "reference": reference[:, 0].numpy(), "rhs": rhs[:, 0].numpy(),
              "native_input_residual": residual[:, 0].numpy(), "spacing": spacing.numpy(), "rho": rho.numpy(),
              "dt": np.array([.01], dtype="<f8")}
    rows = [{"trajectory_id": "arithmetic-fixture", "row_id": "arithmetic-fixture:000", "frame": 0}]
    normalization = model.fit_normalization(arrays, rows)
    return arrays, rows, normalization


def gauge_measure():
    h = torch.tensor([[.5, .25, 2.]], dtype=torch.float64)
    rho = torch.tensor([2.], dtype=torch.float64)
    p = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2)
    b = torch.tensor([-4.625, -3.625, 3.375, 4.375, -4.375, -3.375, 3.625, 4.625], dtype=torch.float64).reshape_as(p)
    assert torch.equal(model.full_operator(p, h, rho), b)
    normalization = {"schema": "rheon-global-pressure-normalization-v1", "scales": {name: {"value": 1.} for name in ("pressure", "residual", "rhs", "hx", "hy", "hz", "rho", "dt")}}
    raw = (p / 13).clone().requires_grad_()
    def loss(v):
        return model.pressure_loss(model.gauge_correction(v, .25), p, b, h, rho, normalization).sum()
    gradient = torch.autograd.grad(loss(raw), raw)[0]
    direction = torch.cos(torch.arange(8, dtype=torch.float64)).reshape_as(p)
    exact = float((gradient * direction).sum())
    epsilon = 1e-6
    fd = float((loss(raw.detach() + epsilon * direction) - loss(raw.detach() - epsilon * direction)) / (2 * epsilon))
    return abs(exact - fd)


def groupnorm_measure():
    # Independent closed derivative of normalization, biased population variance.
    x = torch.linspace(-.5, .7, 48, dtype=torch.float64).reshape(1, 48, 1, 1, 1).requires_grad_()
    gn = nn.GroupNorm(4, 48, eps=1e-5, affine=False).double()
    weights = torch.cos(torch.arange(48, dtype=torch.float64)).reshape_as(x)
    actual = torch.autograd.grad((gn(x) * weights).sum(), x)[0]
    expected = []
    for group, w in zip(x.detach().reshape(4, 12), weights.reshape(4, 12)):
        centered = group - group.mean()
        variance = centered.square().mean()
        jacobian = (torch.eye(12, dtype=torch.float64) - torch.ones(12, 12, dtype=torch.float64) / 12) / torch.sqrt(variance + 1e-5)
        jacobian -= torch.outer(centered, centered) / (12 * (variance + 1e-5)**1.5)
        expected.append(jacobian.T @ w)
    assert torch.allclose(actual.flatten(), torch.cat(expected), atol=1e-12, rtol=1e-12)
    direction = torch.sin(torch.arange(48, dtype=torch.float64)).reshape_as(x)
    epsilon = 1e-6
    fd = float(((gn(x.detach() + epsilon * direction) * weights).sum() - (gn(x.detach() - epsilon * direction) * weights).sum()) / (2 * epsilon))
    return abs(fd - float((actual * direction).sum()))


def measure():
    model.configure_cpu()
    torch.manual_seed(20261014)
    network = model.PressureCorrectionNet().eval()
    initialization = model.initialization_proof(network)
    before = train.state_inventory(network)
    arrays, _rows, normalization = fixture()
    batch = train.tensors(arrays, [0])
    inputs = model.build_inputs(batch["native_input_residual"], batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization)
    group_stats = []
    handles = []
    for module in network.modules():
        if isinstance(module, nn.GroupNorm):
            def hook(_m, _args, output):
                grouped = output.detach().double().reshape(output.shape[0], 4, -1)
                group_stats.append((float(grouped.mean(2).abs().max()), float(grouped.var(2, unbiased=False).max())))
            handles.append(module.register_forward_hook(hook))
    started = time.perf_counter()
    with torch.no_grad():
        original = network(inputs)
    seconds16 = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    # Same batch size and first row, different neighbour: per-case GN independence
    # tested separately from convolution batch-size association differences.
    neighbour0 = torch.cat((inputs, torch.zeros_like(inputs)))
    neighbour1 = torch.cat((inputs, inputs * 3 + 1))
    with torch.no_grad():
        bn_error = float((network(neighbour0)[0] - network(neighbour1)[0]).abs().max())
        network.train()
        train_eval_error = float((network(inputs) - original).abs().max())
        network.eval()
        # Fixed near-constant and measured-chain-sized amplitude fixtures.
        for x in (torch.ones_like(inputs), torch.ones_like(inputs) + inputs * 1e-6, inputs * 1e6):
            assert torch.isfinite(network(x)).all()
    arrays32, _, norm32 = fixture(32)
    b32 = train.tensors(arrays32, [0])
    x32 = model.build_inputs(b32["native_input_residual"], b32["base"], b32["spacing"], b32["rho"], b32["dt"], norm32)
    started = time.perf_counter()
    with torch.no_grad():
        shape32 = network(x32)
    seconds32 = time.perf_counter() - started
    assert shape32.shape == (1, 1, 32, 32, 32) and torch.isfinite(shape32).all()
    x = inputs.clone().requires_grad_()
    retained = []
    handle = network.global_pool.register_forward_hook(lambda _m, _args, out: retained.append(out))
    started = time.perf_counter()
    raw = network(x)
    handle.remove()
    corrected = model.gauge_correction(raw, model._scale_values(normalization)["pressure"])
    # Gauge output at far corner: global_pool derivative isolates actual pooled
    # branch transmission, rather than GN's own sample-wide spatial coupling.
    distant_gradient, pool_gradient = torch.autograd.grad(corrected[0, 0, -1, -1, -1], [x, retained[0]], retain_graph=True)
    objective = lambda v: model.pressure_loss(batch["base"] + model.gauge_correction(v, model._scale_values(normalization)["pressure"]), batch["reference"], batch["rhs"], batch["spacing"], batch["rho"], normalization).mean()
    loss = objective(raw)
    name = "encoders.0.layers.0.weight"
    parameter = dict(network.named_parameters())[name]
    direction = torch.cos(torch.arange(parameter.numel(), dtype=torch.float64) * .37).reshape_as(parameter).float()
    direction = direction / direction.double().norm().float()
    parameter_gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    analytic = float((parameter_gradient.double() * direction.double()).sum())
    # Immutable functional parameter substitution, never .data/load_state_dict
    # on the retained model, and no optimizer state/step.
    errors = []
    for epsilon in (1e-3, 5e-4):
        plus = torch.func.functional_call(network, {name: parameter.detach() + epsilon * direction}, (inputs,))
        minus = torch.func.functional_call(network, {name: parameter.detach() - epsilon * direction}, (inputs,))
        derivative = float(((objective(plus) - objective(minus)) / (2 * epsilon)).detach())
        errors.append(abs(derivative - analytic))
    autograd_s = time.perf_counter() - started
    # Amplitude is an observational gate: nonlinear GN is not presumed linear.
    # Evaluate residual-only amplitudes .5/1/2 and report deviation from homogeneity
    # and derivative sensitivity; do not retune if pressure usefulness is unknown.
    def pressure(amplitude):
        return model.predict_pressure(network, batch["native_input_residual"] * amplitude, batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization) - batch["base"]
    with torch.no_grad():
        chalf, c1, c2 = pressure(.5), pressure(1.), pressure(2.)
        amplitude_error = max(float((c2 - 2 * c1).norm() / (2 * c1.norm())),
                              float((chalf - .5 * c1).norm() / (.5 * c1.norm())))
    amplitude = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    amplitude_output = pressure(amplitude)
    amplitude_grad = torch.autograd.grad(amplitude_output.square().sum(), amplitude)[0]
    bypass = model.predict_pressure(network, torch.zeros_like(batch["native_input_residual"]), batch["base"], batch["spacing"], batch["rho"], batch["dt"], normalization)
    p = torch.tensor([0., 1., -2., .5, 3., -1.], dtype=torch.float64).reshape(1, 1, 1, 2, 3)
    action = model.full_operator(p, torch.tensor([[.25, .4, .7]], dtype=torch.float64), torch.tensor([1.25], dtype=torch.float64)).flatten()
    native = torch.tensor([-1.071, 2.8839999999999995, -3.038, -2.065, 6.524, -3.2339999999999995], dtype=torch.float64)
    assert train.canonical(before) == train.canonical(train.state_inventory(network))
    metrics = {"initialization_tensor_count": len(initialization["mapping"]), "groupnorm_module_count": len(group_stats),
               "group_mean_max_abs": max(v[0] for v in group_stats), "group_variance_max": max(v[1] for v in group_stats),
               "batch_neighbor_max_abs_error": bn_error, "train_eval_max_abs_error": train_eval_error,
               "zero_residual_max_abs_error": float((bypass - batch["base"]).abs().max()),
               "gauge_finite_difference_max_abs_error": gauge_measure(), "groupnorm_finite_difference_max_abs_error": groupnorm_measure(),
               "model_parameter_gradient_abs": abs(analytic), "model_parameter_fd_abs_error": max(errors),
               "distant_input_derivative_abs": float(distant_gradient[0, 0, 0, 0, 0].abs()),
               "global_pool_gradient_l1": float(pool_gradient.abs().sum()), "native_operator_max_abs_error": float((action - native).abs().max()),
               "amplitude_response_relative_error": amplitude_error, "amplitude_derivative_abs": float(amplitude_grad.abs()),
               "dry_forward_s_16": seconds16, "dry_forward_s_32": seconds32, "dry_autograd_s": autograd_s,
               "observed_peak_rss_bytes": train.observed_peak_rss_bytes()}
    assert set(metrics) == train.QUALIFICATION_METRICS
    return metrics, initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch.object(torch.optim.Optimizer, "__init__", side_effect=RuntimeError("optimizer construction forbidden during qualification")):
        metrics, initialization = measure()
    train.write_json_new(args.output, {"status": "pretraining_arithmetic_observations", "metrics": metrics,
                                      "initialization_mapping": initialization, "no_optimizer_constructed": True,
                                      "executable_sources": train.executable_sources()})
    print(train.canonical(metrics).decode(), end="")


if __name__ == "__main__":
    main()
