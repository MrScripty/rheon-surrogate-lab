"""Fixed eager pressure-correction model and independently qualified FV loss.

Inputs contain native residual and previous accepted pressure; reference pressure
is a loss label only. No optimizer, execution service or final-data policy lives
in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = {
    "schema": "rheon-global-pressure-model-v1",
    "input_channels": 10,
    "encoder_widths": [8, 16, 24, 32],
    "bottleneck_width": 48,
    "decoder_widths": [32, 24, 16, 8],
    "convolutions_per_block": 2,
    "kernel": [3, 3, 3],
    "padding": "replicate",
    "activation": "silu",
    "pool": "average_2x2x2",
    "global_path": "add_spatial_mean_broadcast",
    "upsampling": "nearest_to_skip_shape",
    "head": "linear_1x1x1",
    "dtype": "float32",
    "pressure_dtype": "float64",
    "counts_supported": [16, 32],
    "gauge": "subtract_first_correction_cell",
}
CHANNELS = ["native_residual", "previous_pressure", "x", "y", "z", "hx", "hy", "hz", "rho", "dt"]
PHYSICAL_UNITS = {"hx": "m", "hy": "m", "hz": "m", "rho": "kg/m3", "dt": "s"}
MAX_PARAMETERS = 1_000_000
MAX_WEIGHTS_BYTES = 8 * 1024 * 1024


class ModelContractError(ValueError):
    """Unsupported model input, normalization or state representation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelContractError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def numeric_sha256(values: np.ndarray) -> str:
    require(values.dtype.kind in "fi", "numeric array required")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def configure_cpu() -> None:
    """One eager CPU thread; caller owns environment/runtime capture."""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(inputs, outputs, 3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(outputs, outputs, 3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class PressureCorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        previous = 10
        self.encoders = nn.ModuleList()
        for width in ARCHITECTURE["encoder_widths"]:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.AvgPool3d(2)
        self.bottleneck = ConvBlock(previous, 48)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.decoders = nn.ModuleList()
        previous = 48
        for width, skip in zip([32, 24, 16, 8], [32, 24, 16, 8]):
            self.decoders.append(ConvBlock(previous + skip, width))
            previous = width
        self.head = nn.Conv3d(8, 1, 1)
        require(parameter_count(self) <= MAX_PARAMETERS, "parameter budget exceeded")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        require(values.ndim == 5 and values.shape[1] == 10, "expected N,10,Z,Y,X")
        require(values.device.type == "cpu" and values.dtype == torch.float32, "eager f32 CPU inputs required")
        require(tuple(values.shape[2:]) in ((16, 16, 16), (32, 32, 32)), "only exact 16/32 cubes supported")
        require(bool(torch.isfinite(values).all()), "nonfinite model input")
        skips = []
        for encoder in self.encoders:
            values = encoder(values)
            skips.append(values)
            values = self.pool(values)
        values = self.bottleneck(values)
        values = values + self.global_pool(values).expand_as(values)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            values = F.interpolate(values, size=skip.shape[2:], mode="nearest")
            values = decoder(torch.cat((values, skip), dim=1))
        return self.head(values)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _pressure_contract(pressure: torch.Tensor, spacing: torch.Tensor, rho: torch.Tensor) -> None:
    require(pressure.ndim == 5 and pressure.shape[1] == 1, "pressure N,1,Z,Y,X required")
    require(pressure.dtype == torch.float64 and pressure.device.type == "cpu", "f64 CPU pressure required")
    require(spacing.shape == (pressure.shape[0], 3) and spacing.dtype == torch.float64, "spacing N,3 f64 required")
    require(rho.shape == (pressure.shape[0],) and rho.dtype == torch.float64, "rho N f64 required")
    require(spacing.device.type == rho.device.type == "cpu", "CPU conditioning required")
    require(bool(torch.isfinite(pressure).all() and torch.isfinite(spacing).all() and torch.isfinite(rho).all()), "nonfinite operator input")
    require(bool((spacing > 0).all() and (rho > 0).all()), "positive geometry/density required")


def full_operator(pressure: torch.Tensor, spacing: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Full sealed FV graph action; cell zero is an ordinary physical row.

    Interior face differences contribute with opposite signs to adjacent cells.
    The coefficient area*(1/(rho*h)) matches the native physical operator.
    """
    _pressure_contract(pressure, spacing, rho)
    result = torch.zeros_like(pressure)
    for physical_axis, tensor_axis in enumerate((4, 3, 2)):
        other = [axis for axis in range(3) if axis != physical_axis]
        area = spacing[:, other[0]] * spacing[:, other[1]]
        weight = (area * (1.0 / (rho * spacing[:, physical_axis]))).reshape(-1, 1, 1, 1, 1)
        require(bool(torch.isfinite(weight).all() and (weight > 0).all()), "invalid finite-volume coefficient")
        difference = pressure.narrow(tensor_axis, 1, pressure.shape[tensor_axis] - 1) - pressure.narrow(tensor_axis, 0, pressure.shape[tensor_axis] - 1)
        lower_padding = [0] * 6
        upper_padding = [0] * 6
        lower_padding[2 * physical_axis] = 1
        upper_padding[2 * physical_axis + 1] = 1
        result = result + weight * (F.pad(difference, lower_padding) - F.pad(difference, upper_padding))
    return result


def gauge_correction(raw: torch.Tensor, pressure_scale: float) -> torch.Tensor:
    require(type(pressure_scale) in (int, float) and math.isfinite(pressure_scale) and pressure_scale > 0, "positive pressure scale required")
    correction = raw.to(dtype=torch.float64) * pressure_scale
    first = correction[:, :, :1, :1, :1]
    return correction - first


def _scale_values(normalization: Mapping) -> dict[str, float]:
    require(normalization.get("schema") == "rheon-global-pressure-normalization-v1", "normalization schema")
    names = {"residual", "pressure", "rhs", "hx", "hy", "hz", "rho", "dt"}
    require(set(normalization["scales"]) == names, "normalization scale inventory")
    values = {}
    for name, spec in normalization["scales"].items():
        value = spec["value"]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "positive finite scale " + name)
        values[name] = float(value)
    return values


def fit_normalization(train: Mapping[str, np.ndarray], train_rows: list[dict]) -> dict:
    """Train-only RMS and provenance. Validation bytes/labels are never consulted."""
    base = train["base"]
    require(base.ndim == 4 and base.dtype == np.dtype("<f8"), "base N,Z,Y,X f64")
    count = base.shape[0]
    require(count > 0 and len(train_rows) == count, "normalization train coverage")
    fields = {"residual": train["native_input_residual"], "pressure": base, "rhs": train["rhs"]}
    scales = {}
    for name, values in fields.items():
        require(values.shape == base.shape and values.dtype == np.dtype("<f8") and np.isfinite(values).all(), "normalization numeric field " + name)
        value = float(np.sqrt(np.mean(values * values, dtype=np.float64)))
        require(math.isfinite(value) and value > 0, "training-only zero/nonfinite RMS " + name)
        scales[name] = {"value": value, "units": "Pa" if name == "pressure" else "m3/s2", "method": "train_cell_rms", "source": "previous_pressure" if name == "pressure" else "native_input_residual" if name == "residual" else "original_rhs"}
    physical = {name: train["spacing"][:, index] for index, name in enumerate(("hx", "hy", "hz"))}
    physical.update(rho=train["rho"], dt=train["dt"])
    for name, values in physical.items():
        require(values.shape == (count,) and values.dtype == np.dtype("<f8") and np.isfinite(values).all() and (values > 0).all(), "physical train field " + name)
        constant = bool(np.all(values == values[0]))
        value = 1.0 if constant else float(np.sqrt(np.mean(values * values, dtype=np.float64)))
        scales[name] = {"value": value, "units": PHYSICAL_UNITS[name], "method": "constant_unit_scale" if constant else "train_frame_rms", "source": name}
    trajectory_ids = list(dict.fromkeys(row["trajectory_id"] for row in train_rows))
    frame_counts = dict(Counter(row["trajectory_id"] for row in train_rows))
    provenance = {
        "trajectory_ids": trajectory_ids,
        "frame_counts": frame_counts,
        "frames": count,
        "cells_per_frame": math.prod(base.shape[1:]),
        "row_mapping_sha256": hashlib.sha256(canonical(train_rows)).hexdigest(),
        "numeric_sha256": {name: numeric_sha256(train[name]) for name in ("base", "rhs", "native_input_residual", "spacing", "rho", "dt")},
    }
    return {"schema": "rheon-global-pressure-normalization-v1", "channels": CHANNELS, "scales": scales, "training_provenance": provenance}


def build_inputs(native_residual: torch.Tensor, base: torch.Tensor, spacing: torch.Tensor, rho: torch.Tensor, dt: torch.Tensor, normalization: Mapping) -> torch.Tensor:
    _pressure_contract(base, spacing, rho)
    require(native_residual.shape == base.shape and native_residual.dtype == torch.float64 and native_residual.device.type == "cpu", "native residual f64 pressure shape")
    require(dt.shape == rho.shape and dt.dtype == torch.float64 and dt.device.type == "cpu" and bool(torch.isfinite(dt).all() and (dt > 0).all()), "positive dt N f64")
    require(bool(torch.isfinite(native_residual).all()), "nonfinite native residual")
    require(bool((base[:, :, 0, 0, 0] == 0).all()), "previous pressure must be gauge fixed")
    scale = _scale_values(normalization)
    batch, _, nz, ny, nx = base.shape
    coordinates = []
    for tensor_axis, count in ((4, nx), (3, ny), (2, nz)):
        values = (torch.arange(count, dtype=torch.float64) + 0.5) / count
        shape = [1] * 5
        shape[tensor_axis] = count
        coordinates.append(values.reshape(shape).expand(batch, 1, nz, ny, nx))
    conditioning = [spacing[:, axis] / scale[name] for axis, name in enumerate(("hx", "hy", "hz"))] + [rho / scale["rho"], dt / scale["dt"]]
    channels = [native_residual / scale["residual"], base / scale["pressure"], *coordinates]
    channels.extend(values.reshape(batch, 1, 1, 1, 1).expand(batch, 1, nz, ny, nx) for values in conditioning)
    inputs = torch.cat(channels, dim=1).to(dtype=torch.float32)
    require(bool(torch.isfinite(inputs).all()), "nonfinite normalized input")
    return inputs


def predict_pressure(model: nn.Module, native_residual: torch.Tensor, base: torch.Tensor, spacing: torch.Tensor, rho: torch.Tensor, dt: torch.Tensor, normalization: Mapping) -> torch.Tensor:
    inputs = build_inputs(native_residual, base, spacing, rho, dt, normalization)
    active = (native_residual.flatten(1) != 0).any(dim=1)
    if not bool(active.any()):
        return base.clone()
    selected = torch.nonzero(active, as_tuple=False).flatten()
    raw = model(inputs.index_select(0, selected))
    require(raw.shape == (selected.numel(), 1, *base.shape[2:]) and bool(torch.isfinite(raw).all()), "malformed/nonfinite model output")
    corrected = base.index_select(0, selected) + gauge_correction(raw, _scale_values(normalization)["pressure"])
    return base.clone().index_copy(0, selected, corrected)


def pressure_loss(prediction: torch.Tensor, reference: torch.Tensor, rhs: torch.Tensor, spacing: torch.Tensor, rho: torch.Tensor, normalization: Mapping) -> torch.Tensor:
    """Per-frame accepted loss, with f64 reductions and distinct RHS/residual RMS."""
    _pressure_contract(prediction, spacing, rho)
    require(reference.shape == rhs.shape == prediction.shape and reference.dtype == rhs.dtype == torch.float64, "f64 labels/RHS shape")
    require(bool(torch.isfinite(reference).all() and torch.isfinite(rhs).all()), "nonfinite loss labels")
    scale = _scale_values(normalization)
    cells = prediction[0].numel()
    eps_p2 = (1e-6 * math.sqrt(cells) * scale["pressure"]) ** 2
    eps_b2 = (1e-6 * math.sqrt(cells) * scale["rhs"]) ** 2
    pressure_error = (prediction - reference).flatten(1).square().sum(dim=1)
    denominator_p = reference.flatten(1).square().sum(dim=1).clamp_min(eps_p2)
    residual = full_operator(prediction, spacing, rho) - rhs
    residual_error = residual.flatten(1).square().sum(dim=1)
    denominator_b = rhs.flatten(1).square().sum(dim=1).clamp_min(eps_b2)
    return 0.5 * pressure_error / denominator_p + 0.5 * residual_error / denominator_b
