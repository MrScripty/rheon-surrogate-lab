"""Fixed non-affine GroupNorm intervention on the frozen pressure architecture.

Original physical preprocessing/operator/loss are imported from exact verified
bytes. Convolution/head initialization is unchanged; GroupNorm has no parameters.
"""
from __future__ import annotations

import copy
import hashlib
import stat
import sys
import types
from pathlib import Path

import torch
from torch import nn

ORIGINAL_MODEL_SHA256 = "4ecc0e1f2b5d9627f1fe19a46486d3b62b11509ef035b9aa6fc925ddcb406dbb"
ORIGINAL_TRAIN_SHA256 = "68cf5971f89397fa465f3b1a63f0e05dd5b04d0b292a3c2fd7be3d307b48b33d"
ORIGINAL_DATA_SHA256 = "c1395f278910a22a6caa92b37fe6d29d8c7e562aeb6489c13e2cba6acf8ab4cd"
PACKAGE = "rheon_groupnorm_frozen_original"


def load_original(name: str, expected_sha256: str):
    """Execute one immutable verified byte copy in an isolated private package.

    Captured flat sources take precedence. The lab fallback is the exact original
    sibling namespace; an invalid captured file never silently falls back.
    """
    captured = Path(__file__).with_name("original_" + name + ".py")
    try:
        captured.lstat()
        path = captured
    except FileNotFoundError:
        path = Path(__file__).parent.parent / "training" / (name + ".py")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or not 0 < info.st_size <= 1024**2:
        raise RuntimeError("invalid pinned original source: " + name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("pinned original source hash mismatch: " + name)
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = []
        sys.modules[PACKAGE] = package
    identifier = PACKAGE + "." + name
    if identifier in sys.modules:
        module = sys.modules[identifier]
        if module.__pinned_sha256__ != expected_sha256:
            raise RuntimeError("private original module identity mismatch")
        return module
    module = types.ModuleType(identifier)
    module.__file__ = str(path)
    module.__package__ = PACKAGE
    module.__pinned_sha256__ = expected_sha256
    sys.modules[identifier] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(identifier, None)
        raise
    return module


original = load_original("model", ORIGINAL_MODEL_SHA256)
ARCHITECTURE = copy.deepcopy(original.ARCHITECTURE)
ARCHITECTURE.update(schema="rheon-global-pressure-groupnorm-model-v1",
                    block_normalization={"name": "GroupNorm", "groups": 4, "epsilon": 1e-5,
                                         "affine": False, "position": "each block convolution before SiLU"})

ModelContractError = original.ModelContractError
require = original.require
canonical = original.canonical
numeric_sha256 = original.numeric_sha256
configure_cpu = original.configure_cpu
parameter_count = original.parameter_count
full_operator = original.full_operator
gauge_correction = original.gauge_correction
fit_normalization = original.fit_normalization
build_inputs = original.build_inputs
predict_pressure = original.predict_pressure
pressure_loss = original.pressure_loss
CHANNELS = original.CHANNELS
MAX_PARAMETERS = original.MAX_PARAMETERS
MAX_WEIGHTS_BYTES = original.MAX_WEIGHTS_BYTES
_scale_values = original._scale_values


class PressureCorrectionNet(original.PressureCorrectionNet):
    def __init__(self):
        # This executes the exact original RNG operations, then inserts only
        # parameter-free transformations. The head is the original head object.
        super().__init__()
        for block in [*self.encoders, self.bottleneck, *self.decoders]:
            first, first_silu, second, second_silu = block.layers
            block.layers = nn.Sequential(first, nn.GroupNorm(4, first.out_channels, eps=1e-5, affine=False), first_silu,
                                         second, nn.GroupNorm(4, second.out_channels, eps=1e-5, affine=False), second_silu)
        require(parameter_count(self) == 372377, "fixed non-affine parameter inventory")


def parameter_mapping(network) -> dict[str, str]:
    """New tensor name -> original tensor name; all38 tensors explicitly bound."""
    result = {}
    for name in network.state_dict():
        result[name] = name.replace(".layers.3.", ".layers.2.")
    require(len(result) == len(set(result.values())) == 38, "one-to-one original parameter mapping")
    return result


def initialization_proof(network) -> dict:
    """Compare to independently reconstructed original seed without advancing RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261014)
        reference = original.PressureCorrectionNet()
    old = reference.state_dict()
    mapping = parameter_mapping(network)
    require(set(mapping.values()) == set(old), "complete original state inventory")
    inventory = {}
    for name, original_name in mapping.items():
        new_value = network.state_dict()[name].detach()
        old_value = old[original_name].detach()
        require(new_value.dtype == old_value.dtype == torch.float32 and torch.equal(new_value, old_value),
                "seed20261014 original parameter byte mismatch: " + name)
        raw = new_value.numpy().tobytes()
        require(raw == old_value.numpy().tobytes(), "original signed-zero/byte mismatch")
        inventory[name] = {"original_name": original_name, "shape": list(new_value.shape),
                           "dtype": "<f4", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"schema": "rheon-global-pressure-groupnorm-initialization-v1", "seed": 20261014,
            "original_model_sha256": ORIGINAL_MODEL_SHA256, "mapping": inventory,
            "parameter_count": parameter_count(network), "raw_weight_bytes": sum(v["bytes"] for v in inventory.values())}
