"""Unchanged GN arithmetic; experiment selection and host setup live elsewhere."""
from contract import HERE, LAB, load_module


def load():
    captured = HERE / "gn_model.py"
    path = captured if captured.exists() else LAB / "experiments/global-pressure-v1/groupnorm-v1/model.py"
    model = load_module("local_pressure_gn_arithmetic", path)
    helpers = model.load_original("train", model.ORIGINAL_TRAIN_SHA256)
    return model, helpers
