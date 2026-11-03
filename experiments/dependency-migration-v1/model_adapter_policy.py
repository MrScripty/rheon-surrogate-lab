"""Numerical module requirements for the current CPU package, without recipes."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MAX_CELLS = 65536


def require(condition, code, detail=None):
    if not condition:
        raise ValueError(code if detail is None else code + ': ' + detail)


def frozen_model():
    path = HERE / 'frozen_gn.py'
    spec = importlib.util.spec_from_file_location('variable_pressure_frozen_gn', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
