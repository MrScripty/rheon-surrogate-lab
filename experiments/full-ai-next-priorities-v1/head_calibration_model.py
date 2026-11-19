"""Unchanged C107 prediction graph with explicit head-only training metadata."""
import copy
import json
from pathlib import Path
import calibration_base_model
from head_descriptor import describe

ARCHITECTURE={}


def __getattr__(name):return getattr(calibration_base_model,name)


def select(config):
    calibration_base_model.select(config)
    calibration=json.loads(Path(__file__).with_name('calibration.json').read_text())
    ARCHITECTURE.clear();ARCHITECTURE.update(describe(calibration_base_model.ARCHITECTURE,calibration))


PressureCorrectionNet=calibration_base_model.PressureCorrectionNet
