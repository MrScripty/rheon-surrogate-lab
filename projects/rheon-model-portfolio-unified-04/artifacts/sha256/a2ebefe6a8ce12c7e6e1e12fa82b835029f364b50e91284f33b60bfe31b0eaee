"""Unchanged pressure network and deployment; explicit fine-tuning objective."""
import copy
import json
from pathlib import Path

import baseline_model as baseline
import fine_tuning

ARCHITECTURE = {}


def __getattr__(name):
    return getattr(baseline, name)


def select(config):
    baseline.select(config)
    contract = json.loads(Path(__file__).with_name('fine-tuning.json').read_text())
    fine_tuning.select(contract)
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(baseline.ARCHITECTURE))
    ARCHITECTURE['fine_tuning'] = {k: v for k, v in contract.items() if k != 'initial_inventory'}


def training_loss(network, sample, normalization, criterion=None):
    return fine_tuning.loss(baseline, network, sample, normalization, criterion)
