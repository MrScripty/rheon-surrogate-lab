"""Retained pressure network, fixed C53 scaling and explicit paired fit objective."""
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
    ARCHITECTURE['fine_tuning'] = {k:v for k,v in contract.items() if k != 'initial_inventory'}
    ARCHITECTURE['method_comparison'] = dict(normalization='fixed original C53 train-only scales',
        supervision='Recorded physical pressure requests only; inherited auxiliary banks are disabled in the captured trainer.',
        sampling='Balanced spatial tiers; private RNG; fixed update budget.')


def fit_normalization(parents, previous):
    if not parents or any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Training-only input view required')
    return json.loads(Path(__file__).with_name('fixed-normalization.json').read_text())


def training_loss(network, sample, normalization, criterion=None):
    return fine_tuning.loss(baseline, network, sample, normalization, criterion)
