"""Causal pressure-reuse inference composition from exact managed learned modules."""
from mixed_modal_sampler import load_model as load_base
from learned_pressure_history import LearnedPressureHistory


def load_model(pair, library_path):
    return LearnedPressureHistory(load_base(pair, library_path), mode='reuse')
