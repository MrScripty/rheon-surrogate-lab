"""Versioned mixed-domain reader over the retained ordinary training interface."""
from historical_data import *
import historical_data
import methods_view


def load(raw, manifest):
    if manifest.get('schema') != 'rheon-pressure-methods-learning-view-v4':
        return historical_data.load(raw, manifest)
    parents = methods_view.decode(raw, manifest)
    packet = dict(schema='rheon-pressure-methods-physical-training-v1', max_cells=2097152,
        units='SI', model_dt_s=1/60,
        trajectory_ids=[p['entry']['trajectory_id'] for p in parents])
    return packet, parents
