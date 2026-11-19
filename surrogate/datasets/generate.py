"""Simulate a dataset definition's scenes and keep every output step on disk.

A scene is written to a temporary folder and renamed when complete, so an
interrupted run leaves only whole scenes and can simply be started again.
Scenes that are already complete on disk are never simulated again.
"""
import json
import math
import shutil
import time

import numpy as np

from ..solver import RheonProbe
from . import load, scene
from .store import data_root, is_complete, read_records, record, steps_of, write_records

# Measured on the lab's existing compressed state files (27 to 36 bytes per cell per state).
BYTES_PER_CELL_STATE = 36


def select(definition, splits=None, scenes=None):
    rows = [row for row in definition['scenes']
            if (not splits or row['split'] in splits) and (not scenes or row['id'] in scenes)]
    missing = set(scenes or ()) - {row['id'] for row in rows}
    if missing:
        raise ValueError(f'Not in this dataset (or not in the chosen splits): {sorted(missing)}')
    return rows


def plan(definition, rows, root=None):
    """What generating `rows` involves: per split, scenes still to run, states, cells and estimated bytes."""
    records = read_records(definition, root)
    summary = {}
    for row in rows:
        entry = summary.setdefault(row['split'], {'scenes': 0, 'complete': 0, 'partial': 0, 'states': 0,
                                                  'cell_states': 0, 'estimated_bytes': 0})
        steps = steps_of(definition, row)
        entry['scenes'] += 1
        if is_complete(records.get(row['id']), steps):
            entry['complete'] += 1
            continue
        entry['partial'] += row['id'] in records
        cell_states = math.prod(row['counts']) * (steps + 1)
        entry['states'] += steps + 1
        entry['cell_states'] += cell_states
        entry['estimated_bytes'] += cell_states * BYTES_PER_CELL_STATE
    return summary


def _save(path, state, labels):
    arrays = {'pressure': state.pressure, 'vx': state.u, 'vy': state.v, 'vz': state.w, 'tracer': state.tracer}
    np.savez_compressed(path, **{name: tensor.numpy().reshape(-1) for name, tensor in {**arrays, **labels}.items()})


def generate_scene(definition, row, directory):
    """Simulate one scene into `directory`, replacing whatever is there. Returns its index entry."""
    parameters = scene(definition, row)
    steps = parameters['steps']
    partial = directory.with_name(directory.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started = time.perf_counter()
    with RheonProbe(parameters) as probe:
        _, state = probe.state()
        zeros = {name: state.pressure.clone().zero_() for name in ('rhs', 'native_input_residual', 'pressure_base')}
        _save(partial / 'state-000.npz', state, zeros)
        for step in range(1, steps + 1):
            report, labels = probe.advance_reference()
            reached, state = probe.state()
            if not report['advanced'] or reached != step:
                raise RuntimeError(f"The solver rejected step {step} of {row['id']}: {report.get('quality')}")
            _save(partial / f'state-{step:03d}.npz', state, labels)
    receipt = {'identifier': row['id'], 'split': row['split'], 'group': row['group'], 'dataset': definition['id'],
               'origin': 'generated', 'solver': definition['solver'],
               'pressure_solve': "exact cosine transform on the solver's requests",
               'scene': parameters, 'states': steps + 1, 'elapsed_seconds': time.perf_counter() - started}
    (partial / 'receipt.json').write_text(json.dumps(receipt) + '\n')
    if directory.exists():
        shutil.rmtree(directory)       # an incomplete scene (for example linked windows): the full one replaces it
    partial.rename(directory)
    return record(row, range(steps + 1))


def generate(name, root=None, splits=None, scenes=None, allow_reserved=False, report=print):
    """Generate the chosen scenes of dataset `name` that are not complete on disk; returns the index path."""
    definition = load(name)
    rows = select(definition, splits, scenes)
    if any(row['split'] == 'reserved' for row in rows) and not allow_reserved:
        raise ValueError('This selection includes reserved scenes. Reserved scenes are for final checks only; '
                         'pass allow_reserved to generate them on purpose.')
    directory = data_root(root) / definition['id']
    records = read_records(definition, root)
    bank = write_records(definition, records, root)
    for row in rows:
        if is_complete(records.get(row['id']), steps_of(definition, row)):
            continue
        started = time.perf_counter()
        records[row['id']] = generate_scene(definition, row, directory / row['id'])
        write_records(definition, records, root)
        report(f"generated {row['id']} ({records[row['id']]['states']} states, {time.perf_counter() - started:.1f} s)")
    return bank
