"""Simulate a dataset definition's scenes and keep every output step on disk.

Output goes under a data root (default: `data/reference/` in this repository,
which Git ignores), one folder per dataset:

    <root>/<dataset>/bank.json                     index that surrogate.data.load_bank reads
    <root>/<dataset>/<scene>/receipt.json          the scene's parameters and how it was produced
    <root>/<dataset>/<scene>/state-NNN.npz         one file per output step, 0..steps

A scene is written to a temporary folder and renamed when complete, so an
interrupted run leaves only whole scenes and can simply be started again.
"""
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from ..solver import REPOSITORY, RheonProbe, probe_for
from . import load, scene

DEFAULT_ROOT = REPOSITORY / 'data' / 'reference'
# Measured on the lab's existing compressed state files (27 to 36 bytes per cell per state).
BYTES_PER_CELL_STATE = 36


def data_root(root=None):
    """Where generated datasets live: the argument, else $SURROGATE_DATA, else data/reference in the repository."""
    return Path(root or os.environ.get('SURROGATE_DATA') or DEFAULT_ROOT)


def select(definition, splits=None, scenes=None):
    rows = [row for row in definition['scenes']
            if (not splits or row['split'] in splits) and (not scenes or row['id'] in scenes)]
    missing = set(scenes or ()) - {row['id'] for row in rows}
    if missing:
        raise ValueError(f'Not in this dataset (or not in the chosen splits): {sorted(missing)}')
    return rows


def plan(definition, rows, root):
    """What generating `rows` involves: per split, scenes still to run, states, cells and estimated bytes."""
    directory = root / definition['id']
    summary = {}
    for row in rows:
        entry = summary.setdefault(row['split'], {'scenes': 0, 'done': 0, 'not_wired': 0, 'states': 0,
                                                  'cell_states': 0, 'estimated_bytes': 0})
        steps = row.get('steps', definition.get('steps'))
        entry['scenes'] += 1
        if (directory / row['id'] / 'receipt.json').is_file():
            entry['done'] += 1
            continue
        if row.get('solver_substeps', 1) != 1:
            entry['not_wired'] += 1
            continue
        cell_states = math.prod(row['counts']) * (steps + 1)
        entry['states'] += steps + 1
        entry['cell_states'] += cell_states
        entry['estimated_bytes'] += cell_states * BYTES_PER_CELL_STATE
    return summary


def _save(path, state, labels):
    arrays = {'pressure': state.pressure, 'vx': state.u, 'vy': state.v, 'vz': state.w, 'tracer': state.tracer}
    np.savez_compressed(path, **{name: tensor.numpy().reshape(-1) for name, tensor in {**arrays, **labels}.items()})


def generate_scene(definition, row, directory):
    """Simulate one scene into `directory` (must not exist). Returns its bank record."""
    parameters = scene(definition, row)
    steps = parameters['steps']
    partial = directory.with_name(directory.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started = time.perf_counter()
    with RheonProbe(parameters, probe_for(parameters)) as probe:
        _, state = probe.state()
        zeros = {name: state.pressure.clone().zero_() for name in ('rhs', 'native_input_residual', 'pressure_base')}
        _save(partial / 'state-000.npz', state, zeros)
        for step in range(1, steps + 1):
            report, labels = probe.advance_reference()
            if not report['advanced'] or report['step'] != step:
                raise RuntimeError(f"The solver rejected step {step} of {row['id']}: {report.get('quality')}")
            _save(partial / f'state-{step:03d}.npz', probe.state()[1], labels)
    receipt = {'identifier': row['id'], 'split': row['split'], 'group': row['group'], 'dataset': definition['id'],
               'solver': definition['solver'], 'pressure_solve': 'exact cosine transform on the solver\'s requests',
               'scene': parameters, 'states': steps + 1, 'elapsed_seconds': time.perf_counter() - started}
    (partial / 'receipt.json').write_text(json.dumps(receipt) + '\n')
    partial.rename(directory)
    record = {'identifier': row['id'], 'split': row['split'], 'counts_xyz': row['counts'],
              'cells_per_metre': row['cells_per_metre'], 'states': steps + 1, 'receipt': f"{row['id']}/receipt.json"}
    for name in ('group', 'profile', 'motion'):
        if name in row:
            record[name] = row[name]
    return record


def generate(name, root=None, splits=None, scenes=None, allow_reserved=False, report=print):
    """Generate the chosen scenes of dataset `name` that are not on disk yet; returns the bank path."""
    definition = load(name)
    rows = select(definition, splits, scenes)
    if any(row['split'] == 'reserved' for row in rows) and not allow_reserved:
        raise ValueError('This selection includes reserved scenes. Reserved scenes are for final checks only; '
                         'pass allow_reserved to generate them on purpose.')
    directory = data_root(root) / definition['id']
    directory.mkdir(parents=True, exist_ok=True)
    bank = directory / 'bank.json'
    records = {}
    if bank.is_file():
        records = {record['identifier']: record for record in json.loads(bank.read_text())['trajectories']}
    for row in rows:
        if (directory / row['id'] / 'receipt.json').is_file() and row['id'] in records:
            continue
        if row.get('solver_substeps', 1) != 1:
            report(f"skipped {row['id']}: needs {row['solver_substeps']} solver substeps per output, not wired yet")
            continue
        if (directory / row['id']).exists():
            shutil.rmtree(directory / row['id'])       # a whole scene the index never recorded: redo it
        started = time.perf_counter()
        records[row['id']] = generate_scene(definition, row, directory / row['id'])
        document = {'dataset': definition['id'], 'paths_relative_to': 'bank',
                    'trajectories': [records[r['id']] for r in definition['scenes'] if r['id'] in records]}
        temporary = bank.with_suffix('.json.partial')
        temporary.write_text(json.dumps(document, indent=1) + '\n')
        temporary.replace(bank)
        report(f"generated {row['id']} ({records[row['id']]['states']} states, {time.perf_counter() - started:.1f} s)")
    return bank
