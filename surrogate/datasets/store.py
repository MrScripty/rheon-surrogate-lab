"""Where a dataset's fields live on disk, and the index that lists them.

    <root>/<dataset>/bank.json                 index that surrogate.data.load_bank reads
    <root>/<dataset>/<scene>/receipt.json      the scene's parameters and how its fields got here
    <root>/<dataset>/<scene>/state-NNN.npz     one file per saved output step

The root defaults to `data/reference/` in this repository, which Git ignores.
A scene is complete when every step from 0 to its last is saved.
"""
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPOSITORY / 'data' / 'reference'


def data_root(root=None):
    """The data root: the argument, else $SURROGATE_DATA, else data/reference in the repository."""
    return Path(root or os.environ.get('SURROGATE_DATA') or DEFAULT_ROOT)


def bank_path(definition, root=None):
    return data_root(root) / definition['id'] / 'bank.json'


def steps_of(definition, row):
    return row.get('steps', definition.get('steps'))


def record(row, saved_steps):
    """The index entry of one scene whose `saved_steps` are on disk as <scene>/state-NNN.npz."""
    saved_steps = sorted(saved_steps)
    every_step = saved_steps == list(range(len(saved_steps)))
    entry = {'identifier': row['id'], 'split': row['split'], 'counts_xyz': row['counts'],
             'cells_per_metre': row['cells_per_metre'], 'receipt': f"{row['id']}/receipt.json",
             'states': len(saved_steps) if every_step else
             [{'step': step, 'path': f"{row['id']}/state-{step:03d}.npz"} for step in saved_steps]}
    for name in ('group', 'profile', 'motion'):
        if name in row:
            entry[name] = row[name]
    if row.get('solver_substeps', 1) != 1:
        entry['reference_substeps'] = row['solver_substeps']
    return entry


def is_complete(entry, steps):
    return entry is not None and entry['states'] == steps + 1


def read_records(definition, root=None):
    """Index entries by scene id; empty if nothing is on disk yet."""
    bank = bank_path(definition, root)
    if not bank.is_file():
        return {}
    return {entry['identifier']: entry for entry in json.loads(bank.read_text())['trajectories']}


def write_records(definition, records, root=None):
    """Replace the index atomically, keeping the definition's scene order."""
    bank = bank_path(definition, root)
    bank.parent.mkdir(parents=True, exist_ok=True)
    document = {'dataset': definition['id'], 'paths_relative_to': 'bank',
                'trajectories': [records[row['id']] for row in definition['scenes'] if row['id'] in records]}
    temporary = bank.with_suffix('.json.partial')
    temporary.write_text(json.dumps(document, indent=1) + '\n')
    temporary.replace(bank)
    return bank
