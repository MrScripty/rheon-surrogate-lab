"""Use recordings that already exist on this machine as a dataset, without copying or re-simulating them.

The lab's earlier studies recorded many of the scenes the dataset definitions
describe, under runs/. `link` reads the old index files, checks that each
recording was simulated from exactly the scene the definition builds, and puts
symbolic links to its state files at the dataset's place under the data root.
A scene whose every step is recorded counts as complete; a scene with only some
steps is usable for what it has and is replaced when it is generated in full.
"""
import json
import os
import shutil

from .. import data
from . import load, scene
from .store import data_root, is_complete, read_records, record, steps_of, write_records


def _same_scene(expected, recorded):
    """True if a recording's scene has every value the definition's scene has (its step count may differ)."""
    for key, value in expected.items():
        if key != 'steps' and recorded.get(key, 1 if key == 'reference_substeps' else None) != value:
            return False
    return recorded.get('reference_substeps', 1) == expected.get('reference_substeps', 1)


def link(name, banks, root=None, repository=data.REPOSITORY, report=print):
    """Link the recordings listed by the old index files `banks` into dataset `name`; returns the index path."""
    definition = load(name)
    rows = {row['id']: row for row in definition['scenes']}
    directory = data_root(root) / definition['id']
    records = read_records(definition, root)
    # The same scene can be listed by several old indexes; the longest recording wins.
    recordings = {}
    for bank in banks:
        for trajectory in data.load_bank(bank, repository):
            if trajectory.id in rows and len(trajectory.paths) > len(getattr(recordings.get(trajectory.id), 'paths', ())):
                recordings[trajectory.id] = trajectory
    linked = skipped = 0
    for identity, trajectory in recordings.items():
        row = rows[identity]
        steps = steps_of(definition, row)
        if is_complete(records.get(identity), steps):
            skipped += 1
            continue
        expected = scene(definition, row)
        if not _same_scene(expected, trajectory.scene):
            raise ValueError(f'{identity}: the recording was simulated from a different scene than the definition builds.')
        saved = [step for step, path in sorted(trajectory.paths.items()) if step <= steps and path.is_file()]
        target = directory / identity
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for step in saved:
            os.symlink(trajectory.paths[step].resolve(), target / f'state-{step:03d}.npz')
        receipt = {'identifier': identity, 'split': row['split'], 'group': row['group'], 'dataset': definition['id'],
                   'origin': 'linked', 'solver': definition['solver'], 'scene': expected, 'states': len(saved)}
        (target / 'receipt.json').write_text(json.dumps(receipt) + '\n')
        records[identity] = record(row, saved)
        linked += 1
    bank = write_records(definition, records, root)
    complete = sum(is_complete(records.get(row['id']), steps_of(definition, row)) for row in definition['scenes'])
    report(f"{definition['id']}: linked {linked} scenes, {skipped} already complete; "
           f"{complete} of {len(rows)} scenes are complete on disk, {len(records) - complete} have some steps")
    return bank
