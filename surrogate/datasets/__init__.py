"""Reference datasets as definitions: which scenes to simulate, not the simulated fields.

A definition is a small JSON file in `definitions/`: a table of scenes with their
split and the few values each scene is built from. The fields themselves are
produced by running the solver (see generate.py) and are never stored in Git.

    {
      "id", "label", "description",
      "scene_family": "scheduled-forcing" | "sinusoidal-forcing",
      "solver": the reference solver the fields come from,
      "steps": output steps per scene (or per scene, in the scene row),
      "scenes": [{"id", "split", "group", ...the scene family's defining values}]
    }

Splits are assigned per group: scenes that share a forcing seed share a split.
"""
import json
from pathlib import Path

from ..scenes import scheduled_scene

DEFINITIONS = Path(__file__).resolve().parent / 'definitions'
SPLITS = ('train', 'development', 'reserved')
_SINUSOIDAL_KEYS = ('seed', 'counts', 'cells_per_metre', 'center', 'acceleration', 'radius', 'swirl', 'fluctuation',
                    'frequency')


def available():
    return sorted(path.stem for path in DEFINITIONS.glob('*.json'))


def load(name):
    """A dataset definition by name, checked for the rules stated above."""
    path = DEFINITIONS / f'{name}.json'
    if not path.is_file():
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(available())}.")
    definition = json.loads(path.read_text())
    identities = [row['id'] for row in definition['scenes']]
    if len(set(identities)) != len(identities):
        raise ValueError(f'{name}: scene ids must be unique.')
    split_of_group = {}
    for row in definition['scenes']:
        if row['split'] not in SPLITS:
            raise ValueError(f"{name}: scene {row['id']} has unknown split {row['split']!r}.")
        if split_of_group.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError(f"{name}: group {row['group']} appears in more than one split.")
    return definition


def scene(definition, row):
    """The full scene parameters (what the solver and forcing.py read) for one row of a definition."""
    steps = row.get('steps', definition.get('steps'))
    family = definition['scene_family']
    if family == 'scheduled-forcing':
        built = scheduled_scene(row['seed'], row['counts'], row['cells_per_metre'], row['profile'], row['motion'], steps)
    elif family == 'sinusoidal-forcing':
        built = {**{key: row[key] for key in _SINUSOIDAL_KEYS}, 'steps': steps, 'warmup_steps': 0, 'sgs': False}
    else:
        raise ValueError(f'Unknown scene family {family!r}.')
    if row.get('solver_substeps', 1) != 1:
        built['reference_substeps'] = row['solver_substeps']
    return built
