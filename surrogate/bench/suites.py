"""The named benchmark suites: which scenes of which dataset are scored, from where, for how long.

A suite reads a dataset's fields from the data root (see surrogate/datasets), so
the dataset has to be generated, or linked from existing recordings, first.
`population` says how the cases relate to model fitting, in Tuldok's vocabulary:
"consumed" cases have already informed model or checkpoint choices, so a score on
them is not evidence of generalisation.
"""
from .. import data, datasets
from ..datasets.store import bank_path
from .runner import Case, Suite

SPECIFICATIONS = {
    'dense-development-240': {
        'label': 'Sinusoidal development scenes · 240-output rollouts',
        'population': 'consumed',
        'description': 'The eight development scenes of sinusoidal-v1 (four grid shapes at 8 and 16 cells per '
                       'metre), each rolled out on its own outputs for 240 output intervals (4 s) from rest.',
        'dataset': 'sinusoidal-v1', 'split': 'development',
        'starts': [0], 'intervals': 240, 'velocity_error_limit': 0.1},
    'diversity-development-windows': {
        'label': 'Diversity development scenes · 4-output windows',
        'population': 'consumed',
        'description': 'The 24 development scenes of diversity-v1 (six densities from 8 to 100 cells per metre, '
                       'scheduled forcing), four outputs from each of four starts. The windows are short enough '
                       'that the earlier recordings of these scenes cover them.',
        'dataset': 'diversity-v1', 'split': 'development',
        'starts': [0, 32, 96, 192], 'intervals': 4, 'velocity_error_limit': 0.1},
}


def case(trajectory, start, intervals):
    z, y, x = trajectory.grid
    return Case(
        id=f'{trajectory.id}.start-{start}',
        reference=trajectory.states(start, intervals + 1),
        forcing=trajectory.forcing(start, intervals),
        scene=trajectory.scene_constants(),
        start=start,
        parameters=trajectory.scene,
        right_hand_sides=trajectory.right_hand_sides(start, intervals + 1),
        attributes={'scene': trajectory.id, 'split': trajectory.split, 'start': start, 'intervals': intervals,
                    'grid_x': x, 'grid_y': y, 'grid_z': z,
                    'cells_per_metre': trajectory.scene['cells_per_metre'], **trajectory.attributes})


def build(name, root=None):
    """The named suite. Cases load one at a time as the runner asks for them."""
    if name not in SPECIFICATIONS:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(sorted(SPECIFICATIONS))}.")
    specification = SPECIFICATIONS[name]
    definition = datasets.load(specification['dataset'])
    bank = bank_path(definition, root)
    wanted = [row['id'] for row in definition['scenes'] if row['split'] == specification['split']]
    on_disk = {t.id: t for t in data.load_bank(bank)} if bank.is_file() else {}
    intervals = specification['intervals']
    ready = [identity for identity in wanted
             if identity in on_disk and set(specification['starts']) <= set(on_disk[identity].starts(intervals))]
    if len(ready) != len(wanted):
        raise FileNotFoundError(
            f"Suite {name} needs all {len(wanted)} {specification['split']} scenes of {definition['id']} under "
            f"{bank.parent}, and {len(ready)} are usable. Generate them, or link existing recordings, with "
            f'python -m surrogate.datasets (see surrogate/README.md).')

    def cases():
        for identity in wanted:
            for start in specification['starts']:
                yield case(on_disk[identity], start, intervals)

    return Suite(id=name, label=specification['label'], population=specification['population'],
                 description=specification['description'],
                 velocity_error_limit=specification['velocity_error_limit'], cases=cases())
