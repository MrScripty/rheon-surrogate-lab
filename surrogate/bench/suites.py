"""The named benchmark suites: which recorded trajectories are scored, from where, for how long.

Every suite reads reference data that already exists under runs/ (not in Git).
`population` says how the cases relate to model fitting, in Tuldok's vocabulary:
"consumed" cases have already informed model or checkpoint choices, so a score on
them is not evidence of generalisation.
"""
from .. import data
from .runner import Case, Suite

SPECIFICATIONS = {
    'dense-development-240': {
        'label': 'Dense development scenes · 240-output rollouts',
        'population': 'consumed',
        'description': 'The eight development scenes of the dense temporal bank (four grid shapes at 8 and '
                       '16 cells per metre, sinusoidal forcing), each rolled out on its own outputs for 240 '
                       'output intervals (4 s) from rest.',
        'banks': ['runs/full-ai-next-priorities-v1/temporal-development-240-01/bank.json'],
        'starts': [0], 'intervals': 240, 'velocity_error_limit': 0.1},
    'diversity-development-windows': {
        'label': 'Diversity development scenes · 4-output windows',
        'population': 'consumed',
        'description': 'The 24 development scenes of the expanded diversity bank (six densities from 8 to 100 '
                       'cells per metre, scheduled forcing), four outputs from each of four starts. The bank '
                       'keeps only short windows, so this suite cannot measure long-rollout stability.',
        'banks': ['runs/full-ai-data-diversity-v1/verified-development-windows-01.json'],
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


def build(name, repository=data.REPOSITORY):
    """The named suite. Cases load one at a time as the runner asks for them."""
    if name not in SPECIFICATIONS:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(sorted(SPECIFICATIONS))}.")
    specification = SPECIFICATIONS[name]
    trajectories = data.load_banks(specification['banks'], repository)

    def cases():
        for trajectory in trajectories:
            available = set(trajectory.starts(specification['intervals']))
            for start in specification['starts']:
                if start not in available:
                    raise ValueError(f"{trajectory.id} has no {specification['intervals']} consecutive saved "
                                     f'states after step {start}.')
                yield case(trajectory, start, specification['intervals'])

    return Suite(id=name, label=specification['label'], population=specification['population'],
                 description=specification['description'],
                 velocity_error_limit=specification['velocity_error_limit'], cases=cases())
