"""Configuration, source selection and one physical spacing for every cell axis."""
import math
from pathlib import Path
import re

from support import canonical, decode, fresh, load_module, read, require, sha, write, MIB

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
MAX_CELLS = 65536


def dimensions(extents, density):
    require(type(density) is int and 1 <= density <= 128, 'density', 'integer cells/metre, 1–128')
    require(type(extents) is list and len(extents) == 3, 'extents', 'three lengths in metres')
    counts = []
    for length in extents:
        require(type(length) in (int, float) and math.isfinite(length) and length > 0,
                'extent', 'finite positive physical length')
        exact = length * density
        count = round(exact)
        require(abs(exact - count) <= 1e-10 and 2 <= count <= 128,
                'cell_multiple', 'domain lengths must be whole cell multiples, 2–128 per axis')
        counts.append(count)
    require(math.prod(counts) <= MAX_CELLS, 'cell_budget', 'at most 65,536 cells')
    return counts


def cubic_spacing(spacing):
    require(len(spacing) == 3 and all(type(h) in (int, float) and math.isfinite(h) and h > 0 for h in spacing)
            and spacing[0] == spacing[1] == spacing[2], 'noncubic_cells', 'one shared positive cell side length in metres')
    return spacing[0]


def experiment(value):
    from support import experiment as capture_experiment
    require(type(value) is dict and set(value) == {'schema', 'id', 'captures', 'training', 'resources'}
            and value['schema'] == 'rheon-variable-pressure-experiment-v1', 'experiment', 'closed variable-shape experiment')
    require(type(value['id']) is str and re.fullmatch('[a-z][a-z0-9-]{0,63}', value['id']), 'id', 'safe identifier')
    require(type(value['captures']) is list and 1 <= len(value['captures']) <= 4, 'captures', '1–4 complete capture profiles')
    seeds, ids = set(), set()
    for capture in value['captures']:
        require(type(capture) is dict and set(capture) == {'schema', 'cells_per_metre', 'experiment'}
                and capture['schema'] == 'rheon-cubic-pressure-v1', 'capture', 'existing cubic capture profile')
        original = capture_experiment(capture['experiment'])
        for lengths in original['geometries_extents_m'].values():
            dimensions(lengths, capture['cells_per_metre'])
        for entry in original['trajectories']:
            require(entry['seed'] not in seeds and entry['trajectory_id'] not in ids, 'parent_overlap', 'unique whole parents across densities')
            require(entry['dt_s'] == 1 / 60, 'native_dt', 'qualified development timestep 1/60 s')
            seeds.add(entry['seed']); ids.add(entry['trajectory_id'])
    training = value['training']
    require(type(training) is dict and set(training) == {'model', 'seed', 'updates', 'batch_size', 'learning_rate', 'checkpoint_interval'}
            and training['model'] == 'cubic-groupnorm-v2', 'training', 'variable-shape GroupNorm profile')
    require(type(training['seed']) is int and 0 <= training['seed'] < 2**63, 'seed', 'integer initialization seed')
    for name, high in [('updates', 200), ('batch_size', 4), ('checkpoint_interval', 200)]:
        require(type(training[name]) is int and 1 <= training[name] <= high, 'training_budget', name)
    require(training['updates'] % training['checkpoint_interval'] == 0, 'checkpoint', 'include final update')
    require(type(training['learning_rate']) in (int, float) and 0 < training['learning_rate'] <= .001, 'learning_rate', 'positive bounded Adam rate')
    require(type(value['resources']) is dict and set(value['resources']) == {'wall_seconds', 'max_output_bytes'}
            and type(value['resources']['wall_seconds']) is int and 30 <= value['resources']['wall_seconds'] <= 600
            and type(value['resources']['max_output_bytes']) is int and 0 < value['resources']['max_output_bytes'] <= 32 * MIB,
            'resources', 'bounded CPU execution and output')
    return decode(canonical(value))


def frozen_model():
    path = HERE/'frozen_gn.py'
    if not path.exists():
        path = LAB/'experiments/global-pressure-v1/groupnorm-v1/model.py'
    return load_module('variable_pressure_frozen_gn', path)
