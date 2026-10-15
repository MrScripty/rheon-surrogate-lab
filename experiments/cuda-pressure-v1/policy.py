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


def frozen_model():
    path = HERE/'frozen_gn.py'
    if not path.exists():
        path = LAB/'experiments/global-pressure-v1/groupnorm-v1/model.py'
    return load_module('variable_pressure_frozen_gn', path)


def experiment(value):
    require(type(value) is dict and set(value) == {'schema','id','training','resources'}
            and value['schema']=='rheon-cuda-pressure-experiment-v1', 'experiment', 'closed CUDA training configuration')
    require(type(value['id']) is str and re.fullmatch('[a-z][a-z0-9-]{0,63}',value['id']), 'id','safe identifier')
    training=value['training']
    require(type(training) is dict and set(training)=={'model','seed','sampling_seed','updates','batch_size','learning_rate','checkpoints','schedule'}
            and training['model']=='cubic-groupnorm-v3','training','unchanged seven-channel model')
    for name in ('seed','sampling_seed'):
        require(type(training[name]) is int and 0<=training[name]<2**63,'seed',name)
    for name,maximum in [('updates',10000),('batch_size',4)]:
        require(type(training[name]) is int and 1<=training[name]<=maximum,'training_budget',name)
    checkpoints=training['checkpoints']
    require(type(checkpoints) is list and 2<=len(checkpoints)<=11 and all(type(n) is int for n in checkpoints)
            and checkpoints==sorted(set(checkpoints)) and checkpoints[0]==0 and checkpoints[-1]==training['updates'],
            'checkpoints','2–11 ordered states, including initialization and final update')
    require(type(training['learning_rate']) in (int,float) and 0<training['learning_rate']<=.001,'learning_rate','bounded Adam rate')
    schedule=training['schedule'];last=0
    require(type(schedule) is list and len(schedule)<=8,'schedule','bounded explicit learning rate schedule')
    for item in schedule:
        require(type(item) is dict and set(item)=={'update','learning_rate'}
                and type(item['update']) is int and last<item['update']<=training['updates']
                and type(item['learning_rate']) in (int,float) and 0<item['learning_rate']<=.001,'schedule','ordered positive rates')
        last=item['update']
    resources=value['resources']
    require(type(resources) is dict and set(resources)=={'wall_seconds','max_output_bytes'}
            and type(resources['wall_seconds']) is int and 30<=resources['wall_seconds']<=600
            and type(resources['max_output_bytes']) is int and 0<resources['max_output_bytes']<=32*MIB,
            'resources','bounded elapsed runtime and complete artifact budget')
    return decode(canonical(value))
