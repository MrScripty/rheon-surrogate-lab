"""Lossless heterogeneous whole-parent packets; each parent retains its grid."""
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from policy import HERE, LAB, MIB, canonical, cubic_spacing, decode, dimensions, fresh, load_module, read, require, sha, write
import model
packet_policy = load_module('cuda_pressure_packet_policy', HERE/'packet_policy.py')

FIELDS = ('base', 'reference', 'rhs', 'native_input_residual')


def helpers():
    return model.frozen.load_original('train', model.frozen.ORIGINAL_TRAIN_SHA256)


def inventory(arrays):
    return {name: {'shape': list(value.shape), 'dtype': value.dtype.str, 'bytes': value.nbytes,
                   'sha256': sha(value.tobytes())} for name, value in arrays.items()}


def load(raw, manifest):
    require(manifest['schema'] == 'rheon-variable-pressure-packet-v2' and sha(raw) == manifest['dataset_sha256'], 'packet', 'selected lossless packet')
    config = packet_policy.experiment(manifest['experiment'])
    expected = [(capture, entry) for capture in config['captures'] for entry in capture['experiment']['trajectories']]
    require(len(manifest['parents']) == len(expected), 'parents', 'complete whole-parent coverage')
    shapes = {}
    for index, (parent, (capture, entry)) in enumerate(zip(manifest['parents'], expected)):
        counts = dimensions(capture['experiment']['geometries_extents_m'][entry['geometry']], capture['cells_per_metre'])
        h = 1 / capture['cells_per_metre']
        require(parent['entry'] == entry and parent['counts_xyz'] == counts and parent['cell_size_m'] == h,
                'physical_lineage', 'actual per-axis counts and shared spacing')
        cubic_spacing([parent['cell_size_m']]*3)
        require(len(parent['frames']) == 24 and [f['index'] for f in parent['frames']] == list(range(24)), 'history', 'full ordered frames')
        for field in FIELDS:
            name = f'p{index:03d}_{field}'
            declaration = manifest['arrays'][name]
            require(declaration['shape'] == [24, *counts[::-1]] and declaration['dtype'] == '<f8'
                    and declaration['bytes'] == 24*math.prod(counts)*8, 'packet_shape', name)
            shapes[name] = declaration
    require(set(shapes) == set(manifest['arrays']), 'array_inventory', 'closed numeric fields')
    arrays = helpers().read_npz(raw, shapes, 32*MIB)
    parents = []
    for index, parent in enumerate(manifest['parents']):
        values = {field: arrays[f'p{index:03d}_{field}'] for field in FIELDS}
        require(bool((values['base'][:, 0, 0, 0] == 0).all() and (values['reference'][:, 0, 0, 0] == 0).all()), 'gauge', 'cell zero')
        for frame in parent['frames']:
            for field in FIELDS:
                require(sha(values[field][frame['index']].tobytes()) == frame['sha256'][field], 'raw_lineage', field)
        parents.append({**parent, 'arrays': values})
    return config, parents


def sample(parent, index, labels=True, device='cpu'):
    fields = FIELDS if labels else ('base', 'native_input_residual')
    result = {name: torch.from_numpy(parent['arrays'][name][index:index+1].copy()).unsqueeze(1) for name in fields}
    result.update(cell_size_m=torch.tensor([parent['cell_size_m']], dtype=torch.float64),
                  rho=torch.tensor([1.25], dtype=torch.float64),
                  dt=torch.tensor([parent['entry']['dt_s']], dtype=torch.float64))
    return {name: value.to(device) for name,value in result.items()}


def fit_normalization(parents):
    trains = [p for p in parents if p['entry']['split'] == 'train']
    require(trains, 'training_population', 'train-only parents')
    cells = sum(p['arrays']['base'].size for p in trains)
    scales = {}
    for name, field, units in [('residual', 'native_input_residual', 'm3/s2'), ('pressure', 'base', 'Pa'), ('rhs', 'rhs', 'm3/s2')]:
        rms = math.sqrt(math.fsum(float(np.square(p['arrays'][field]).sum(dtype=np.float64)) for p in trains)/cells)
        require(math.isfinite(rms) and rms > 0, 'normalization', 'positive train-cell RMS')
        scales[name] = {'value': rms, 'units': units, 'method': 'train_cell_rms', 'source': field}
    for name, values, units in [('rho', [1.25]*(len(trains)*24), 'kg/m3'),
                                ('dt', [p['entry']['dt_s'] for p in trains for _ in range(24)], 's')]:
        constant = all(v == values[0] for v in values)
        scales[name] = {'value': 1.0 if constant else math.sqrt(math.fsum(v*v for v in values)/len(values)),
                        'units': units, 'method': 'constant_unit_scale' if constant else 'train_frame_rms', 'source': name}
    return {'schema': 'rheon-cubic-pressure-normalization-v2', 'model_profile': model.ARCHITECTURE['schema'],
            'channels': model.CHANNELS, 'coordinates': 'actual cell centres in metres; unit scale 1 m', 'scales': scales,
            'training_provenance': {'trajectory_ids': [p['entry']['trajectory_id'] for p in trains],
                                    'frames': len(trains)*24, 'cells': cells,
                                    'grids': [p['counts_xyz'] for p in trains],
                                    'parent_numeric_sha256': [{field: sha(p['arrays'][field].tobytes()) for field in FIELDS} for p in trains]}}
