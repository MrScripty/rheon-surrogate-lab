"""Lossless heterogeneous whole-parent packets; each parent retains its grid."""
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from policy import HERE, LAB, MIB, canonical, cubic_spacing, decode, dimensions, experiment, fresh, load_module, read, require, sha, write
import model

FIELDS = ('base', 'reference', 'rhs', 'native_input_residual')


def helpers():
    return model.frozen.load_original('train', model.frozen.ORIGINAL_TRAIN_SHA256)


def inventory(arrays):
    return {name: {'shape': list(value.shape), 'dtype': value.dtype.str, 'bytes': value.nbytes,
                   'sha256': sha(value.tobytes())} for name, value in arrays.items()}


def load(raw, manifest):
    require(manifest['schema'] == 'rheon-variable-pressure-packet-v2' and sha(raw) == manifest['dataset_sha256'], 'packet', 'selected lossless packet')
    config = experiment(manifest['experiment'])
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


def sample(parent, index, labels=True):
    fields = FIELDS if labels else ('base', 'native_input_residual')
    result = {name: torch.from_numpy(parent['arrays'][name][index:index+1].copy()).unsqueeze(1) for name in fields}
    result.update(cell_size_m=torch.tensor([parent['cell_size_m']], dtype=torch.float64),
                  rho=torch.tensor([1.25], dtype=torch.float64),
                  dt=torch.tensor([parent['entry']['dt_s']], dtype=torch.float64))
    return result


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


def pack(config, captures, binary, output, observer=None):
    import build_native
    config = experiment(config)
    require(len(captures) == len(config['captures']), 'captures', 'every selected density')
    native = build_native.admit(binary)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(LAB/'experiments/global-pressure-v1'))
        oracle = load_module('variable_packet_independent_oracle', LAB/'experiments/global-pressure-v1/verify_capture.py')
    finally:
        sys.path[:] = original_path
    arrays, parents, reports = {}, [], []
    for capture_config, root in zip(config['captures'], captures):
        result = subprocess.run([sys.executable, str(LAB/'experiments/cubic-pressure-v1/pipeline.py'), 'verify', '--capture', str(root)], capture_output=True, timeout=180, check=True)
        report = decode(result.stdout)
        require(report['status'] == 'verified_development' and report['experiment_sha256'] == sha(canonical(capture_config)), 'capture_identity', 'qualified selected development configuration')
        reports.append(report)
        origin = decode(read(root/'manifest.json', 4*MIB))
        for trajectory in origin['trajectories']:
            index = len(parents); geometry = trajectory['geometry']; counts = geometry['counts_xyz']
            h = 1/capture_config['cells_per_metre']; cubic_spacing([h]*3)
            fields = {name: np.empty((24, *counts[::-1]), dtype='<f8') for name in FIELDS}
            frames = []
            for frame in trajectory['frames']:
                number = frame['index']; descriptors = {name: frame['arrays'][source] for name, source in [('reference', 'pressure'), ('rhs', 'rhs')]}
                previous = next(d for d in [trajectory['initial']['pressure'], *[f['arrays']['pressure'] for f in trajectory['frames']]] if d['path'] == frame['previous_pressure'])
                descriptors['base'] = previous
                for field, descriptor in descriptors.items():
                    raw = read(root/descriptor['path'], MIB)
                    require(sha(raw) == descriptor['sha256'], 'capture_payload', 'immutable reference bytes')
                    fields[field][number] = np.frombuffer(raw, dtype='<f8').reshape(counts[::-1])
                request = {'schema': 'rheon-variable-pressure-native-residual-v1', 'counts_xyz': counts, 'spacing_m': [h]*3,
                           'density_kg_m3': 1.25, 'original_rhs': fields['rhs'][number].reshape(-1).tolist(), 'base_pressure': fields['base'][number].reshape(-1).tolist()}
                response, _ = build_native.call(binary, '--residual', request, timeout=10)
                require(response['identity'] == native['identity'], 'residual_identity', 'selected native operator')
                residual = np.asarray(response['residual'], dtype='<f8')
                require(residual.shape == (math.prod(counts),) and np.isfinite(residual).all(), 'residual', 'full finite native residual')
                action = oracle.full_action(request['base_pressure'], counts, [h]*3, 1.25)
                require(all(abs(r - (b-a)) <= 4e-14*max(abs(b), abs(a), 1e-3) for r,b,a in zip(residual, request['original_rhs'], action)), 'residual_oracle', 'independent face-incidence full action')
                fields['native_input_residual'][number] = residual.reshape(counts[::-1])
                frames.append({'index': number, 'sha256': {field: sha(values[number].tobytes()) for field, values in fields.items()}})
                if observer and number % 8 == 0:
                    total = sum(len(c['experiment']['trajectories'])*24 for c in config['captures'])
                    observer.publish('Preparing variable-grid training', f'Qualified native residuals: {counts} cells, h={h} m', index*24+number+1, total)
            for name, values in fields.items():
                arrays[f'p{index:03d}_{name}'] = values
            parents.append({'entry': trajectory['reservation_entry'], 'counts_xyz': counts, 'cell_size_m': h,
                            'capture_root': str(root), 'capture_manifest_sha256': report['manifest_sha256'], 'frames': frames})
    output = fresh(output)
    with (output/'dataset.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    raw = read(output/'dataset.npz', 32*MIB)
    manifest = {'schema': 'rheon-variable-pressure-packet-v2', 'experiment': config, 'parents': parents,
                'dataset_sha256': sha(raw), 'arrays': inventory(arrays), 'native_build_receipt': native,
                'verified_capture_reports': reports}
    write(output/'splits.json', manifest)
    load(raw, manifest)
    require(build_native.admit(binary) == native, 'native_drift', 'unchanged native operator')
    write(output/'dataset-receipt.json', {'status': 'verified_lossless_development', 'dataset_sha256': sha(raw),
          'manifest_sha256': sha(canonical(manifest)), 'reader_source_sha256': sha(read(HERE/'data.py')),
          'train_frames': sum(24 for p in parents if p['entry']['split'] == 'train'),
          'validation_frames': sum(24 for p in parents if p['entry']['split'] == 'validation')})
    return manifest
