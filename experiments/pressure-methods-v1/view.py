"""Bounded versioned reader for lossless mixed-domain training views."""
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

FIELDS = ('base', 'rhs', 'reference', 'native_input_residual')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def decode(raw_shards, manifest):
    if (manifest.get('schema') != 'rheon-pressure-methods-learning-view-v1'
            or manifest.get('status') != 'lossless_view_verified'
            or manifest.get('population') != 'train_only' or manifest.get('max_cells') != 262144
            or len(raw_shards) != 2 or len(manifest['shards']) != 2):
        raise ValueError('Versioned train-only mixed-domain view required')
    inventory = manifest['arrays']
    parents = manifest['parents']
    if not 1 <= len(parents) <= 256 or len(inventory) != 4*len(parents):
        raise ValueError('Bounded complete parent inventory required')
    projected = 0
    for index, parent in enumerate(parents):
        counts = parent['counts_xyz']
        if (parent['split'] != 'train' or len(counts) != 3
                or any(type(n) is not int or not 1 <= n <= 64 for n in counts)
                or math.prod(counts) > 262144 or not 1 <= len(parent['frames']) <= 2):
            raise ValueError('Training parent grid/frame budget failed')
        for name in ('cell_size_m', 'density_kg_m3', 'dt_s'):
            value = parent[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite physical metadata required')
        frame_indices = [f['index'] for f in parent['frames']]
        if frame_indices != sorted(set(frame_indices)):
            raise ValueError('Unique ordered frame indices required')
        for frame in parent['frames']:
            if frame['time_s'] != (frame['index']+1)*parent['dt_s']:
                raise ValueError('Physical frame time mismatch')
        expected_shape = [len(parent['frames']), *counts[::-1]]
        expected_bytes = math.prod(expected_shape)*8
        for field in FIELDS:
            spec = inventory.get(f'p{index:04d}_{field}')
            if (spec is None or spec['shape'] != expected_shape or spec['dtype'] != '<f8'
                    or spec['bytes'] != expected_bytes):
                raise ValueError('Parent array declaration differs from physical shape')
            projected += expected_bytes
    if projected > 64*1024**2:
        raise ValueError('64 MiB total decoded field budget exceeded')
    arrays = {}
    for payload, shard in zip(raw_shards, manifest['shards']):
        if (len(payload) > 32*1024**2 or len(payload) != shard['bytes'] or digest(payload) != shard['sha256']
                or set(arrays).intersection(shard['keys'])):
            raise ValueError('Shard size/hash/duplicate-key mismatch')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [item.filename for item in archive.infolist()]
            if len(set(names)) != len(names) or set(names) != {k+'.npy' for k in shard['keys']}:
                raise ValueError('Shard members differ from declared arrays')
            for info in archive.infolist():
                key = info.filename[:-4]
                spec = inventory.get(key)
                if spec is None or info.file_size > spec['bytes']+1024:
                    raise ValueError('Decoded array exceeds declared bound')
        with np.load(io.BytesIO(payload), allow_pickle=False) as shard_arrays:
            for key in shard['keys']:
                array = shard_arrays[key]
                spec = inventory[key]
                if (array.dtype.str != '<f8' or list(array.shape) != spec['shape']
                        or array.nbytes != spec['bytes'] or digest(array.tobytes()) != spec['sha256']
                        or not bool(np.isfinite(array).all())):
                    raise ValueError('Decoded field identity/dtype/finite mismatch')
                arrays[key] = array
    if set(arrays) != set(inventory):
        raise ValueError('Incomplete numeric population')
    identities = set()
    result = []
    for index, parent in enumerate(parents):
        if parent['trajectory_id'] in identities:
            raise ValueError('Duplicate whole-scene training identity')
        identities.add(parent['trajectory_id'])
        fields = {name:arrays[f'p{index:04d}_{name}'] for name in FIELDS}
        if not all(bool((fields[name][:, 0, 0, 0] == 0).all()) for name in ('base', 'reference')):
            raise ValueError('Changed pressure gauge')
        result.append(dict(**parent, entry=dict(trajectory_id=parent['trajectory_id'], split=parent['split'], dt_s=parent['dt_s']),
                           arrays=fields))
    return result


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    payloads = []
    for shard in manifest['shards']:
        path = directory/shard['path']
        if path.stat().st_size > 32*1024**2:
            raise ValueError('Encoded shard exceeds input budget')
        payloads.append(path.read_bytes())
    return manifest, decode(tuple(payloads), manifest)


def sample(parent, index, torch, device='cpu'):
    result = {name:torch.from_numpy(parent['arrays'][name][index:index+1].copy()).unsqueeze(1) for name in FIELDS}
    result.update(cell_size_m=torch.tensor([parent['cell_size_m']], dtype=torch.float64),
        rho=torch.tensor([parent['density_kg_m3']], dtype=torch.float64), dt=torch.tensor([parent['dt_s']], dtype=torch.float64))
    return {name:value.to(device) for name, value in result.items()}
