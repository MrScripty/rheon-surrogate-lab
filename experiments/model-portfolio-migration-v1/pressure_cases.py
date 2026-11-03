"""Model-independent decoding of the frozen consumed native pressure requests.

Training readers have different packet schemas. Benchmark requests instead bind
the original packet bytes, array/frame identities and explicit physical metadata.
No model normalization, prediction or reference generation occurs here.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'runs/water-early-pressure-v1/control/project/inputs'
FIELDS = ('base', 'native_input_residual', 'reference', 'rhs')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load(torch, np):
    manifest = json.loads((DIRECTORY / 'validation_splits.json').read_text())
    if manifest['schema'] != 'rheon-water-learning-view-v1' or manifest['population'] != 'validation':
        raise ValueError('Frozen consumed native validation packet required')
    arrays = {}
    for name, shard in zip(('validation_dataset.npz', 'validation_dataset_extra.npz'), manifest['shards'], strict=True):
        path = DIRECTORY / name
        if path.stat().st_size != shard['bytes'] or digest(path.read_bytes()) != shard['sha256']:
            raise ValueError('Native pressure packet identity changed')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(shard['keys']):
                raise ValueError('Native pressure array inventory changed')
            for key in archive.files:
                value, spec = archive[key], manifest['arrays'][key]
                if (key in arrays or value.dtype.str != spec['dtype'] or list(value.shape) != spec['shape']
                        or value.nbytes != spec['bytes'] or digest(value.tobytes()) != spec['sha256']
                        or not bool(np.isfinite(value).all())):
                    raise ValueError('Native pressure array contract differs: ' + key)
                arrays[key] = value
    if set(arrays) != set(manifest['arrays']):
        raise ValueError('Incomplete native pressure packet')
    rows = []
    for index, parent in enumerate(manifest['parents']):
        if parent['entry']['split'] != 'validation':
            raise ValueError('Training/reserved case cannot enter consumed validation bank')
        for slot, frame in enumerate(parent['frames']):
            sample = {}
            for field in FIELDS:
                value = arrays[f'p{index:04d}_{field}'][slot]
                if (value.dtype.str != '<f8' or list(value.shape) != parent['counts_xyz'][::-1]
                        or digest(value.tobytes()) != frame['sha256'][field]):
                    raise ValueError('Original native pressure frame identity changed')
                sample[field] = torch.from_numpy(np.ascontiguousarray(value).copy())[None, None]
            for name, value in (('cell_size_m', parent['cell_size_m']),
                                ('rho', parent['density_kg_m3']), ('dt', parent['entry']['dt_s'])):
                if not 0 < value < float('inf'):
                    raise ValueError('Positive finite physical metadata required')
                sample[name] = torch.tensor([value], dtype=torch.float64)
            identity = dict(trajectory_id=parent['entry']['trajectory_id'], parent=index,
                            native_frame=frame['index'], counts_xyz=parent['counts_xyz'],
                            cell_size_m=parent['cell_size_m'])
            rows.append((identity, sample))
    if len(rows) != 128:
        raise ValueError('Complete original128 population required')
    return rows
