"""Small immutable experiment inputs and explicit managed-runtime selection."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def environment(runtime, device='cpu'):
    result = {**os.environ, 'PYTHONPATH': runtime['site_packages'], 'PYTHONDONTWRITEBYTECODE': '1',
              'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'RAYON_NUM_THREADS': '1', 'TULDOK_TORCH_DEVICE': device}
    if device == 'cpu':
        result['CUDA_VISIBLE_DEVICES'] = ''
    return result


def source_inputs(project):
    witness = read(project / 'source-witness.json')
    document = read(project / 'project.json')
    refs = {r['id']: r for r in document['data']}
    result = {}
    for slot in ['dataset', 'dataset_extra', 'splits']:
        path = ROOT / refs[slot]['path']
        if sha(path) != witness['inputs'][slot]['sha256'] or sha(path) != refs[slot]['sha256']:
            raise ValueError('Captured input changed: ' + slot)
        result[slot] = path
    return result
