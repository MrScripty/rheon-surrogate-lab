"""Common paths and bounded, atomic evidence writes for the full AI study."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'runs/full-ai-next-priorities-v1'
DOCS = ROOT / 'docs/plans/full-ai-next-priorities-v1'
sys.path[:0] = [str(ROOT / 'experiments' / n) for n in (
    'pressure-four-pass-distillation-v1', 'pressure-pass-sweep-v1',
    'pressure-methods-v1', 'pressure-methods-v2', 'pressure-methods-v3',
    'pressure-methods-v4')]

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def runtime():
    return read(ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json')

def load_checkpoint(identifier):
    import inference
    arm = {'C106': 'v4-student50-distill', 'C107': 'v4-student50-control'}.get(identifier, identifier)
    entry_path = ROOT / 'runs/pressure-methods-v4/entries' / (arm + '.json')
    if identifier not in ('C106', 'C107'):
        entry_path = BASE / 'entries' / (identifier + '.json')
    entry = read(entry_path)
    if entry.get('loader') == 'routed-pressure-assembly-v1':
        if inference._CACHE is not None:
            raise ValueError('One checkpoint per numerical process required')
        import routed_inference
        return routed_inference.load(ROOT, entry, identifier)
    if 'routed_inference' in sys.modules and sys.modules['routed_inference']._CACHE is not None:
        raise ValueError('One checkpoint per numerical process required')
    material = read(ROOT / 'runs/pressure-pass-sweep-v1/cpu-01/material.json')
    return inference.load(dict(material, models=[entry]), entry['id'])

def historical_cases(state):
    inputs = ROOT / 'runs/water-early-pressure-v1/control/project/inputs'
    manifest = read(inputs / 'validation_splits.json')
    _, parents = state.data.load((
        (inputs / 'validation_dataset.npz').read_bytes(),
        (inputs / 'validation_dataset_extra.npz').read_bytes()), manifest)
    result = []
    for index, parent in enumerate(parents):
        for slot, frame in enumerate(parent['frames']):
            identity = dict(trajectory_id=parent['entry']['trajectory_id'], parent=index,
                            native_frame=frame['index'], counts_xyz=parent['counts_xyz'],
                            cell_size_m=parent['cell_size_m'])
            result.append((identity, state.data.sample(parent, slot)))
    if len(result) != 128:
        raise ValueError('Historical comparison must retain all 128 requests')
    return result

def equation_error(state, pressure, sample):
    torch = state.torch
    residual = state.model.full_operator(pressure, sample['cell_size_m'], sample['rho']) - sample['rhs']
    return float(torch.linalg.vector_norm(residual)) / max(float(torch.linalg.vector_norm(sample['rhs'])), 1e-30)

def source_identity():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))}
