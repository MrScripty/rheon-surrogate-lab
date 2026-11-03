"""Lossless mixed-domain training view; retain all original small-grid fields."""
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, source_inputs

FIELDS = ['base', 'rhs', 'reference', 'native_input_residual']


def pack():
    runs = ROOT/'runs/pressure-methods-v1'
    output = runs/'domain-training-view-01'
    if output.exists():
        raise ValueError('Fresh mixed-domain view required')
    source = ROOT/'runs/pressure-current-policy-v1/current-late-project-02'
    paths = source_inputs(source)
    original = read(paths['splits'])
    arrays = {}
    for name in ['dataset', 'dataset_extra']:
        with np.load(paths[name], allow_pickle=False) as shard:
            arrays.update({key:shard[key].copy() for key in shard.files})
    parents = []
    values = {}
    source_evidence = {str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    for index, row in enumerate(original['parents']):
        if row['entry']['split'] != 'train':
            raise ValueError('Original view must contain training parents only')
        parents.append(dict(trajectory_id=row['entry']['trajectory_id'], split='train',
            counts_xyz=row['counts_xyz'], cell_size_m=row['cell_size_m'], density_kg_m3=row['density_kg_m3'],
            dt_s=row['entry']['dt_s'], frames=[dict(index=f['index'], time_s=(f['index']+1)*row['entry']['dt_s']) for f in row['frames']],
            source='retained-small-grid-view', original_parent_index=index))
        for name in FIELDS:
            value = arrays[f'p{index:04d}_{name}']
            if value.dtype.str != '<f8' or not bool(np.isfinite(value).all()):
                raise ValueError('Original dtype/finite field changed')
            values[f'p{index:04d}_{name}'] = value
    retained_count = len(parents)
    reservation = read(runs/'native-01/reservation.json')
    for row in reservation['scenes']:
        if not row['trajectory_id'].startswith('large-'):
            continue
        directory = runs/'native-pilots/capture'/row['trajectory_id']
        receipt = read(directory/'receipt.json')
        if (receipt['status'] != 'qualified_pilot' or receipt['native_dt_s'] != 1/60
                or receipt['native_steps_qualified'] != 64 or not receipt['native_process']['reaped']):
            raise ValueError('Qualified full-interval native domain pilot required')
        source_evidence[str((directory/'receipt.json').relative_to(ROOT))] = sha(directory/'receipt.json')
        index = len(parents)
        parents.append(dict(trajectory_id=row['trajectory_id'], split='train', counts_xyz=row['counts_xyz'],
            cell_size_m=1/row['cells_per_metre'], density_kg_m3=1000., dt_s=1/60,
            frames=[dict(index=f['index'], time_s=f['time_s']) for f in receipt['frames']],
            source='qualified-native-domain-pilot', native_receipt=str((directory/'receipt.json').relative_to(ROOT))))
        for name in FIELDS:
            captured = 'pressure' if name == 'reference' else name
            frames = []
            for frame in receipt['frames']:
                field = frame['arrays'][captured]
                path = directory/field['path']
                if sha(path) != field['sha256'] or field['dtype'] != '<f8':
                    raise ValueError('Native field identity changed')
                frames.append(np.fromfile(path, dtype='<f8').reshape(row['counts_xyz'][::-1]))
            values[f'p{index:04d}_{name}'] = np.stack(frames)
    if len(parents) != retained_count+4:
        raise ValueError('All four new size/spacing combinations required')
    inventory = {k:dict(shape=list(v.shape), dtype=v.dtype.str, bytes=v.nbytes,
                        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in values.items()}
    for row in parents:
        if len(row['counts_xyz']) != 3 or np.prod(row['counts_xyz']) > 262144:
            raise ValueError('Mixed-domain cell bound')
    output.mkdir(parents=True)
    shards = []
    for number, fields in enumerate([['base', 'rhs'], ['reference', 'native_input_residual']]):
        selected = {k:v for k, v in values.items() if any(k.endswith('_'+f) for f in fields)}
        path = output/f'dataset-{number:02d}.npz'
        np.savez(path, **selected)
        if path.stat().st_size > 32*1024**2:
            raise ValueError('Explicit per-shard 32 MiB input budget exceeded')
        with np.load(path, allow_pickle=False) as restored:
            if set(restored.files) != set(selected) or any(not np.array_equal(restored[k], selected[k]) for k in selected):
                raise ValueError('Lossless training-view roundtrip failed')
        shards.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path), keys=sorted(selected)))
    manifest = dict(schema='rheon-pressure-methods-learning-view-v1', population='train_only',
        status='lossless_view_verified', parents=parents, arrays=inventory, shards=shards,
        max_cells=262144, retained_small_parents=retained_count, added_larger_parents=4,
        source_evidence=source_evidence, source_sha256=sha(__file__), config_sha256=sha(runs/'config.json'),
        no_legacy_reader_compatibility_claimed=True, model_training_completed=False)
    write(output/'manifest.json', manifest)
    write(ROOT/'docs/plans/pressure-methods-v1/domain-training-view.json', dict(
        status=manifest['status'], manifest=str((output/'manifest.json').relative_to(ROOT)), manifest_sha256=sha(output/'manifest.json'),
        retained_small_parents=retained_count, added_larger_parents=4, snapshots=sum(len(p['frames']) for p in parents),
        spacings_cells_per_metre=sorted({1/p['cell_size_m'] for p in parents}),
        shard_bytes=[s['bytes'] for s in shards], source_sha256=sha(__file__),
        pending='Versioned training reader, model shape-budget qualification, matched fits and evaluation.'))
    print('Packed', len(parents), 'training parents into two losslessly verified shards', flush=True)


if __name__ == '__main__':
    pack()
