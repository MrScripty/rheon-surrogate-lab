"""Build paired temporal and spacing views from qualified native trajectories.

Fine-step pressures are rescaled to the model's unchanged pressure equation.
Temporal triplets share the coarse-history initial guess, avoiding an easier
warm start caused solely by a shorter previous timestep. These are requests
from finer teacher trajectories, not exact coarse transition targets.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha

view_spec = importlib.util.spec_from_file_location('pressure_methods_training_view', HERE / 'view.py')
view = importlib.util.module_from_spec(view_spec)
view_spec.loader.exec_module(view)

ORACLE = ROOT / 'runs/dependency-migration-v1/lab/experiments/global-pressure-v1/verify_capture.py'
spec = importlib.util.spec_from_file_location('training_view_oracle', ORACLE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
FIELDS = ('base', 'rhs', 'reference', 'native_input_residual')


def field(directory, frame, name):
    row = frame['arrays'][name]
    path = directory / row['path']
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Retained native field changed')
    return np.fromfile(path, dtype=row['dtype']).reshape(row['shape_xyz'][::-1])


def captured(row):
    directory = ROOT / 'runs/pressure-methods-v1/broader-native-data' / row['campaign_group'] / 'capture' / row['trajectory_id']
    receipt = read(directory / 'receipt.json')
    if (receipt['status'] != 'qualified_pilot' or receipt['fixture']
            or receipt['row'] != row or receipt['native_steps_qualified'] != 64 * row['teacher_substeps']
            or not receipt['native_process']['reaped'] or receipt['native_process']['returncode'] != 0):
        raise ValueError('Complete qualified native trajectory required')
    return directory, receipt


def emit(name, parents, values, evidence, checks, method):
    output = ROOT / 'runs/pressure-methods-v1' / (name + '-training-view-01')
    if output.exists():
        raise ValueError('Fresh view required')
    inventory = {k:dict(shape=list(v.shape), dtype=v.dtype.str, bytes=v.nbytes,
                        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}
    output.mkdir()
    shards = []
    for number, names in enumerate([('base','rhs'), ('reference','native_input_residual')]):
        selected = {k:v for k,v in values.items() if any(k.endswith('_' + f) for f in names)}
        path = output / f'dataset-{number:02d}.npz'
        np.savez(path, **selected)
        shards.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path), keys=sorted(selected)))
    manifest = dict(schema='rheon-pressure-methods-learning-view-v1', population='train_only',
        status='lossless_view_verified', parents=parents, arrays=inventory, shards=shards,
        max_cells=262144, source_evidence=evidence, source_sha256=sha(__file__),
        method=method, independent_equation_checks=checks, model_training_completed=False)
    # The exact bounded production reader must decode the just-written bytes.
    decoded = view.decode(tuple((output / s['path']).read_bytes() for s in shards), manifest)
    if len(decoded) != len(parents) or any(not np.array_equal(
            decoded[i]['arrays'][f], values[f'p{i:04d}_{f}']) for i in range(len(parents)) for f in FIELDS):
        raise ValueError('Actual training-reader lossless roundtrip failed')
    write(output / 'manifest.json', manifest)
    result = dict(status='qualified_training_view', manifest=str((output / 'manifest.json').relative_to(ROOT)),
        manifest_sha256=sha(output / 'manifest.json'), parents=len(parents), snapshots=sum(len(p['frames']) for p in parents),
        shard_bytes=[s['bytes'] for s in shards], cells_per_metre=sorted({1/p['cell_size_m'] for p in parents}),
        method=method, worst_reference_relative_residual=max(c['reference_relative_residual'] for c in checks),
        worst_common_base_reconstruction_error=max(c['base_reconstruction_error'] for c in checks),
        optimizer_updates_executed=0, source_sha256=sha(__file__))
    write(ROOT / 'docs/plans/pressure-methods-v1' / (name + '-training-view.json'), result)
    print(name, 'qualified', len(parents), 'parents', flush=True)


def pack(name):
    reservation_path = ROOT / 'runs/pressure-methods-v1/native-training-02/reservation.json'
    reservation = read(reservation_path)
    teacher = name.startswith('teacher-')
    substeps = {'teacher-coarse':1, 'teacher-half':2, 'teacher-quarter':4}.get(name)
    rows = [r for r in reservation['scenes'] if r['campaign_group'] == ('teacher' if teacher else 'spacing')
            and (not teacher or r['teacher_substeps'] == substeps)]
    if len(rows) != (16 if teacher else 64) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Complete train-only population required')
    coarse_rows = {r['source_training_parent']:r for r in reservation['scenes']
                   if r['campaign_group'] == 'teacher' and r['teacher_substeps'] == 1}
    parents, values, checks = [], {}, []
    evidence = {str(reservation_path.relative_to(ROOT)):sha(reservation_path), str(ORACLE.relative_to(ROOT)):sha(ORACLE)}
    for index,row in enumerate(rows):
        directory, receipt = captured(row)
        evidence[str((directory / 'receipt.json').relative_to(ROOT))] = sha(directory / 'receipt.json')
        coarse_directory, coarse = captured(coarse_rows[row['source_training_parent']]) if teacher else (directory,receipt)
        evidence[str((coarse_directory / 'receipt.json').relative_to(ROOT))] = sha(coarse_directory / 'receipt.json')
        factor = row['teacher_substeps'] if teacher else 1
        fields = {name:[] for name in FIELDS}
        h = 1 / row['cells_per_metre']
        frames = []
        for frame, common_frame in zip(receipt['frames'], coarse['frames']):
            model_index = (frame['index'] + 1) // factor - 1
            if (frame['index'] + 1) % factor or frame['time_s'] != (model_index+1)/60:
                raise ValueError('Matched physical teacher time required')
            base = field(coarse_directory, common_frame, 'base').astype('<f8')
            rhs = (field(directory, frame, 'rhs') / factor).astype('<f8')
            reference = (field(directory, frame, 'pressure') / factor).astype('<f8')
            pre = [field(directory, frame, 'pre_' + a).ravel().tolist() for a in 'uvw']
            rebuilt = np.asarray(oracle.reconstruct_rhs(pre, row['counts_xyz'], [h]*3, 1/60)).reshape(rhs.shape)
            if not np.allclose(rhs, rebuilt, rtol=2e-14, atol=1e-14):
                raise ValueError('Converted teacher does not match model-dt RHS')
            action = lambda p:np.asarray(oracle.full_action(p.ravel().tolist(), row['counts_xyz'], [h]*3, 1000.)).reshape(rhs.shape)
            residual = rhs - action(base)
            native_residual = field(directory, frame, 'native_input_residual') / factor
            native_base = field(directory, frame, 'base') / factor
            reconstructed_residual = native_residual + action(native_base - base)
            if not np.allclose(residual, reconstructed_residual, rtol=2e-12, atol=2e-14):
                raise ValueError('Changed-base residual failed independent native identity')
            closure = rhs - action(reference)
            relative = float(np.linalg.norm(closure.ravel()) / max(np.linalg.norm(rhs.ravel()), 1e-30))
            if np.linalg.norm(closure.ravel()) > max(1e-12, 1e-8*np.linalg.norm(rhs.ravel())):
                raise ValueError('Converted physical target closure failed')
            checks.append(dict(trajectory_id=row['trajectory_id'], index=model_index,
                reference_relative_residual=relative, base_reconstruction_error=float(np.max(np.abs(reconstructed_residual-residual))),
                model_dt_rhs_reconstruction_max=float(np.max(np.abs(rhs-rebuilt)))))
            for key,value in dict(base=base, rhs=rhs, reference=reference, native_input_residual=residual).items():
                fields[key].append(value)
            frames.append(dict(index=model_index, time_s=frame['time_s'], native_source_index=frame['index']))
        for key,stack in fields.items():
            values[f'p{index:04d}_{key}'] = np.stack(stack)
        parents.append(dict(trajectory_id=row['trajectory_id'], split='train', counts_xyz=row['counts_xyz'],
            cell_size_m=h, density_kg_m3=1000., dt_s=1/60, frames=frames,
            source_training_parent=row['source_training_parent'], teacher_substeps=factor,
            native_receipt=str((directory / 'receipt.json').relative_to(ROOT)),
            common_base_receipt=str((coarse_directory / 'receipt.json').relative_to(ROOT))))
    method = (dict(experiment='teacher_temporal_refinement', teacher_substeps=substeps,
        model_dt_s=1/60, pressure_and_rhs_divisor=substeps, initial_guess='Matched coarse-trajectory pressure history for every arm',
        limitation='Fine-trajectory instantaneous pressure requests at model dt; not an exact fine-integrated coarse transition target')
        if teacher else dict(experiment='spacing_variation', cells_per_metre=[4,8,16,32],
            cell_counts='Same original 16 grid/force profiles at each spacing', model_dt_s=1/60))
    emit(name, parents, values, evidence, checks, method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=['teacher-coarse','teacher-half','teacher-quarter','spacing'], required=True)
    pack(parser.parse_args().name)
