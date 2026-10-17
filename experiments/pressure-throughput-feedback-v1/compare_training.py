"""Bind matched actual CUDA fits and fresh CPU restores to a retained control."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(study, artifact_root):
    if not study.is_relative_to(artifact_root):
        raise ValueError('Study must be inside the selected artifact root')
    receipt = json.loads((study/'receipt.json').read_text())
    if receipt['status'] != 'completed' or not all(receipt['artifact_parity'].values()):
        raise ValueError('completed CUDA fit and exact fresh CPU restore required')
    trained, restored = receipt['training_receipt'], receipt['restore_receipt']
    required = {'weights.npz','normalization.json','predictions.npz','history.json',
                f"checkpoint-{receipt['optimizer_updates']:03d}.npz"}
    if set(receipt['artifact_parity']) != required:
        raise ValueError('complete five-artifact CPU parity inventory required')
    if (trained['mode'] != 'train' or not trained['execution']['device'].startswith('cuda')
            or restored['mode'] != 'restore' or restored['execution']['device'] != 'cpu'
            or restored['execution']['cuda_initialized'] or restored['optimizer_updates'] != 0
            or trained['optimizer_updates'] != receipt['optimizer_updates']
            or trained['closure'] != restored['closure']):
        raise ValueError('actual CUDA/CPU execution or captured closure differs')
    if any(digest(study/'train'/name) != digest(study/'restore'/name) for name in required):
        raise ValueError('actual CPU artifact bytes differ from CUDA output')
    if digest(study/'train/weights.npz') != receipt['weights_sha256']:
        raise ValueError('actual weights differ from Study identity')
    for download in receipt['downloads']:
        path = Path(download['output'])
        if not path.is_absolute():
            path = artifact_root/path
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(artifact_root)
                or path.stat().st_size != download['bytes']
                or digest(path) != download['sha256']):
            raise ValueError('retained downloaded artifact changed')
    history = json.loads((study/'train/history.json').read_text())
    initial = study/'train/checkpoint-000.npz'
    return receipt, dict(
        study=str(study), receipt_sha256=digest(study/'receipt.json'),
        training_run_id=receipt['training_run_id'], restore_run_id=receipt['restore_run_id'],
        weights_sha256=receipt['weights_sha256'], initial_checkpoint_sha256=digest(initial),
        optimizer_updates=receipt['optimizer_updates'], band_feedback=receipt['band_feedback'],
        optimizer_s=history['training_finished_unix']-history['training_started_unix'],
        trainer_s=trained['elapsed_s'], complete_study_command_s=receipt['elapsed_s'],
        peak_cuda_bytes=trained['execution']['peak_allocated_bytes'],
        artifact_parity=receipt['artifact_parity'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Artifact root for relative retained download paths')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', action='append', required=True, help='LABEL=completed Study directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not 1 <= len(a.candidate) <= 8:
        raise ValueError('fresh report and one to eight candidates required')
    root = a.root.resolve(strict=True)
    reference, reference_row = measured(a.reference.resolve(strict=True), root)
    rows = {}
    for item in a.candidate:
        label, separator, path = item.partition('=')
        if not separator or not label or label in rows:
            raise ValueError('unique labelled Study required')
        receipt, row = measured(Path(path).resolve(strict=True), root)
        checks = {key: receipt['training_receipt']['closure'][key] == reference['training_receipt']['closure'][key]
                  for key in ('inputs', 'normalization_sha256', 'runtime_manifest_sha256', 'sample_order_sha256')}
        checks.update(initial_checkpoint_exact=row['initial_checkpoint_sha256'] == reference_row['initial_checkpoint_sha256'],
                      optimizer_updates_exact=row['optimizer_updates'] == reference_row['optimizer_updates'])
        if not all(checks.values()):
            raise ValueError('actual matched training material differs: '+label)
        rows[label] = {**row, 'matched_checks': checks}
    result = dict(format='pressure_matched_weight_training_v1', status='passed',
                  source_sha256=digest(Path(__file__)), artifact_root=str(root), reference=reference_row, candidates=rows,
                  scientific_accuracy_asserted=False, optimizer_updates_executed_by_report=0,
                  scope='Retained control fit is reused, not reexecuted. Complete command clocks include queue/preparation/readback; optimizer and trainer clocks are separate. Accuracy requires independent CPU trajectories.')
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({label: {key: row[key] for key in ('optimizer_s','trainer_s','peak_cuda_bytes')}
                      for label, row in rows.items()}))


if __name__ == '__main__':
    main()
