"""Prepare broader teacher/spacing data and independent domain development scenes.

Historical reservations stay immutable. All new rows are declared before capture.
"""
import argparse
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha


def prepare(output):
    spec = importlib.util.spec_from_file_location('retained_native_preparation', HERE/'native_prepare.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    owner.prepare(output)
    original = read(ROOT/'runs/dependency-migration-v1/training-project-6/inputs/splits.json')
    parents = original['parents']
    if len(parents) != 16 or any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Exact original 16 training profiles required')
    reservation = read(output/'reservation.json')
    rows = []
    for index, parent in enumerate(parents):
        for substeps in [1, 2, 4]:
            rows.append(dict(trajectory_id=f'teacher-parent{index:02d}-substeps{substeps}', split='train',
                counts_xyz=parent['counts_xyz'], cells_per_metre=round(1/parent['cell_size_m']),
                force=parent['scene']['force'], seed=parent['scene']['seed'], teacher_substeps=substeps,
                campaign_group='teacher', source_training_parent=parent['entry']['trajectory_id']))
        for spacing in [4, 8, 16, 32]:
            rows.append(dict(trajectory_id=f'spacing-parent{index:02d}-cpm{spacing}', split='train',
                counts_xyz=parent['counts_xyz'], cells_per_metre=spacing,
                force=parent['scene']['force'], seed=parent['scene']['seed'], teacher_substeps=1,
                campaign_group='spacing', source_training_parent=parent['entry']['trajectory_id']))
    rng = random.Random(2026101810)
    geometries = [([32]*3, 8), ([32]*3, 16), ([64]*3, 8), ([64]*3, 16),
                  ([31, 25, 17], 8), ([63, 49, 33], 16)]
    for index, (counts, spacing) in enumerate(geometries):
        force = copy.deepcopy(parents[(index+3)%len(parents)]['scene']['force'])
        direction = [rng.uniform(-1, 1) for _ in range(3)]
        norm = math.sqrt(sum(v*v for v in direction))
        force.update(direction=[v/norm for v in direction], center_fraction=[rng.uniform(.3, .7) for _ in range(3)],
            acceleration_m_s2=rng.uniform(.2, .6), radius_fraction=rng.uniform(.2, .35), phase=rng.uniform(0, 2*math.pi))
        rows.append(dict(trajectory_id=f'domain-development-{index:02d}', split='validation',
            counts_xyz=counts, cells_per_metre=spacing, force=force, seed=rng.randrange(2**53),
            teacher_substeps=1, campaign_group='domain-development'))
    reservation.update(schema='rheon-pressure-methods-training-reservation-v1', scenes=rows,
        scope='48 teacher and 64 spacing training trajectories from original training profiles; six new development validation scenes. No original reserved-test scene is loaded.')
    write(output/'reservation.json', reservation)
    original_capture = (HERE/'capture_native.py').read_text()
    old = "if row not in reservation['scenes'] or row['split'] != 'train':"
    if original_capture.count(old) != 1:
        raise ValueError('Exact development capture admission seam required')
    capture = original_capture.replace(old, "if row not in reservation['scenes'] or row['split'] not in ('train', 'validation'):")
    capture = capture.replace("raise ValueError('Only exactly reserved training pilots allowed')", "raise ValueError('Only exactly reserved training or development validation scenes allowed')")
    capture = capture.replace("ROOT = Path(__file__).resolve().parents[2]", 'ROOT = Path('+repr(str(ROOT))+')')
    (output/'capture_development.py').write_text(capture)
    snapshot = output/'preparer-source.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    binding = read(output/'binding.json')
    binding.update(reservation_sha256=sha(output/'reservation.json'),
        training_extension=dict(source_sha256=sha(snapshot), original_capture_sha256=sha(HERE/'capture_native.py'),
            capture_admission_change='Exact reservation entries with train/validation split; no reserved_test accepted.',
            dataset_counts=dict(teacher=48, spacing=64, domain_development=6)))
    for path in [snapshot, output/'capture_development.py']:
        binding['source_sha256'][str(path)] = sha(path)
    write(output/'binding.json', binding)
    print('Prepared', len(rows), 'broader training/development reservations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve())
