"""Reserve new force profiles at fixed counts across four independent spacings."""
import copy
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha
import native_training


def prepare():
    output = ROOT/'runs/pressure-methods-v1/native-spacing-validation-03'
    native_training.prepare(output)
    original = read(ROOT/'runs/dependency-migration-v1/training-project-6/inputs/splits.json')
    rng = random.Random(2026101813)
    rows = []
    for profile,counts in enumerate([[16]*3,[13,9,7],[11,15,6]]):
        force = copy.deepcopy(original['parents'][profile+6]['scene']['force'])
        direction = [rng.uniform(-1,1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in direction))
        force.update(direction=[x/norm for x in direction],center_fraction=[rng.uniform(.3,.7) for _ in range(3)],
            acceleration_m_s2=rng.uniform(.15,.45),radius_fraction=rng.uniform(.2,.35),phase=rng.uniform(0,2*math.pi))
        seed = rng.randrange(2**53)
        for spacing in [4,8,16,32]:
            rows.append(dict(trajectory_id=f'spacing-development-profile{profile:02d}-cpm{spacing}',split='validation',
                counts_xyz=counts,cells_per_metre=spacing,force=force,seed=seed,teacher_substeps=1,
                campaign_group='spacing-development',paired_profile=profile))
    reservation = read(output/'reservation.json')
    reservation.update(scenes=rows,scope='Twelve new development-validation trajectories: three new force/grid profiles at 4/8/16/32 cells per metre. No optimizer access or original reserved-test scenes.')
    write(output/'reservation.json',reservation)
    snapshot = output/'spacing-validation-preparer.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    binding = read(output/'binding.json')
    binding.update(reservation_sha256=sha(output/'reservation.json'),
        spacing_validation_extension=dict(source_sha256=sha(snapshot),rows=12,split='validation',
            supersedes_training_extension_population=True,independent_force_seed=2026101813))
    binding['source_sha256'][str(snapshot)] = sha(snapshot)
    write(output/'binding.json',binding)
    print('Reserved twelve independent spacing validation trajectories',flush=True)


if __name__ == '__main__':
    prepare()
