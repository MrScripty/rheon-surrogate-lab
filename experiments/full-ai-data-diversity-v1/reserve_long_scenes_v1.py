"""Freeze independent scenario splits before long reference capture or learning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def control_identity(frames):
    value = np.asarray([[*f['center'], f['pulse']] for f in frames], dtype='<f8')
    return dict(shape=list(value.shape), dtype='<f8',
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def instantiate(row):
    from continuous_force_program import evaluate
    scene = copy.deepcopy(row['scene'])
    dt = 1 / (60 * scene['reference_substeps'])
    scene['forcing_schedule'] = [evaluate(scene['forcing_program'], step * dt)
        for step in range(scene['steps'] * scene['reference_substeps'])]
    if control_identity(scene['forcing_schedule']) != row['fine_controls']:
        raise ValueError('Frozen complete fine control path changed')
    return scene


def run(args):
    from support import read, digest
    from force_program import PROFILES, MOTIONS
    from long_reference_scene_v1 import make_long_scene
    if args.output.exists():
        raise ValueError('Fresh scenario reservation required')
    reg = read(args.protocol)
    for name, h in reg['source_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('Registered reservation source changed')
    used = set()
    for name in reg['prior_reservation_metadata']:
        document = read(ROOT / name)
        for row in document.get('rows', []):
            value = row.get('seed', row.get('scene', {}).get('seed'))
            if value is not None:
                used.add(value)
    previous_seeds = set(used)
    rng = random.Random(reg['seed'])
    shapes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (16, 40, 24),
              (24, 32, 48), (64, 64, 64), (64, 48, 32), (12, 128, 24), (64, 64, 12)]
    densities = [8, 10, 16, 25, 50, 100]
    pilots = {(0, 0, 0): ((32, 32, 32), .5),
              (1, 1, 0): ((16, 40, 24), 3.),
              (2, 2, 0): ((64, 48, 32), 9.),
              (3, 2, 0): ((24, 32, 48), 17.)}
    rows = []
    for p, profile in enumerate(PROFILES):
        for m, motion in enumerate(MOTIONS):
            cell = p * 3 + m
            for replica in range(3):
                seed = rng.randrange(1, 2 ** 31)
                while seed in used:
                    seed = rng.randrange(1, 2 ** 31)
                used.add(seed)
                density = densities[(cell + replica * 3) % 6]
                counts = shapes[(cell * 5 + replica * 2) % len(shapes)]
                counts = tuple(rng.sample(list(counts), 3))
                pilot = (p, m, replica) in pilots
                frequency = None
                if pilot:
                    counts, frequency = pilots[p, m, replica]
                scene = make_long_scene(seed, counts, density, profile, motion,
                                        force_frequency_hz=frequency)
                is_quiescent = (p, m, replica) == (0, 0, 1)
                if is_quiescent:
                    scene['acceleration'] = 0.
                recipe = dict(seed=seed ^ 0x36189AB, tracer_blobs=2 + replica,
                    initial_velocity_max_m_per_s=(.002 + .002 * replica)
                        if (cell + replica) % 3 == 1 and not is_quiescent else 0.,
                    curl_z_modes=1 + replica % 2)
                controls = scene.pop('forcing_schedule')
                split = 'train' if replica < 2 else 'development'
                identifier = f'long-v1-{profile}-{motion}-r{replica}-cpm{density}'
                rows.append(dict(identifier=identifier, physical_scenario_group_id=identifier,
                    split=split, seed=seed, profile=profile, motion=motion, replica=replica,
                    pilot=pilot, scene=scene, initial_state_recipe=recipe,
                    quiescent_control=is_quiescent, fine_controls=control_identity(controls),
                    physical_extent_m=[n / density for n in counts],
                    voxel_count=int(np.prod(counts)), fine_steps=4800,
                    physical_duration_s=20))
    counts = {split: sum(r['split'] == split for r in rows) for split in ('train', 'development')}
    if counts != dict(train=24, development=12) or len({r['seed'] for r in rows}) != 36:
        raise ValueError('Complete independent scenario population differs')
    if any(r['seed'] in previous_seeds for r in rows) or sum(r['pilot'] for r in rows) != 4:
        raise ValueError('New scenario/pilot independence differs')
    for split, n in counts.items():
        for density in densities:
            if sum(r['split'] == split and r['scene']['cells_per_metre'] == density for r in rows) != n // 6:
                raise ValueError('Density margin is unbalanced')
    document = dict(status='reserved_before_long_capture_or_learning', created_unix=time.time(),
        rows=rows, unique_master_trajectories=36, scene_counts_by_split=counts,
        densities=densities, all12temporal_views_share_physical_scene_split=True,
        seeds_disjoint_from_prior_metadata=True, new_references_opened=0,
        reserved52_references_opened=0, optimizer_updates=0,
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print('Reserved36new20second scenarios:24training,12development; four pilot cases.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
