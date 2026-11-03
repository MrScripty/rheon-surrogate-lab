"""Reserve complete density groups, unseen shapes and varied prescribed forces."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from force_program import PROFILES, MOTIONS, make_scene

ROOT = Path(__file__).resolve().parents[2]
DENSITIES = (8, 10, 16, 25, 50, 100)
TRAIN_SHAPES = ((8,8,8), (16,16,16), (9,11,11), (15,6,8), (24,16,12),
    (32,32,32), (40,24,16), (48,32,24), (64,64,64), (64,48,32),
    (128,24,12), (64,64,12))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve(args):
    if args.output.exists():
        raise ValueError('Fresh whole-scene reservation required')
    rng = random.Random(2026102013)
    prior_seeds, prior_sources = set(), {}
    for folder in ('full-ai-tight-targets-v1', 'full-ai-next-priorities-v1'):
        for path in (ROOT/'docs/plans'/folder).glob('*reservation*.json'):
            doc = json.loads(path.read_text())
            prior_seeds.update(r['seed'] for r in doc.get('rows', []) if 'seed' in r)
            prior_sources[str(path.relative_to(ROOT))] = digest(path)
    groups, rows = [], []
    used_seeds = set(prior_seeds)
    def add(split, shape, profile, motion, shape_role):
        seed = rng.randrange(1, 2**31)
        while seed in used_seeds:
            seed = rng.randrange(1, 2**31)
        used_seeds.add(seed)
        group = f'diversity-{split}-g{len(groups):03d}'
        groups.append(dict(identifier=group, split=split, seed=seed, counts=list(shape),
                           profile=profile, motion=motion, shape_role=shape_role))
        for density in DENSITIES:
            scene = make_scene(seed, shape, density, profile, motion)
            wire_bytes = len((json.dumps(scene, separators=(',',':'))+'\n').encode())
            if wire_bytes > 65536 or max(shape)>128 or min(shape)<5 or __import__('math').prod(shape)>262144:
                raise ValueError('Initial capture contract exceeded')
            rows.append(dict(identifier=f'{group}-cpm{density}', split=split, group=group,
                shape_role=shape_role, profile=profile, motion=motion, scene=scene,
                physical_extent_m=[n*(1/density) for n in shape],
                voxel_count=__import__('math').prod(shape), initial_packet_bytes=wire_bytes))
    for p, profile in enumerate(PROFILES):
        for m, motion in enumerate(MOTIONS):
            for replicate in range(2):
                index=(p*6+m*2+replicate)%len(TRAIN_SHAPES)
                add('train', TRAIN_SHAPES[index], profile, motion, 'training_shape')
    for i, profile in enumerate(PROFILES):
        add('development', TRAIN_SHAPES[5+i], profile, MOTIONS[i%3], 'familiar_shape_new_scene')
    for i, profile in enumerate(PROFILES):
        add('reserved', TRAIN_SHAPES[5+i], profile, MOTIONS[(i+1)%3], 'familiar_shape_new_scene')
        while True:
            shape=tuple(rng.randrange(9,113) for _ in range(3))
            volume=__import__('math').prod(shape)
            if (20000<=volume<=262144 and max(shape)/min(shape)>=1.5 and
                    sorted(shape) not in [sorted(s) for s in TRAIN_SHAPES] and
                    all(sorted(shape)!=sorted(g['counts']) for g in groups)):
                break
        add('reserved', shape, profile, MOTIONS[(i+2)%3], 'unseen_shape_and_new_scene')
    result=dict(status='reserved_before_new_dataset_capture_or_learning', created_unix=time.time(),
        seed=2026102013, steps=240, densities=list(DENSITIES), groups=groups, rows=rows,
        scene_counts_by_split={s:sum(r['split']==s for r in rows) for s in ('train','development','reserved')},
        independent_groups_by_split={s:sum(g['split']==s for g in groups) for s in ('train','development','reserved')},
        density_variants_of_same_forcing_group_share_one_split=True,
        seeds_disjoint_from_prior_reservations=True, original_sources=prior_sources,
        reserved_references_captured=False, optimizer_updates=0,
        unseen_shapes_must_report_unsupported_as_failure=True,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__).resolve(),Path(__file__).with_name('force_program.py').resolve()]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Reserved',len(rows),'scenes in',len(groups),'independent density groups',result['scene_counts_by_split'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    reserve(parser.parse_args())
