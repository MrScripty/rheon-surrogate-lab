"""Freeze training, development and blind whole-scene identities before fitting."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def reserve():
    destination=ROOT/'docs/plans/pressure-methods-v2/reservation.json'
    if destination.exists():
        raise ValueError('Reservation already exists; never redraw observed scenes')
    template=read(ROOT/'runs/pressure-methods-v1/native-01/reservation.json')['scenes'][0]['force']
    rng=random.Random(2026101819)
    rows=[]
    shapes={
        'train':[[31,23,17],[61,39,27],[48,13,9]],
        'development':[[29,21,15],[59,43,29],[45,11,7],[13,9,7]],
        'blind':[[27,19,13],[57,41,31],[47,15,11],[11,8,6]],
    }
    for population,grids in shapes.items():
        for profile,counts in enumerate(grids):
            force=copy.deepcopy(template)
            direction=[rng.uniform(-1,1) for _ in range(3)]
            norm=math.sqrt(sum(x*x for x in direction))
            force.update(direction=[x/norm for x in direction],
                center_fraction=[rng.uniform(.25,.75) for _ in range(3)],
                acceleration_m_s2=rng.uniform(.18,.5), radius_fraction=rng.uniform(.18,.32),
                phase=rng.uniform(0,2*math.pi),spatial_modes=[rng.randint(1,5) for _ in range(3)])
            seed=rng.randrange(2**53)
            for spacing in [8,16]:
                rows.append(dict(trajectory_id=f'v2-{population}-{profile:02d}-cpm{spacing}',
                    split='train' if population=='train' else 'validation',population=population,
                    counts_xyz=counts,cells_per_metre=spacing,force=force,seed=seed,teacher_substeps=1))
    configuration=read(ROOT/'runs/pressure-methods-v1/native-01/reservation.json')['configuration']
    reservation=dict(schema='rheon-pressure-methods-v2-reservation',configuration=configuration,
        scenes=rows,blind_capture_gate='Only after candidate selection is written and hashed',
        seed=2026101819,source_sha256=sha(__file__),
        request_sha256=sha('/home/jeremy/.codex/attachments/14676bfb-2793-4ddd-996a-11b2f507747b/pasted-text-1.txt'))
    write(destination,reservation)
    campaign=dict(status='running',request_sha256=reservation['request_sha256'],
        reservation_sha256=sha(destination),directions=[dict(id=i,status='pending') for i in range(1,7)],
        baseline='C53; no promotion',training_retry=False)
    write(destination.with_name('campaign.json'),campaign)
    print('Reserved 6 training, 8 development and 8 blind trajectories',flush=True)

def prepare(output):
    spec=importlib.util.spec_from_file_location('v1_native',ROOT/'experiments/pressure-methods-v1/native_training.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    prior.prepare(output)
    reservation=read(ROOT/'docs/plans/pressure-methods-v2/reservation.json')
    # No blind data are produced or exposed to the trainer before selection.
    active=dict(reservation,scenes=[r for r in reservation['scenes'] if r['population']!='blind'])
    write(output/'reservation.json',active)
    snapshot=output/'v2-reserver.py';snapshot.write_bytes(Path(__file__).read_bytes())
    binding=read(output/'binding.json')
    binding.update(reservation_sha256=sha(output/'reservation.json'),
        v2_reservation_sha256=sha(ROOT/'docs/plans/pressure-methods-v2/reservation.json'))
    binding['source_sha256'][str(snapshot)]=sha(snapshot)
    write(output/'binding.json',binding)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['reserve','prepare'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='reserve':reserve()
    else:prepare(args.output.resolve())
