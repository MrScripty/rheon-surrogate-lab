"""Freeze synthetic point labels after independent native qualification."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from prepare_flux_teacher import fields
from training_displacement_teacher import displacement_labels
from transport_teacher_support import native, packed_native_points, point_mask
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


def run(args):
    if args.output.exists():
        raise ValueError('Fresh point teacher required')
    registration_path = DOCS / 'point-sampling-registration-01.json'
    registration = read(registration_path)
    for name, digest in registration['source_sha256'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Registered point source changed')
    os.sched_setaffinity(0, {22}); torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    rng = np.random.Generator(np.random.PCG64(registration['data_seed']))
    counts = [12, 12, 12]
    values, physical = fields(rng, counts, 80)
    targets = []
    for i in range(80):
        _, points = displacement_labels(values[i:i+1], counts, 0.)
        targets.append(-points.reshape(1, 12, 13, 13, 13).numpy())
    targets = np.concatenate(targets)
    mask = point_mask(counts).expand(1, 4, 3, 13, 13, 13).reshape(1, 12, 13, 13, 13).numpy()
    cases = []
    for index in registration['native_training_pattern_indices']:
        cases.append((dict(counts=counts, cell_size_m=1/16, dt_s=1/60, point_only=True,
            velocity=[v[index].astype('f8').tolist() for v in physical], tracer=[0.] * 12**3),
            -targets[index:index+1], dict(kind='actual_dataset', index=index)))
    for shape in ([16,16,16], [9,11,11], [15,6,8], [32,32,32]):
        for cpm in (8,16):
            value, faces = fields(rng, shape, 1)
            _, points = displacement_labels(value, shape, 0.)
            cases.append((dict(counts=shape, cell_size_m=1/cpm, dt_s=1/60, point_only=True,
                velocity=[v[0].astype('f8').tolist() for v in faces], tracer=[0.] * int(np.prod(shape))),
                points.reshape(1,12,*value.shape[-3:]).numpy(), dict(kind='geometry', counts=shape, cpm=cpm)))
    qualified = []
    for start in range(0,len(cases),2):
        selected = cases[start:start+2]
        refs = native([c[0] for c in selected])
        for (case, target, identity), reference in zip(selected, refs):
            truth = packed_native_points(reference, case['counts'], 'velocity_samples').reshape(target.shape).numpy()
            relative = float(np.linalg.norm(target-truth)) / max(float(np.linalg.norm(truth)),1e-30)
            if relative > 1e-12:
                raise ValueError('Point teacher fails independent native sampling')
            qualified.append(dict(**identity, relative_error=relative))
    archive = args.output / 'teacher.npz'
    np.savez_compressed(archive, velocity=values.numpy(), target=targets, mask=mask)
    with np.load(archive,allow_pickle=False) as data:
        inventory = {name:dict(shape=list(data[name].shape),dtype=data[name].dtype.str,
            sha256=__import__('hashlib').sha256(data[name].tobytes()).hexdigest()) for name in data.files}
    write(args.output/'manifest.json',dict(status='native_qualified',schema='rheon-point-sampling-teacher-v1',
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,inventory=inventory,
        train_indices=list(range(64)),development_indices=list(range(64,80)),counts_xyz=counts,
        native_qualification=qualified,registration_sha256=sha(registration_path),source_sha256=registration['source_sha256'],
        native_build_receipt_sha256=sha(BASE/'transport-teacher-build-01/build-receipt.json'),
        temporal_scenes_used=False,reserved_scenes_consumed=False,optimizer_updates=0,
        target='Negative sampled velocity at vx/vy/vz-face and tracer-cell points; 12 learned channels'))
    print('Native-qualified point teacher:',len(qualified),'checks;',archive.stat().st_size,'bytes',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
