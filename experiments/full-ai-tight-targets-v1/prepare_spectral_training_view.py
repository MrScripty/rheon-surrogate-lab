"""Fixed, train-only double DCT features from original native pressure/RHS.

No gain targets, coefficient ratios, fits or optimizer updates are computed.
"""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from preflight_global_pressure import bases, transform
from numerics import fine_action
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh train-only source view required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered fixed preprocessing source changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']}); torch.set_num_threads(1)
    selected = [r for r in read(ROOT / reg['bank'])['trajectories'] if r['split'] == 'train']
    if len(selected) != 16 or any(not r['identifier'].endswith(('-r0', '-r1')) for r in selected):
        raise ValueError('All 16 original training scenes, without development/reserved, are required')
    args.output.mkdir(parents=True)
    groups, sources, records = {}, {}, []
    with torch.inference_mode():
        for row in selected:
            path = ROOT / row['receipt']
            if digest(path) != row['receipt_sha256']:
                raise ValueError('Original train receipt changed')
            receipt = read(path); scene = receipt['scene']; shape = tuple(reversed(scene['counts']))
            if scene['cells_per_metre'] not in (8, 16) or max(shape) > 32:
                raise ValueError('Higher density/size phase remains gated')
            axes = bases(shape, torch.float64); key = 'g' + '_'.join(map(str, shape))
            group = groups.setdefault(key, dict(shape=shape, q=[], p=[], samples=[]))
            sources[str(path.relative_to(ROOT))] = digest(path)
            h = torch.tensor([1. / scene['cells_per_metre']], dtype=torch.float64)
            rho = torch.tensor([1000.], dtype=torch.float64)
            for step in reg['states']:
                entry = receipt['states'][step]; state = path.parent / entry['path']
                if digest(state) != entry['sha256']:
                    raise ValueError('Original double pressure/RHS archive changed')
                sources[str(state.relative_to(ROOT))] = digest(state)
                with np.load(state, allow_pickle=False) as archive:
                    p = torch.from_numpy(archive['pressure'].copy()).double().reshape(1, 1, *shape)
                    rhs = torch.from_numpy(archive['rhs'].copy()).double().reshape_as(p)
                error = float((fine_action(p, h, rho) - rhs).norm() / rhs.norm().clamp_min(1e-30))
                if error > reg['maximum_native_control_residual']:
                    raise ValueError('Native teacher/operator identity failed')
                # Unit conversion and basis are fixed preprocessing, without
                # forming an inverse, eigenvalues, targets or fitted gains.
                q = rhs * (1000. * scene['cells_per_metre'])
                group['q'].append(transform(q, axes).numpy()[0, 0].copy())
                group['p'].append(transform(p, axes).numpy()[0, 0].copy())
                sample = dict(identifier=row['identifier'], state=step, cells_per_metre=scene['cells_per_metre'],
                              source=str(state.relative_to(ROOT)), source_sha256=digest(state), native_control_residual=error)
                group['samples'].append(sample); records.append(sample)
            print(row['identifier'], 'fixed transform features prepared', flush=True)
    geometries = []
    for key, group in groups.items():
        path = args.output / (key + '.npz')
        np.savez_compressed(path, q_coefficients=np.stack(group['q']).astype('<f8'),
                            pressure_coefficients=np.stack(group['p']).astype('<f8'))
        geometries.append(dict(key=key, counts_zyx=group['shape'], samples=group['samples'],
                               path=path.name, sha256=digest(path), bytes=path.stat().st_size))
    normalization = dict(schema='rheon-pressure-unit-modal-view-v1', pressure_dtype='<f8', rhs_dtype='<f8',
                         basis='Full orthonormal real DCT-II, native cell centres, all three axes',
                         q_units='native RHS * fluid mass density * cells_per_metre',
                         pressure_gauge='Native cell-zero; model DC is fixed to zero and physical output regauged',
                         fitted_or_normalized_gain_targets=False, training_scenes=16,
                         development_or_reserved_used=False)
    write(args.output / 'normalization.json', normalization)
    write(args.output / 'manifest.json', dict(status='verified_train_only_fixed_transform_view',
        geometries=geometries, scenes=16, samples=len(records), source_evidence=sources,
        normalization_sha256=digest(args.output / 'normalization.json'), protocol_sha256=digest(args.protocol),
        source_sha256=reg['source_sha256'], optimizer_updates=0, gain_targets_computed=False,
        analytic_pressure_inverse_constructed=False, development_or_reserved_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
