"""Train-only pressure label round-trip controls; no gain targets or fitting."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from modal_basis import bases, transform
from numerics import fine_action
import numpy as np
import torch


def explicit_coefficients(p, axes):
    z, y, x = axes
    value = torch.einsum('kz,bzyx->bkyx', z, p[:, 0])
    value = torch.einsum('ly,bkyx->bklx', y, value)
    return torch.einsum('mx,bklx->bklm', x, value)


def explicit_decode(c, axes):
    z, y, x = axes
    value = torch.einsum('mx,bklm->bklx', x, c)
    value = torch.einsum('ly,bklx->bkyx', y, value)
    return torch.einsum('kz,bkyx->bzyx', z, value)[:, None]


def basis_inverse_encode(p, axes):
    # This inverts only the small fixed cosine-coordinate transform in OFFLINE
    # label encoding. It does not form or invert a physical pressure operator,
    # inspect operator eigenvalues, or compute an input-to-pressure multiplier.
    value = p[:, 0]
    for dimension, axis in zip((1, 2, 3), axes):
        value = (value.movedim(dimension, -1) @ torch.linalg.inv(axis)).movedim(-1, dimension)
    return value


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh native-label encoding diagnosis required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered native label source changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_path = ROOT / reg['view'] / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['scenes'] != 16 or manifest['samples'] != 144 or manifest['development_or_reserved_used']:
        raise ValueError('Original 144 training-only native physical states required')
    rows = []
    with torch.inference_mode():
        for geometry in manifest['geometries']:
            axes = bases(geometry['counts_zyx'])
            archive = manifest_path.parent / geometry['path']
            if digest(archive) != geometry['sha256']:
                raise ValueError('Original projected pressure training labels changed')
            with np.load(archive, allow_pickle=False) as data:
                saved_p = data['pressure_coefficients'].copy()
            for index, spec in enumerate(geometry['samples']):
                path = ROOT / spec['source']
                if digest(path) != spec['source_sha256']:
                    raise ValueError('Original physical pressure/RHS changed')
                with np.load(path, allow_pickle=False) as data:
                    p = torch.from_numpy(data['pressure'].copy()).reshape(1, 1, *geometry['counts_zyx'])
                    rhs = torch.from_numpy(data['rhs'].copy()).reshape_as(p)
                h = torch.tensor([1 / spec['cells_per_metre']], dtype=torch.float64)
                rho = torch.tensor([1000.], dtype=torch.float64)
                native_error = relative(fine_action(p, h, rho), rhs)
                if native_error > reg['maximum_native_control_residual']:
                    raise ValueError('Original native teacher does not satisfy its operator')
                controls = dict(original_projected=torch.from_numpy(saved_p[index:index + 1]),
                                explicit_projected=explicit_coefficients(p, axes),
                                fixed_basis_inverse_encoded=basis_inverse_encode(p, axes))
                metrics = {}
                for name, coefficients in controls.items():
                    reconstructed = explicit_decode(coefficients, axes)
                    reconstructed = reconstructed - reconstructed[:, :, :1, :1, :1]
                    metrics[name] = dict(physical_pressure_relative_l2=relative(reconstructed, p),
                                         reference_equation_relative_l2=relative(fine_action(reconstructed, h, rho), rhs))
                rows.append(dict(identifier=spec['identifier'], state=spec['state'], key=geometry['key'],
                                 counts_zyx=geometry['counts_zyx'], cells_per_metre=spec['cells_per_metre'],
                                 native_equation_relative_l2=native_error, controls=metrics))
            print(geometry['key'], 'train-only label round trips diagnosed', flush=True)
    if len(rows) != 144 or torch.cuda.is_initialized():
        raise ValueError('Label diagnostic population/device changed')
    summaries = []
    for key in [row['key'] for row in manifest['geometries']]:
        chosen = [row for row in rows if row['key'] == key]
        summaries.append(dict(key=key, native_samples=len(chosen),
            worst_native_equation=max(row['native_equation_relative_l2'] for row in chosen),
            controls={name: {metric: max(row['controls'][name][metric] for row in chosen)
                      for metric in next(iter(chosen))['controls'][name]}
                      for name in chosen[0]['controls']}))
    write(args.output, dict(status='completed_train_only_label_encoding_diagnostic', rows=rows,
          summaries=summaries, optimizer_updates=0, learned_weights_modified=False,
          gain_ratios_or_inverse_operator_targets_computed=False,
          physical_pressure_operator_inverse_constructed=False,
          fixed_coordinate_basis_inverse_used_only_in_offline_label_control=True,
          development_or_reserved_used=False, full_ai_candidate=False,
          source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
