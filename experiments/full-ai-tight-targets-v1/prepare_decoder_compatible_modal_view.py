"""Encode native pressure labels in the fixed decoder coordinates, without fitting."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from learned_modal_pressure import LearnedModalPressure
from train_modal_pressure import inventory, load
from modal_basis import bases
from diagnose_modal_label_encoding import explicit_coefficients, basis_inverse_encode
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh decoder-compatible training view required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered decoder-label source changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    original = ROOT / reg['view']
    manifest = read(original / 'manifest.json')
    if manifest['status'] != 'verified_train_only_fixed_transform_view' or manifest['scenes'] != 16 or manifest['samples'] != 144 or manifest['development_or_reserved_used']:
        raise ValueError('Exact original train-only native physical data required')
    pair = ROOT / reg['pair']
    receipt = read(pair / 'receipt.json')
    restored = read(pair / 'restore/receipt.json')
    if receipt['status'] != 'completed' or not all(receipt['artifact_parity'].values()) or not restored['full_cpu_probe_bitwise_verified'] or restored['optimizer_updates'] != 0:
        raise ValueError('Exact previous managed pressure fit/CPU restore required')
    for name in ('training_run_id', 'restore_run_id'):
        terminal = read(pair / ('terminal-' + receipt[name] + '.json'))
        if terminal['state'] != 'completed' or terminal['custody_phase'] != 'drained':
            raise ValueError('Both inherited managed owners must drain')
    model = LearnedModalPressure(manifest['geometries'])
    load(model, pair / 'restore/weights.npz', restored['final_inventory'])
    args.output.mkdir(parents=True)
    geometries, sources = [], {}
    with torch.inference_mode():
        for geometry in manifest['geometries']:
            axes = bases(geometry['counts_zyx'])
            qs, ps = [], []
            for sample in geometry['samples']:
                path = ROOT / sample['source']
                if digest(path) != sample['source_sha256']:
                    raise ValueError('Original native physical pressure/RHS changed')
                sources[sample['source']] = sample['source_sha256']
                with np.load(path, allow_pickle=False) as data:
                    p = torch.from_numpy(data['pressure'].copy()).reshape(1, 1, *geometry['counts_zyx'])
                    rhs = torch.from_numpy(data['rhs'].copy()).reshape_as(p)
                q = rhs * (1000. * sample['cells_per_metre'])
                qs.append(explicit_coefficients(q, axes).numpy()[0].copy())
                ps.append(basis_inverse_encode(p, axes).numpy()[0].copy())
            path = args.output / geometry['path']
            np.savez_compressed(path, q_coefficients=np.stack(qs).astype('<f8'),
                                pressure_coefficients=np.stack(ps).astype('<f8'))
            geometries.append(dict(geometry, sha256=digest(path), bytes=path.stat().st_size))
            print(geometry['key'], 'native pressure labels encoded; no fitting', flush=True)
    normalization = read(original / 'normalization.json')
    normalization.update(
        initial_pressure_parameters={name: value.detach().numpy().tolist() for name, value in model.state_dict().items()},
        initial_pressure_inventory=inventory(model), initial_pressure_pair=reg['pair'],
        initial_pressure_archive_sha256=digest(pair / 'restore/weights.npz'),
        parameter_learning_executed=0,
        label_coordinates='Offline inversion of only fixed cosine-coordinate matrices; no physical pressure inverse, coefficient ratios or gain targets',
        input_coordinates='Explicit fixed cosine projection matching the declared complete pressure execution')
    write(args.output / 'normalization.json', normalization)
    if (args.output / 'normalization.json').stat().st_size > 2 * 1024**2:
        raise ValueError('Declared normalization output cap would be exceeded')
    new = dict(manifest, geometries=geometries, source_evidence=sources,
               normalization_sha256=digest(args.output / 'normalization.json'),
               original_view_manifest_sha256=digest(original / 'manifest.json'),
               initialization_provenance=reg['pair'],
               decoder_compatible_label_view_protocol_sha256=digest(args.protocol),
               decoder_compatible_label_view_source_sha256=reg['source_sha256'],
               optimizer_updates=0, inverse_pressure_operator_targets_computed=False,
               fixed_basis_inverse_used_only_for_offline_label_coordinates=True)
    write(args.output / 'manifest.json', new)
    if torch.cuda.is_initialized():
        raise ValueError('Label preparation unexpectedly initialized CUDA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
