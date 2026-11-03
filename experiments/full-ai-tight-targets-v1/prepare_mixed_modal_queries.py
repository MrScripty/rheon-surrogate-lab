"""Retain native data and add bounded mixed-mode OFFLINE pressure supervision."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT, read, write, digest
from learned_modal_pressure import LearnedModalPressure
from train_modal_pressure import inventory, load
from modal_basis import bases
from diagnose_modal_label_encoding import explicit_coefficients, explicit_decode, basis_inverse_encode
from numerics import correction, fine_action
import numpy as np
import torch


def conditioning(coefficients):
    rms = coefficients.square().mean(0).sqrt()
    mask = torch.ones_like(rms, dtype=torch.bool)
    mask[0, 0, 0] = False
    normalized = rms[mask] / rms[mask].square().mean().sqrt().clamp_min(1e-300)
    return dict(minimum_mode_rms_over_global=float(normalized.min()),
                median_mode_rms_over_global=float(normalized.median()),
                modes_below_one_millionth=int((normalized < 1e-6).sum()),
                nonDC_modes=int(mask.sum()))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh mixed-mode pressure training view required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered mixed-query source changed: ' + path)
    if reg['queries_per_geometry'] != 32:
        raise ValueError('Exactly 32 predeclared mixed queries per geometry required')
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    original = ROOT / reg['view']
    manifest = read(original / 'manifest.json')
    if manifest['samples'] != 144 or manifest['scenes'] != 16 or manifest['development_or_reserved_used']:
        raise ValueError('All original native training pairs without development data required')
    pair = ROOT / reg['pair']
    receipt, restored = read(pair / 'receipt.json'), read(pair / 'restore/receipt.json')
    if receipt['status'] != 'completed' or not all(receipt['artifact_parity'].values()) or not restored['full_cpu_probe_bitwise_verified'] or restored['optimizer_updates'] != 0:
        raise ValueError('Exact inherited managed checkpoint required')
    for key in ('training_run_id', 'restore_run_id'):
        owner = read(pair / ('terminal-' + receipt[key] + '.json'))
        if owner['state'] != 'completed' or owner['custody_phase'] != 'drained':
            raise ValueError('Inherited owners must complete and drain')
    model = LearnedModalPressure(manifest['geometries'])
    load(model, pair / 'restore/weights.npz', restored['final_inventory'])
    before = inventory(model)
    generator = torch.Generator(device='cpu').manual_seed(reg['seed'])
    args.output.mkdir(parents=True)
    geometries, diagnostics = [], []
    one = torch.ones(1, dtype=torch.float64)
    with torch.inference_mode():
        for geometry in manifest['geometries']:
            if max(geometry['counts_zyx']) > 32:
                raise ValueError('Larger-domain stage remains gated')
            path = original / geometry['path']
            if digest(path) != geometry['sha256']:
                raise ValueError('Original encoded native pairs changed')
            with np.load(path, allow_pickle=False) as archive:
                original_q = archive['q_coefficients'].copy()
                original_p = archive['pressure_coefficients'].copy()
            q_values, p_values = list(original_q), list(original_p)
            key = geometry['key']
            axes = bases(geometry['counts_zyx'])
            gains = model.gains(key).detach().clone()
            gains[0, 0, 0] = 1.
            source_pressure = torch.from_numpy(original_p)
            mask = torch.ones_like(gains, dtype=torch.bool)
            mask[0, 0, 0] = False
            amplitude = source_pressure[:, mask].square().mean().sqrt()
            physical_q, physical_p, samples, errors = [], [], list(geometry['samples']), []
            for index in range(reg['queries_per_geometry']):
                # Excite all non-DC modes together. The frozen learned gains
                # only shape INPUT queries; no pressure/gain ratio is a label.
                modes = torch.randn((1, *geometry['counts_zyx']), generator=generator, dtype=torch.float64) * amplitude
                modes[:, 0, 0, 0] = 0.
                q = explicit_decode(modes / gains, axes)
                q = q - q.mean()
                # The numerical inverse is a label-only OFFLINE reference.
                # Neither this function nor any eigenvalues enter inference.
                p = correction(q, one, one, max_mode=None)
                error = float((fine_action(p, one, one) - q).norm() / q.norm().clamp_min(1e-30))
                if error > reg['maximum_reference_equation_error']:
                    raise ValueError('Mixed-query pressure reference failed its equation')
                q_values.append(explicit_coefficients(q, axes).numpy()[0].copy())
                p_values.append(basis_inverse_encode(p, axes).numpy()[0].copy())
                physical_q.append(q.numpy()[0, 0].copy())
                physical_p.append(p.numpy()[0, 0].copy())
                errors.append(error)
                samples.append(dict(kind='mixed_mode_reference_query', index=index,
                    geometry=key, input_modes='All nonDC modes jointly excited by fixed-seed noise',
                    query_scale='Original train-only nonDC pressure RMS; shaped using frozen learned gains',
                    reference='Offline full numerical pressure reference in the existing unit-normalized coordinates',
                    native_reference_equation_error=error))
            q_values, p_values = np.stack(q_values).astype('<f8'), np.stack(p_values).astype('<f8')
            if not np.array_equal(q_values[:len(original_q)].view('<u8'), original_q.view('<u8')) or not np.array_equal(p_values[:len(original_p)].view('<u8'), original_p.view('<u8')):
                raise ValueError('Original native pairs were not retained bitwise')
            target = args.output / geometry['path']
            np.savez_compressed(target, q_coefficients=q_values, pressure_coefficients=p_values)
            if target.stat().st_size > 64 * 1024**2:
                raise ValueError('Registered augmented shard cap exceeded')
            physical = args.output / (key + '-offline-query-reference.npz')
            np.savez_compressed(physical, q=np.stack(physical_q), pressure=np.stack(physical_p))
            geometries.append(dict(geometry, samples=samples, sha256=digest(target), bytes=target.stat().st_size,
                original_native_samples=len(original_q), augmented_component_queries=len(errors),
                offline_reference=dict(path=physical.name, sha256=digest(physical), bytes=physical.stat().st_size)))
            diagnostics.append(dict(key=key, native=conditioning(source_pressure),
                augmented=conditioning(torch.from_numpy(p_values)),
                maximum_reference_equation_error=max(errors), original_pairs_bitwise_retained=True))
            print(key, 'retained36native pairs and prepared32mixed-mode reference queries', flush=True)
    if inventory(model) != before or torch.cuda.is_initialized():
        raise ValueError('Preprocessing changed learned weights or initialized CUDA')
    normalization = read(original / 'normalization.json')
    normalization.update(initial_pressure_parameters={name: value.detach().numpy().tolist() for name, value in model.state_dict().items()},
        initial_pressure_inventory=before, initial_pressure_pair=reg['pair'],
        initial_pressure_archive_sha256=digest(pair / 'restore/weights.npz'),
        parameter_learning_executed=0, mixed_query_seed=reg['seed'],
        mixed_query_reference_scope='OFFLINE training-label generation only; no reference solver or eigenvalues at inference')
    write(args.output / 'normalization.json', normalization)
    if (args.output / 'normalization.json').stat().st_size > 2 * 1024**2:
        raise ValueError('Normalization publication cap would be exceeded')
    augmented = dict(manifest, geometries=geometries, samples=272, native_training_samples=144,
        augmented_component_queries=128, normalization_sha256=digest(args.output / 'normalization.json'),
        initialization_provenance=reg['pair'], original_native_pairs_bitwise_retained=True,
        mixed_query_view_protocol_sha256=digest(args.protocol), mixed_query_view_source_sha256=reg['source_sha256'],
        offline_numerical_reference_used_only_for_labels=True, gain_ratios_or_inverse_coefficient_targets_computed=False,
        optimizer_updates=0, higher_density_phase_started=False)
    write(args.output / 'manifest.json', augmented)
    write(args.output / 'conditioning-diagnostic.json', dict(status='completed_training_data_conditioning_preparation',
        diagnostics=diagnostics, optimizer_updates=0, original_native_states=144, mixed_queries=128,
        learned_tensors_unchanged=True, development_or_reserved_used=False,
        numerical_pressure_reference_is_training_only=True, protocol_sha256=digest(args.protocol)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
