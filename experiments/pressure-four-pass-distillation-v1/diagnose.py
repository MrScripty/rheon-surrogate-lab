"""Measure residual frequency and amplitude on original fitting parents only."""
import argparse
import os
from pathlib import Path
import resource
import sys
import time

from common import ROOT, HERE, read, sha, source_inputs, write


def diagnose(config_path, identifier, cpu, output):
    started = time.perf_counter()
    config = read(config_path)
    os.sched_setaffinity(0, {cpu})
    sys.path[:0] = [str(ROOT / 'experiments/pressure-pass-sweep-v1'),
                   str(ROOT / config['source_lab'] / 'experiments/water-residual-learning-v1')]
    import inference
    import audit
    material = read(ROOT / config['baseline_material'])
    state = inference.load(material, identifier)
    torch = state.torch
    audit.TORCH = torch
    inputs = source_inputs(ROOT / state.entry['project'])
    _, parents = state.data.load(tuple(inputs[k].read_bytes() for k in ['dataset', 'dataset_extra']), read(inputs['splits']))
    if not parents or any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Diagnostic must contain only original training parents')
    rows = []
    with torch.no_grad():
        for parent in parents:
            for slot, frame in enumerate(parent['frames']):
                sample = state.data.sample(parent, slot)
                base = sample['base']
                initial = sample['native_input_residual']
                exact = audit.spectral_correction(initial, sample)
                defect = state.model.full_operator(exact, sample['cell_size_m'], sample['rho']) - (initial - initial.mean())
                initial_norm = float(torch.linalg.vector_norm(initial))
                if float(torch.linalg.vector_norm(defect)) / max(initial_norm, 1e-300) > 1e-9:
                    raise ValueError('Analytical target failed captured operator check')
                bases = audit.BASES[tuple(exact.shape[2:])][0]
                signal = audit.cosine_transform((exact-exact.mean())[0, 0], bases)
                signal_norm = float(torch.linalg.vector_norm(signal))
                amplitude = signal.abs() / max(signal_norm, 1e-300)
                indices = torch.meshgrid(*[torch.arange(n) for n in exact.shape[2:]], indexing='ij')
                low = (indices[0] <= 3) & (indices[1] <= 3) & (indices[2] <= 3)
                dc = (indices[0] == 0) & (indices[1] == 0) & (indices[2] == 0)
                pressure = base
                for passes in range(max(config['prefixes']) + 1):
                    if passes:
                        pressure = inference.step(state, sample, pressure)
                    if passes not in config['prefixes']:
                        continue
                    residual = initial - state.model.full_operator(pressure-base, sample['cell_size_m'], sample['rho'])
                    remaining = audit.spectral_correction(residual, sample)
                    errors = audit.cosine_transform((remaining-remaining.mean())[0, 0], bases)
                    rhs_modes = audit.cosine_transform((residual-residual.mean())[0, 0], bases)
                    error_energy = float(errors.square().sum())
                    residual_energy = float(rhs_modes.square().sum())
                    bands = []
                    for band, mask in [('low_modes_0_to_3', low & ~dc), ('remaining_modes', ~low)]:
                        for lo, hi in [(0., 1e-6), (1e-6, 1e-4), (1e-4, .01), (.01, 2.)]:
                            selected = mask & (amplitude >= lo) & (amplitude < hi)
                            bands.append(dict(band=band, signal_coefficient_over_full_l2=[lo, hi], modes=int(selected.sum()),
                                              signal_energy_pa2=float(signal[selected].square().sum()),
                                              pressure_error_energy_pa2=float(errors[selected].square().sum()),
                                              residual_error_energy=float(rhs_modes[selected].square().sum())))
                    rows.append(dict(trajectory_id=parent['entry']['trajectory_id'], split='train', frame=frame['index'],
                                     counts_xyz=parent['counts_xyz'], cell_size_m=parent['cell_size_m'], passes=passes,
                                     relative_equation_residual=float(torch.linalg.vector_norm(residual))/max(initial_norm, 1e-300),
                                     pressure_error_energy_pa2=error_energy, residual_error_energy=residual_energy,
                                     remaining_pressure_relative_l2=error_energy**.5/max(signal_norm, 1e-300), bands=bands))
                audit.BASES.clear()
    summary = []
    for passes in config['prefixes']:
        group = [r for r in rows if r['passes'] == passes]
        total = sum(r['pressure_error_energy_pa2'] for r in group)
        low = sum(b['pressure_error_energy_pa2'] for r in group for b in r['bands'] if b['band'] == 'low_modes_0_to_3')
        weak = sum(b['pressure_error_energy_pa2'] for r in group for b in r['bands'] if b['signal_coefficient_over_full_l2'][1] <= 1e-4)
        summary.append(dict(passes=passes, cases=len(group), worst_equation_residual=max(r['relative_equation_residual'] for r in group),
                            cases_meeting_target=sum(r['relative_equation_residual'] <= config['target_relative_residual'] for r in group),
                            low_mode_pressure_error_energy_fraction=low/total if total else None,
                            weak_signal_pressure_error_energy_fraction=weak/total if total else None))
    write(output, dict(status='passed', model=identifier, weights_sha256=state.entry['weights_sha256'], summary=summary, rows=rows,
                       population='original training parents only; no validation or holdout fields loaded',
                       frequency_definition='Spatial orthonormal Neumann cosine modes; low means each axis index <=3, excluding DC. Not temporal frequencies.',
                       amplitude_definition='Absolute original correction coefficient divided by original full correction L2; bins fixed before measuring error.',
                       cuda_initialized=torch.cuda.is_initialized(), threads=torch.get_num_threads(), cpu=cpu,
                       elapsed_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       inputs={k:dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in inputs.items()},
                       source_sha256=sha(__file__), config_sha256=sha(config_path)))
    print(identifier, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.config.resolve(), args.model, args.cpu, args.output.resolve())
