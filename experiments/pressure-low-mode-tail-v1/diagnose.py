"""Train-only manufactured mode contraction; never read validation pressures."""
import argparse
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write


def diagnose(config_path, identifier, cpu, output):
    started = time.perf_counter()
    config = read(config_path)
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(ROOT / 'experiments/pressure-pass-sweep-v1'))
    import inference
    state = inference.load(read(ROOT/config['baseline_material']), identifier)
    import analytical
    import band_feedback
    torch = state.torch
    inputs = source_inputs(ROOT/state.entry['project'])
    _, parents = state.data.load(tuple(inputs[k].read_bytes() for k in ['dataset', 'dataset_extra']), read(inputs['splits']))
    if not parents or any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Original training parents only')
    actual = [state.data.sample(p, slot) for p in parents for slot in range(len(p['frames']))]
    amplitude = analytical.training_correction_rms(actual)
    template = actual[0]
    rows = []
    worst_defect = 0.
    with torch.no_grad():
        for grid in config['hard_bank']['grids']:
            shape = tuple(reversed(grid['counts_xyz']))
            parent = {**template, 'base': torch.zeros((1, 1, *shape), dtype=torch.float64),
                      'cell_size_m': torch.tensor([1/grid['cells_per_metre']], dtype=torch.float64)}
            for mode in config['diagnostic_modes_zyx']:
                for gain in config['diagnostic_amplitude_multipliers']:
                    sample = analytical.make_sample(parent, [(tuple(mode), 1.)], amplitude*gain)
                    initial = sample['native_input_residual']
                    norm = float(torch.linalg.vector_norm(initial))
                    defect = float(torch.linalg.vector_norm(state.model.full_operator(sample['reference'], sample['cell_size_m'], sample['rho'])-initial))/norm
                    worst_defect = max(worst_defect, defect)
                    if defect > 1e-10:
                        raise ValueError('Independent eigenvalue label disagrees with captured operator')
                    pressure = sample['base']
                    for passes in range(1, 9):
                        pressure = inference.step(state, sample, pressure)
                        if passes not in config['validation_passes']:
                            continue
                        residual = initial-state.model.full_operator(pressure, sample['cell_size_m'], sample['rho'])
                        error = pressure-sample['reference']
                        error = error-error.mean()
                        target = sample['reference']-sample['reference'].mean()
                        total = float(error.square().sum())
                        low = float(band_feedback.coefficients(error, 3).square().sum())
                        rows.append(dict(counts_xyz=grid['counts_xyz'], cell_size_m=1/grid['cells_per_metre'],
                            mode_zyx=mode, amplitude_multiplier=gain, amplitude_rms_pa=amplitude*abs(gain), passes=passes,
                            relative_equation_residual=float(torch.linalg.vector_norm(residual))/norm,
                            mean_free_pressure_relative_l2=float(torch.linalg.vector_norm(error))/float(torch.linalg.vector_norm(target)),
                            low_mode_pressure_error_fraction=low/total if total else 0.))
    summary = []
    for passes in config['validation_passes']:
        for mode in config['diagnostic_modes_zyx']:
            group = [r for r in rows if r['passes'] == passes and r['mode_zyx'] == mode]
            summary.append(dict(passes=passes, mode_zyx=mode, cases=len(group),
                worst_relative_residual=max(r['relative_equation_residual'] for r in group),
                median_relative_residual=statistics.median(r['relative_equation_residual'] for r in group),
                qualified=sum(r['relative_equation_residual'] <= config['target_relative_residual'] for r in group)))
    if torch.cuda.is_initialized():
        raise ValueError('CPU diagnosis only')
    write(output, dict(status='passed', model=identifier, weights_sha256=state.entry['weights_sha256'],
        population='240 new manufactured single-mode cases; original train-only amplitude/scalars; no recorded validation labels',
        frequency='Spatial Neumann cosine modes, not temporal frequency', rows=rows, summary=summary,
        independent_label_worst_operator_defect=worst_defect, amplitude_reference_rms_pa=amplitude,
        elapsed_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu=cpu, threads=torch.get_num_threads(), cuda_initialized=False,
        config_sha256=sha(config_path), source_sha256=sha(__file__),
        inputs={k:dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k,p in inputs.items()}))
    print(identifier, 'manufactured cases', len(rows)//4, 'elapsed', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.config.resolve(), args.model, args.cpu, args.output.resolve())
