"""Qualified frozen-model numerical correction screen; no optimizer or simulation.

Workers own one checkpoint each. Immutable common manufactured inputs are made
from C53 training-only scalars and amplitude, never from validation labels.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, source_inputs, write


def load(config, identifier, cpu):
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    state = inference.load(read(ROOT/config['baseline_material']), identifier)
    return state, inference


def freeze(config_path, output, cpu):
    config = read(config_path)
    state, _ = load(config, 'C53', cpu)
    import analytical
    inputs = source_inputs(ROOT/state.entry['project'])
    _, parents = state.data.load(tuple(inputs[k].read_bytes() for k in ['dataset', 'dataset_extra']), read(inputs['splits']))
    if not parents or any(p['entry']['split'] != 'train' for p in parents):
        raise ValueError('Train-only calibration required')
    actual = [state.data.sample(p, i) for p in parents for i in range(len(p['frames']))]
    amplitude = analytical.training_correction_rms(actual)
    diagnostic = read(ROOT/'experiments/pressure-low-mode-tail-v1/config.json')
    torch, np = state.torch, state.np
    rng = np.random.Generator(np.random.PCG64(config['mixed_mode_seed']))
    rows, arrays = [], {}
    for grid in diagnostic['hard_bank']['grids']:
        shape = tuple(reversed(grid['counts_xyz']))
        template = {**actual[0], 'base': torch.zeros((1, 1, *shape), dtype=torch.float64),
                    'cell_size_m': torch.tensor([1/grid['cells_per_metre']], dtype=torch.float64)}
        cases = []
        for mode in diagnostic['diagnostic_modes_zyx']:
            for gain in diagnostic['diagnostic_amplitude_multipliers']:
                cases.append(('manufactured240', [(tuple(mode), 1.)], gain))
        for _ in range(config['mixed_cases_per_grid']):
            high = tuple(int(rng.integers(4, n)) for n in shape)
            low = tuple(int(rng.integers(0, 4)) for _ in shape)
            if not any(low):
                low = (0, 0, 1)
            cases.append(('mixed32', [(low, float(rng.uniform(.5, 1.5))),
                                     (high, float(rng.uniform(.1, 1.)))], 1.))
        for population, components, gain in cases:
            sample = analytical.make_sample(template, components, amplitude*gain)
            # Independently formed spectral labels must agree with the captured stencil.
            action = state.model.full_operator(sample['reference'], sample['cell_size_m'], sample['rho'])
            defect = float(torch.linalg.vector_norm(action-sample['rhs']))/float(torch.linalg.vector_norm(sample['rhs']))
            if defect > 1e-10:
                raise ValueError('Manufactured label disagrees with retained physical operator')
            index = len(rows)
            for key, value in sample.items():
                arrays[f'{index:04d}_{key}'] = value.numpy()
            rows.append(dict(index=index, population=population, counts_xyz=grid['counts_xyz'],
                cells_per_metre=grid['cells_per_metre'], components_zyx=components, amplitude_multiplier=gain,
                independent_label_operator_defect=defect))
    if output.exists():
        raise ValueError('Fresh bank output required')
    output.mkdir(parents=True)
    np.savez(output/'samples.npz', **arrays)
    write(output/'bank.json', dict(status='frozen', cases=rows, amplitude_rms_pa=amplitude,
        population='240 existing manufactured diagnostics plus 32 prespecified mixed modes; train-only amplitude/scalars',
        data_sha256=sha(output/'samples.npz'), config_sha256=sha(config_path), source_sha256=sha(__file__),
        sources={k:dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in inputs.items()}))
    print('Frozen', len(rows), 'manufactured cases', flush=True)


def policies(config):
    rows = [dict(name=f'neural-{n}', passes=n, ordering='none', mode=None)
            for n in config['neural_only_passes']]
    for ordering in config['orderings']:
        rows += [dict(name=f'{ordering}-{n}', passes=n, ordering=ordering,
                      mode=config['numerical_max_mode']) for n in config['hybrid_passes']]
    return rows + [dict(name='numerical-low-only', passes=0, ordering='before', mode=config['numerical_max_mode']),
                   dict(name='numerical-full-only', passes=0, ordering='before', mode=None)]


def evaluate(config_path, bank_dir, identifier, output, cpu):
    started = time.perf_counter()
    config = read(config_path)
    state, inference = load(config, identifier, cpu)
    import numerics
    torch, np = state.torch, state.np
    selected = read(ROOT/config['baseline_material'])
    inputs = ROOT/selected['config']['validation_inputs']
    for name, expected in selected['validation_sources'].items():
        if sha(inputs/name) != expected:
            raise ValueError('Consumed validation input changed')
    _, parents = state.data.load(tuple((inputs/name).read_bytes() for name in
        ['validation_dataset.npz', 'validation_dataset_extra.npz']), read(inputs/'validation_splits.json'))
    identities = [dict(trajectory_id=p['entry']['trajectory_id'], native_frame=f['index'],
                       counts_xyz=p['counts_xyz'], cell_size_m=p['cell_size_m'])
                  for p in parents for f in p['frames']]
    def key(c):
        return (c['trajectory_id'], c.get('native_frame', c.get('frame')), tuple(c['counts_xyz']), c['cell_size_m'])
    if len(identities) != 128 or sorted(map(key, identities)) != sorted(map(key, selected['validation_identities'])):
        raise ValueError('Exact consumed 128-request population required')
    bank = read(bank_dir/'bank.json')
    if bank['config_sha256'] != sha(config_path) or bank['data_sha256'] != sha(bank_dir/'samples.npz'):
        raise ValueError('Common manufactured bank changed')
    cases = [(dict(population='consumed128', **identity), state.data.sample(p, i))
             for p in parents for i, _ in enumerate(p['frames'])
             for identity in [dict(trajectory_id=p['entry']['trajectory_id'], native_frame=p['frames'][i]['index'],
                                   counts_xyz=p['counts_xyz'], cell_size_m=p['cell_size_m'])]]
    with np.load(bank_dir/'samples.npz', allow_pickle=False) as arrays:
        for row in bank['cases']:
            sample = {name:torch.from_numpy(arrays[f"{row['index']:04d}_{name}"].copy())
                      for name in ('base', 'rhs', 'reference', 'native_input_residual', 'cell_size_m', 'rho', 'dt')}
            cases.append((row, sample))
    output.mkdir(parents=True, exist_ok=False)
    measurements = []
    selected_policies = policies(config)
    if identifier != 'C53':
        selected_policies = [p for p in selected_policies if p['passes']]
    calls = []
    hook = state.network.register_forward_hook(lambda *_: calls.append(1))

    def action(sample, value):
        return state.model.full_operator(value, sample['cell_size_m'], sample['rho'])

    def run(sample, policy, trace=False):
        current = sample['base']
        rhs_norm = max(float(torch.linalg.vector_norm(sample['rhs'])), 1e-300)
        residuals = []
        def record():
            if trace:
                residuals.append(float(torch.linalg.vector_norm(sample['rhs']-action(sample, current)))/rhs_norm)
        record()
        if policy['ordering'] == 'before':
            residual = sample['rhs']-action(sample, current)
            current = current+numerics.correction(residual, sample['cell_size_m'], sample['rho'], policy['mode'])
            record()
        for _ in range(policy['passes']):
            current = inference.step(state, sample, current)
            record()
        if policy['ordering'] == 'after':
            residual = sample['rhs']-action(sample, current)
            current = current+numerics.correction(residual, sample['cell_size_m'], sample['rho'], policy['mode'])
            record()
        remaining = sample['rhs']-action(sample, current)
        relative = float(torch.linalg.vector_norm(remaining))/rhs_norm
        return current, relative, residuals

    try:
        with torch.no_grad():
            for index, (identity, sample) in enumerate(cases):
                # Verify captured base/initial residual closure before comparing policies.
                closure = sample['rhs']-action(sample, sample['base'])-sample['native_input_residual']
                closure_relative = float(torch.linalg.vector_norm(closure))/max(float(torch.linalg.vector_norm(sample['rhs'])), 1e-300)
                if closure_relative > 1e-10:
                    raise ValueError('Incoming pressure residual closure failed')
                for policy in selected_policies:
                    calls.clear()
                    pressure, relative, trace = run(sample, policy, trace=True)
                    trace_calls = len(calls)
                    times = []
                    for _ in range(config['timing_repetitions']):
                        calls.clear()
                        tick = time.perf_counter()
                        timed_pressure, timed_relative, _ = run(sample, policy)
                        times.append(time.perf_counter()-tick)
                        if len(calls) != trace_calls or timed_relative != relative or not torch.equal(timed_pressure, pressure):
                            raise ValueError('Repeated policy output/work changed')
                    delta = pressure-sample['reference']
                    centered = delta-delta.mean()
                    reference = sample['reference']-sample['reference'].mean()
                    gradient_error = sum(float(torch.diff(delta, dim=a).square().sum()) for a in (2, 3, 4))
                    gradient_signal = sum(float(torch.diff(reference, dim=a).square().sum()) for a in (2, 3, 4))
                    if not math.isfinite(relative) or not bool(torch.isfinite(pressure).all()) or float(pressure.flatten()[0]) != 0.:
                        raise ValueError('Nonfinite output or changed pressure gauge')
                    measurements.append(dict(case=index, **identity, policy=policy['name'],
                        checkpoint=identifier, relative_equation_residual=relative,
                        mean_free_pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)), 1e-300),
                        gradient_relative_l2=math.sqrt(gradient_error/max(gradient_signal, 1e-300)),
                        residual_trace=trace, network_calls=trace_calls,
                        numerical_corrections=int(policy['ordering'] != 'none'),
                        explicit_operator_applications=policy['passes']+1+int(policy['ordering'] != 'none'),
                        incoming_closure_relative=closure_relative,
                        timing_samples_s=times, median_policy_s=statistics.median(times)))
                if index % 16 == 0:
                    write(output/'progress.json', dict(completed_cases=index+1, total=len(cases), checkpoint=identifier))
                    print(identifier, index+1, '/', len(cases), flush=True)
    finally:
        hook.remove()
    if torch.cuda.is_initialized() or torch.get_num_threads() != 1:
        raise ValueError('Single-thread CPU evaluation required')
    summaries = []
    for population in sorted({r['population'] for r in measurements}):
        for policy in selected_policies:
            group = [r for r in measurements if r['population'] == population and r['policy'] == policy['name']]
            summaries.append(dict(population=population, policy=policy['name'], cases=len(group),
                qualified=sum(r['relative_equation_residual'] <= config['target_relative_residual'] for r in group),
                worst_relative_equation_residual=max(r['relative_equation_residual'] for r in group),
                median_relative_equation_residual=statistics.median(r['relative_equation_residual'] for r in group),
                worst_pressure_relative_l2=max(r['mean_free_pressure_relative_l2'] for r in group),
                worst_gradient_relative_l2=max(r['gradient_relative_l2'] for r in group),
                median_policy_ms=1000*statistics.median(r['median_policy_s'] for r in group)))
    write(output/'results.json', dict(status='completed', checkpoint=identifier, summary=summaries, rows=measurements,
        config_sha256=sha(config_path), bank_sha256=sha(bank_dir/'bank.json'), material_sha256=sha(ROOT/config['baseline_material']),
        weights_sha256=state.entry['weights_sha256'], source_sha256=sha(__file__), numerics_sha256=sha(HERE/'numerics.py'),
        elapsed_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu=cpu, threads=torch.get_num_threads(), cuda_initialized=False, optimizer_updates=0,
        timing_scope='Warm-cache policy from prepared physical tensors through final relative residual; includes numerical transforms, reconstruction, neural inference and recurrence. Excludes sample loading/assembly, oracle comparison, trace diagnostics, IPC and simulation. All compared policies share this scope.',
        cold_setup_scope='Geometry-cache population occurs in untimed first diagnostic policy invocation; full native deployment timing remains required.',
        holdout_scenes_consumed=0, deployment_qualified=False))
    print('Completed', identifier, 'elapsed', time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'evaluate'])
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--checkpoint', choices=['C53', 'C44', 'C37'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.config.resolve(), args.output.resolve(), args.cpu)
    else:
        if args.bank is None or args.checkpoint is None:
            parser.error('--bank and --checkpoint required for evaluation')
        evaluate(args.config.resolve(), args.bank.resolve(), args.checkpoint, args.output.resolve(), args.cpu)
