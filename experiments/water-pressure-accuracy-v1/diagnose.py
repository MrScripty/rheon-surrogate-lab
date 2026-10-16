"""Captured native inputs and independent pressure-error/call frontiers on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-residual-learning-v1'))
import audit
import rollout
sys.path.insert(0, str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import equation_quality, predict_proposal, stopping_policy


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(config, model, witness, output, fixture=False, policy_name='original'):
    import numpy as np
    output = output.absolute()
    scene = dict(witness['scene'])
    wanted = witness['capture_steps']
    if fixture:
        scene['steps'] = scene['warmup_steps']+1
        wanted = [scene['steps']]
    samples = []
    original = rollout.correct
    step = scene['warmup_steps']

    def observe(*values, **options):
        nonlocal step
        step += 1
        if step in wanted:
            path = output/f'sample-{step:03d}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in values[3].items()})
            samples.append({'step': step, 'path': str(path.relative_to(LAB)), 'sha256': digest(path)})
        return original(*values, **options)

    output.mkdir(parents=True, exist_ok=False)
    rollout.correct = observe
    started = time.perf_counter()
    try:
        receipt = rollout.run(LAB/config['native_binary'], scene, output/'native',
            LAB/model['project'], LAB/model['artifacts'], timeout=config['native_timeout_seconds'],
            proposals=config['maximum_calls'], stopping=stopping_policy(config['policies'][policy_name]),
            single_evaluation=True)
    finally:
        rollout.correct = original
    assert receipt['native_process']['reaped'] and receipt['native_process']['returncode'] == 0
    assert [r['step'] for r in samples] == wanted
    result = {'samples': samples, 'scene': scene, 'model': model, 'policy': policy_name,
        'native_receipt_sha256': digest(output/'native/receipt.json'),
        'elapsed_seconds': time.perf_counter()-started, 'source_sha256': digest(__file__)}
    write(output/'capture.json', result)
    return result


def independent_action(delta, sample):
    from importlib.util import module_from_spec, spec_from_file_location
    import numpy as np
    spec = spec_from_file_location('pressure_neighbor_oracle', LAB/'experiments/global-pressure-v1/verify_capture.py')
    oracle = module_from_spec(spec)
    spec.loader.exec_module(oracle)
    h = float(sample['cell_size_m'][0])
    values = oracle.full_action(delta.flatten().tolist(), list(delta.shape[2:][::-1]), [h]*3, float(sample['rho'][0]))
    return audit.TORCH.from_numpy(np.asarray(values, dtype='<f8')).reshape_as(delta)


def error_metrics(pressure, exact):
    torch = audit.TORCH
    error = pressure-exact
    centered = error-error.mean()
    reference = exact-exact.mean()
    gradient_error = sum(float(torch.diff(error, dim=a).square().sum()) for a in (2, 3, 4))
    gradient_reference = sum(float(torch.diff(exact, dim=a).square().sum()) for a in (2, 3, 4))
    return {'pressure_relative_l2': audit.vector_norm(error)/max(audit.vector_norm(exact), 1e-30),
        'mean_free_pressure_relative_l2': audit.vector_norm(centered)/max(audit.vector_norm(reference), 1e-30),
        'gradient_relative_l2': (gradient_error/max(gradient_reference, 1e-60))**.5,
        'pressure_max_abs_pa': float(error.abs().amax()), 'mean_error_pa': float(error.mean())}


def mode_metrics(pressure, exact):
    torch = audit.TORCH
    (a, b, c), _ = audit.BASES[pressure.shape[2:]]
    error = (pressure-exact)[0, 0]
    coefficients = audit.cosine_transform(error-error.mean(), (a, b, c))
    energy = coefficients.square()
    total = float(energy.sum())
    indices = torch.topk(energy.flatten(), min(5, energy.numel())).indices.tolist()
    nz, ny, nx = energy.shape
    return {'low_modes_per_axis': 3, 'low_mode_error_energy_fraction': float(energy[:3, :3, :3].sum())/max(total, 1e-60),
        'top_modes_zyx': [{'mode': [i//(ny*nx), i//nx % ny, i % nx],
            'energy_fraction': float(energy.flatten()[i])/max(total, 1e-60)} for i in indices]}


def replay(config, model, sample_path, output):
    import numpy as np
    sample_path = sample_path.absolute()
    started = time.perf_counter()
    audit.initialize(LAB/model['project'], LAB/model['artifacts'], sorted(os.sched_getaffinity(0)))
    torch = audit.TORCH
    with np.load(sample_path, allow_pickle=False) as arrays:
        sample = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    base = sample['base']
    residual = sample['native_input_residual']
    compatible = residual-residual.mean()
    delta = audit.spectral_correction(residual, sample)
    exact = base+delta
    oracle_error = audit.vector_norm(independent_action(delta, sample)-compatible)/max(audit.vector_norm(compatible), 1e-30)
    assert oracle_error < 1e-9 and float(exact.flatten()[0]) == 0
    rhs = residual+audit.MODEL.full_operator(base, sample['cell_size_m'], sample['rho'])
    pressures = [base]
    frontier = []
    inference_seconds = 0.
    with torch.no_grad():
        for calls in range(config['maximum_calls']+1):
            pressure = pressures[-1]
            if calls:
                correction_start = time.perf_counter()
                remaining = residual-audit.MODEL.full_operator(pressure-base, sample['cell_size_m'], sample['rho'])
                pressure = predict_proposal(audit.MODEL, audit.NETWORK,
                    {**sample, 'base': pressure, 'native_input_residual': remaining}, audit.NORM, True)
                inference_seconds += time.perf_counter()-correction_start
                pressures.append(pressure)
            qualities = {name: equation_quality(audit.MODEL, pressure, sample, rhs, policy)
                         for name, policy in config['policies'].items()}
            frontier.append({'calls': calls, 'correction_seconds_cumulative': inference_seconds,
                **error_metrics(pressure, exact), **mode_metrics(pressure, exact), 'quality': qualities})
    selected = {}
    for name in config['policies']:
        first = next((row for row in frontier if row['quality'][name]['equation_passed']), None)
        selected[name] = first
    result = {'schema': 'rheon-pressure-accuracy-replay-v1', 'sample': str(Path(sample_path).relative_to(LAB)),
        'sample_sha256': digest(sample_path), 'model': model,
        'weights_sha256': digest(LAB/model['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/model['artifacts']/'normalization.json'),
        'source_witness_sha256': digest(LAB/model['project']/'source-witness.json'),
        'counts_xyz': list(base.shape[2:][::-1]), 'oracle_relative_error': oracle_error,
        'removed_constant_over_initial_residual': audit.vector_norm(residual-compatible)/max(audit.vector_norm(residual), 1e-30),
        'rounded_full_pressure': error_metrics(exact.float().double(), exact),
        'negative_zero_correction': error_metrics(base, exact),
        'selected': selected, 'frontier': frontier, 'optimizer_updates': 0,
        'cuda_initialized': torch.cuda.is_initialized(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cpu_threads': torch.get_num_threads(), 'elapsed_seconds': time.perf_counter()-started,
        'source_sha256': digest(__file__), 'scope': 'same supplied RHS exact sealed-Neumann solve; not native reference-state pressure or autonomous trajectory accuracy'}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'replay'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--witness', type=int)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--policy', default='original')
    args = parser.parse_args()
    config = read(args.config)
    model = config['models'][args.model]
    if args.mode == 'capture':
        result = capture(config, model, config['witnesses'][args.witness], args.output, args.fixture, args.policy)
        print({'samples': len(result['samples']), 'elapsed_seconds': result['elapsed_seconds']})
    else:
        result = replay(config, model, args.sample, args.output)
        print({'selected_calls': {k: v['calls'] if v else None for k, v in result['selected'].items()},
            'original_pressure_error': result['selected']['original']['pressure_relative_l2'],
            'oracle_relative_error': result['oracle_relative_error'], 'elapsed_seconds': result['elapsed_seconds']})


if __name__ == '__main__':
    main()
