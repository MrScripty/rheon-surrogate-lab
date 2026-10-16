"""Separate temporal sampling bounds from a saved-state spatial pressure solve.

CPU inference only: no trajectory advancement, optimizer or new training labels.
Changing dt while holding the residual fixed is a contract check, not a physical
resimulation at a different dt. Manufactured stationary modes test spatial error
without any temporal undersampling. They are diagnostics, never fitting inputs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-pressure-accuracy-v1'))
import diagnose
from learned_iteration import correct, equation_quality


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampling(frequency, dt):
    sample_rate = 1/dt
    return {'frequency_hz': frequency, 'dt_seconds': dt,
        'period_seconds': 1/frequency, 'samples_per_cycle': sample_rate/frequency,
        'nyquist_hz': sample_rate/2, 'below_nyquist': frequency < sample_rate/2,
        'aliased_frequency_hz': abs((frequency+sample_rate/2) % sample_rate-sample_rate/2),
        'one_cycle_observation_seconds': 1/frequency}


def measure(sample, calls):
    audit = diagnose.audit
    return correct(
        audit.MODEL, audit.NETWORK, audit.NORM, sample, calls, single_evaluation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, choices=range(1, 33), default=4)
    parser.add_argument('--dt-factors', type=float, nargs='+', default=[.25, .5, 1., 2., 4.])
    parser.add_argument('--frequencies-hz', type=float, nargs='+', default=[.1, 1., 10., 40.])
    args = parser.parse_args()
    assert all(math.isfinite(x) and x > 0 for x in args.dt_factors+args.frequencies_hz)
    assert not args.output.exists(), 'preserve existing receipts'
    started = time.perf_counter()
    study = diagnose.read(args.study)
    source = study['models'][args.model]
    audit = diagnose.audit
    audit.initialize(LAB/source['project'], LAB/source['artifacts'], sorted(os.sched_getaffinity(0)))
    # Import/init before limiting growth, allowing existing runtime libraries but
    # preventing an unexpected multi-GB diagnostic temporary.
    current = int(next(x for x in Path('/proc/self/status').read_text().splitlines()
                       if x.startswith('VmSize:')).split()[1])*1024
    resource.setrlimit(resource.RLIMIT_AS, (current+512*1024**2, current+512*1024**2))
    import numpy as np
    torch = audit.TORCH
    records = []
    first = None
    with torch.no_grad():
        for witness in study['witnesses']:
            capture_path = LAB/study['root']/(witness['name']+'-'+args.model+'-original')/'capture.json'
            capture = diagnose.read(capture_path)
            assert capture['model'] == source
            for item in (capture['samples'][0], capture['samples'][-1]):
                path = LAB/item['path']
                assert digest(path) == item['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    sample = {k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}
                first = first or sample
                baseline = measure(sample, args.calls)
                values = audit.MODEL.build_inputs(sample['native_input_residual'], sample['base'],
                    sample['cell_size_m'], sample['rho'], sample['dt'], audit.NORM)
                rhs = sample['native_input_residual']+audit.MODEL.full_operator(
                    sample['base'], sample['cell_size_m'], sample['rho'])
                variants = []
                for factor in args.dt_factors:
                    changed = {**sample, 'dt': sample['dt']*factor}
                    pressure = measure(changed, args.calls)
                    inputs = audit.MODEL.build_inputs(changed['native_input_residual'], changed['base'],
                        changed['cell_size_m'], changed['rho'], changed['dt'], audit.NORM)
                    assert torch.equal(inputs, values) and torch.equal(pressure, baseline)
                    variants.append({'dt_factor': factor, 'dt_seconds': float(changed['dt'][0]),
                        'input_bits_equal': True, 'pressure_bits_equal': True,
                        'quality': equation_quality(
                            audit.MODEL, pressure, changed, rhs, study['policies']['original'])})
                negative = measure({**sample, 'native_input_residual': 2*sample['native_input_residual']}, args.calls)
                assert not torch.equal(negative, baseline), 'negative control must detect changed data'
                records.append({'sample': item, 'counts_xyz': list(sample['base'].shape[2:][::-1]),
                    'variants': variants, 'changed_residual_negative_control_passed': True})
        stationary = []
        shape = first['base'].shape
        for axis in (2, 3, 4):
            n = shape[axis]
            view = [1]*5
            view[axis] = n
            exact = torch.cos(torch.pi*(torch.arange(n, dtype=torch.float64)+.5)/n).reshape(view).expand(shape).clone()
            exact -= exact[:, :, :1, :1, :1].clone()
            sample = {**first, 'base': torch.zeros_like(exact)}
            residual = audit.MODEL.full_operator(exact, sample['cell_size_m'], sample['rho'])
            sample['native_input_residual'] = residual
            independent = diagnose.independent_action(exact, sample)
            defect = audit.vector_norm(independent-residual)/audit.vector_norm(residual)
            assert defect < 1e-9
            pressure = measure(sample, args.calls)
            # Populate the bounded separable basis cache and verify its inverse.
            oracle = audit.spectral_correction(residual, sample)
            assert audit.vector_norm(oracle-exact)/audit.vector_norm(exact) < 1e-9
            stationary.append({'mode_zyx': [int(i == axis) for i in (2, 3, 4)],
                'temporal_frequency_hz': 0, 'independent_operator_relative_defect': defect,
                **diagnose.error_metrics(pressure, exact), **diagnose.mode_metrics(pressure, exact)})
    dt = float(first['dt'][0])
    result = {'schema': 'rheon-pressure-temporal-sampling-probe-v1',
        'scope': 'fixed saved residual/base contract check plus stationary manufactured spatial modes; no changed-dt trajectories',
        'study': str(args.study), 'study_sha256': digest(args.study), 'model': source,
        'weights_sha256': digest(LAB/source['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/source['artifacts']/'normalization.json'),
        'source_witness_sha256': digest(LAB/source['project']/'source-witness.json'),
        'source_sha256': {str(Path(__file__).relative_to(LAB)): digest(__file__),
            str(Path(diagnose.__file__).relative_to(LAB)): digest(diagnose.__file__),
            str(Path(audit.__file__).relative_to(LAB)): digest(audit.__file__)},
        'network_calls_per_solve': args.calls, 'nominal_dt_seconds': dt, 'saved_states': records,
        'stationary_spatial_modes': stationary,
        'sampling_bounds': [sampling(f, dt*factor) for factor in args.dt_factors for f in args.frequencies_hz],
        'sampling_scope': 'ideal band-limited sinusoid; Nyquist is an identifiability bound, not an integration-accuracy guarantee; one cycle is illustrative, not a universal detection limit',
        'spatial_to_temporal_relation': 'f=abs(k dot U)/(2*pi) only for a frozen pattern advected at known constant U; no universal conversion from wavelength alone',
        'optimizer_updates': 0, 'simulations_advanced': 0, 'cuda_initialized': torch.cuda.is_initialized(),
        'cpu_threads': torch.get_num_threads(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds': time.perf_counter()-started}
    assert not result['cuda_initialized']
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'saved_states': len(records), 'dt_variants_per_state': len(args.dt_factors),
        'all_predictions_bitwise_equal': True, 'stationary_modes': stationary,
        'elapsed_seconds': result['elapsed_seconds'], 'peak_rss_bytes': result['peak_rss_bytes']}))


if __name__ == '__main__':
    main()
