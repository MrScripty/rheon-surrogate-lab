"""Fixed-call spatial transfer diagnostics; no optimization or CFD advancement.

Use consumed manufactured grids and independently checked sealed cosine modes.
Absolute spectral energies distinguish small signals from relative-error ratios;
high-only fields measure leakage without claiming a causal noise-floor estimate.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'water-early-pressure-v1'))
import amplitude


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))) * 1024


def reserve():
    if available() < 8 * 1024 ** 3:
        raise MemoryError('eight GiB available reserve no longer present')


def energies(value, bases, cutoff=3):
    """Orthonormal cosine energy per cell, excluding the constant gauge mode."""
    field = value[0, 0] - value.mean()
    coefficients = amplitude.audit.cosine_transform(field, bases)
    coefficients[0, 0, 0] = 0
    low = float(coefficients[:cutoff + 1, :cutoff + 1, :cutoff + 1].square().sum()) / field.numel()
    total = float(field.square().mean())
    transformed = float(coefficients.square().sum()) / field.numel()
    if abs(total - transformed) > 1e-11 * max(total, 1e-300):
        raise ValueError('orthonormal spectral energy does not match spatial energy')
    return dict(total_pa2=total, low_pa2=low, remaining_pa2=max(0., total - low))


def measure():
    audit = amplitude.audit
    torch = audit.TORCH
    rows = []
    calls_selected = (1, 4, 8, 12)
    for counts in ((33, 11, 9), (16, 16, 16), (25, 16, 23)):
        shape = counts[::-1]
        axis = max(range(3), key=lambda i: shape[i])
        bases = []
        for n in shape:
            k = torch.arange(n, dtype=torch.float64)
            basis = torch.cos(torch.pi * k[:, None] * (k[None, :] + .5) / n) * (2 / n) ** .5
            basis[0] /= 2 ** .5
            bases.append(basis)
        def wave(mode):
            factors = [torch.cos(torch.pi * (mode if i == axis else 0) *
                                 (torch.arange(n, dtype=torch.float64) + .5) / n)
                       for i, n in enumerate(shape)]
            return torch.einsum('z,y,x->zyx', *factors)[None, None]
        scalar = {key: torch.tensor([value], dtype=torch.float64)
                  for key, value in [('cell_size_m', .125), ('rho', 1000.), ('dt', 1 / 60)]}
        high = wave(7)
        for low_mode in (1, 2, 3):
            low = wave(low_mode)
            low_norm = float(low.square().sum())
            for kind in ('pure-low', 'low-plus-high', 'high-only'):
                for alpha in ((1., .001, 1e-8) if kind != 'high-only' else (0.,)):
                    reserve()
                    exact = alpha * low + (high if kind != 'pure-low' else 0.)
                    exact -= exact.flatten()[0].clone()
                    rhs = audit.MODEL.full_operator(exact, scalar['cell_size_m'], scalar['rho'])
                    eigen_low = 2 - 2 * math.cos(math.pi * low_mode / shape[axis])
                    eigen_high = 2 - 2 * math.cos(math.pi * 7 / shape[axis])
                    analytic_rhs = (alpha * eigen_low * low +
                                    (eigen_high * high if kind != 'pure-low' else 0.)) * .125 / 1000
                    defect = audit.vector_norm(rhs - analytic_rhs) / max(audit.vector_norm(analytic_rhs), 1e-300)
                    if defect >= 1e-10:
                        raise ValueError('independent analytic RHS check failed')
                    signal = energies(exact, bases)
                    expected_signal = .5 * (alpha ** 2 + (1 if kind != 'pure-low' else 0))
                    if abs(signal['total_pa2'] - expected_signal) > 1e-11 * expected_signal:
                        raise ValueError('known cosine signal energy failed')
                    pressure = torch.zeros_like(exact)
                    sample = {**scalar, 'base': pressure, 'native_input_residual': rhs}
                    for calls in range(1, max(calls_selected) + 1):
                        state = {**sample, 'base': pressure, 'native_input_residual': rhs -
                                 audit.MODEL.full_operator(pressure, scalar['cell_size_m'], scalar['rho'])}
                        pressure = amplitude.predict_proposal(audit.MODEL, audit.NETWORK, state, audit.NORM, True)
                        if calls not in calls_selected:
                            continue
                        error = pressure - exact
                        error_energy = energies(error, bases)
                        gradient_error = gradient_reference = 0.
                        for dimension in (2, 3, 4):
                            gradient_error += float((torch.diff(error, dim=dimension) / .125).square().sum())
                            gradient_reference += float((torch.diff(exact, dim=dimension) / .125).square().sum())
                        equation = rhs - audit.MODEL.full_operator(pressure, scalar['cell_size_m'], scalar['rho'])
                        coefficient = float((error * low).sum()) / low_norm
                        rows.append(dict(counts_xyz=counts, low_mode=low_mode, kind=kind,
                                         low_amplitude_pa=alpha, high_amplitude_pa=0 if kind == 'pure-low' else 1.,
                                         calls=calls, signal_energy=signal, error_energy=error_energy,
                                         pressure_relative_l2=(error_energy['total_pa2'] / signal['total_pa2']) ** .5,
                                         gradient_relative_l2=(gradient_error / gradient_reference) ** .5,
                                         equation_relative_l2=audit.vector_norm(equation) / audit.vector_norm(rhs),
                                         low_coefficient_error_pa=coefficient,
                                         low_relative_amplitude_error=abs(coefficient / alpha) if alpha else None,
                                         independent_rhs_relative_defect=defect))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'artifacts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh report required')
    available_at_start = available()
    if available_at_start < 10 * 1024 ** 3:
        raise MemoryError('two GiB diagnostic accounting plus eight GiB reserve unavailable')
    started = time.perf_counter()
    sources = {str(p): digest(p) for p in (Path(__file__), Path(amplitude.__file__),
               Path(amplitude.audit.__file__), HERE.parent / 'water-stability-learning-v1/learned_iteration.py')}
    amplitude.audit.initialize(args.project.resolve(strict=True), args.artifacts.resolve(strict=True),
                               sorted(os.sched_getaffinity(0)))
    reserve()
    vm = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines()
                  if x.startswith('VmSize:'))) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (vm + 512 * 1024 ** 2,) * 2)
    with amplitude.audit.TORCH.no_grad():
        rows = measure()
    if len(rows) != 252 or amplitude.audit.TORCH.cuda.is_initialized():
        raise ValueError('incomplete fixed CPU diagnostic')
    if any(digest(path) != identity for path, identity in sources.items()):
        raise ValueError('diagnostic source changed during measurement')
    result = dict(format='pressure_spatial_transfer_v1', status='passed', rows=rows,
                  weights_sha256=digest(args.artifacts / 'weights.npz'),
                  source_witness_sha256=digest(args.project / 'source-witness.json'),
                  sources=sources, available_at_start_bytes=available_at_start,
                  calls_per_case=[1, 4, 8, 12], network_calls=756, optimizer_updates=0, simulations_advanced=0,
                  elapsed_s=time.perf_counter() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=amplitude.audit.TORCH.get_num_threads(),
                  cuda_initialized=False,
                  method='Consumed grids; stationary cosine pressure modes1/2/3 and7, cubic h=.125m. Independent analytic eigenvalue RHS and Parseval checks. Mean-free pressure and orthonormal DCT energy in Pa2 per cell. Low band: all per-axis modes<=3 excluding DC. Gradient uses internal face differences/h. Same fixed NN calls; no native solver or spectral correction. Concurrent diagnostic timings are not inference latency. High-only measures leakage; ratios alone do not identify a causal noise floor.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_s', 'peak_rss_bytes', 'network_calls')}))


if __name__ == '__main__':
    main()
