"""Measure fixed transform costs and pressure precision; no candidate fitting.

All transform multipliers here are identity controls, not learned predictors or
pressure inverses. Native labels qualify representations offline only.
"""
import argparse
import math
import os
from pathlib import Path
import statistics
import time

from profile_frozen_components import ROOT, read, write, digest
import numpy as np
import torch
from numerics import fine_action


def bases(shape, dtype):
    values = []
    for n in shape:
        modes = torch.arange(n, dtype=torch.float64)[:, None]
        positions = torch.arange(n, dtype=torch.float64)[None, :] + .5
        basis = torch.cos(math.pi / n * modes * positions) * math.sqrt(2. / n)
        basis[0] /= math.sqrt(2.)
        values.append(basis.to(dtype))
    return tuple(values)


def transform(value, axes, inverse=False):
    # Batched channels, real orthonormal DCT-II at cell centres. Each complete
    # transform traverses all three axes; no modes are truncated.
    for axis, basis in zip((2, 3, 4), axes):
        matrix = basis if inverse else basis.T
        value = (value.movedim(axis, -1) @ matrix).movedim(-1, axis)
    return value


def run(args):
    if args.output.exists():
        raise ValueError('A fresh primitive preflight is required')
    protocol = read(args.protocol)
    for path, expected in protocol['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered source changed: ' + path)
    if read(ROOT / protocol['completed_diagnosis'])['status'] != 'completed_offline_teacher_input_diagnostic':
        raise ValueError('Earlier numerical diagnosis must finish before timing')
    os.sched_setaffinity(0, {protocol['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, sources = [], {}
    with torch.inference_mode():
        for scene_row in read(ROOT / protocol['bank'])['trajectories']:
            path = ROOT / scene_row['receipt']
            if digest(path) != scene_row['receipt_sha256']:
                raise ValueError('Development receipt changed')
            receipt = read(path)
            shape = tuple(reversed(receipt['scene']['counts']))
            scene = receipt['scene']
            sources[str(path.relative_to(ROOT))] = digest(path)
            h = torch.tensor([1. / scene['cells_per_metre']], dtype=torch.float64)
            rho = torch.tensor([1000.], dtype=torch.float64)
            axes_by_dtype = {torch.float32: bases(shape, torch.float32), torch.float64: bases(shape, torch.float64)}
            for step in protocol['states']:
                entry = receipt['states'][step]
                state = path.parent / entry['path']
                if digest(state) != entry['sha256']:
                    raise ValueError('Native archive changed')
                sources[str(state.relative_to(ROOT))] = digest(state)
                with np.load(state, allow_pickle=False) as source:
                    truth = torch.from_numpy(source['pressure'].copy()).double().reshape(1, 1, *shape)
                    rhs = torch.from_numpy(source['rhs'].copy()).double().reshape(1, 1, *shape)
                scale = protocol['pressure_scale']
                typed = (truth / scale) * scale
                typed = typed - typed[:, :, :1, :1, :1]
                typed_error = float((fine_action(typed, h, rho) - rhs).norm() / rhs.norm())
                if typed_error > protocol['typed_pressure_control_residual_limit']:
                    raise ValueError('Separate float64 pressure representation failed its control')
                for dtype in (torch.float32, torch.float64):
                    axes = axes_by_dtype[dtype]
                    for horizons in protocol['horizons']:
                        # Same actual full spatial tensor for every output.
                        # Cloning this control is setup; output reconstruction
                        # and materialization are included in each timed call.
                        value = truth.to(dtype).expand(horizons, 1, *shape).contiguous()
                        def execute():
                            coefficients = transform(value, axes)
                            output = transform(coefficients, axes, inverse=True)
                            physical = ((output / scale) * scale).double()
                            return (physical - physical[:, :, :1, :1, :1]).contiguous()
                        for unused in range(protocol['warmups']):
                            execute()
                        times = []
                        for unused in range(protocol['repeats']):
                            tick = time.perf_counter()
                            output = execute()
                            times.append((time.perf_counter() - tick) * 1000)
                        error = float((fine_action(output[:1], h, rho) - rhs).norm() / rhs.norm())
                        pressure_error = float(((output[:1] - output[:1].mean()) - (truth - truth.mean())).norm() / (truth - truth.mean()).norm())
                        rows.append(dict(identifier=scene_row['identifier'], counts_xyz=scene['counts'], state=step,
                                         dtype=str(dtype), horizons=horizons, raw_complete_primitive_block_ms=times,
                                         median_block_ms=statistics.median(times), maximum_block_ms=max(times),
                                         maximum_ms_per_output=max(times) / horizons,
                                         equation_relative_l2=error, mean_free_pressure_relative_l2=pressure_error,
                                         separate_float64_pressure_codec_equation_relative_l2=typed_error))
            print(scene_row['identifier'], 'completed fixed-transform/precision controls', flush=True)
    summaries = []
    for regime in ('small', '32cubed'):
        for dtype in ('torch.float32', 'torch.float64'):
            for horizons in protocol['horizons']:
                selected = [r for r in rows if (max(r['counts_xyz']) <= 16) == (regime == 'small') and r['dtype'] == dtype and r['horizons'] == horizons]
                summaries.append(dict(regime=regime, dtype=dtype, horizons=horizons,
                                      median_block_ms=statistics.median(t for row in selected for t in row['raw_complete_primitive_block_ms']),
                                      maximum_block_ms=max(r['maximum_block_ms'] for r in selected),
                                      maximum_ms_per_output=max(r['maximum_ms_per_output'] for r in selected),
                                      worst_equation_relative_l2=max(r['equation_relative_l2'] for r in selected),
                                      worst_mean_free_pressure_relative_l2=max(r['mean_free_pressure_relative_l2'] for r in selected)))
    write(args.output, dict(status='completed_fixed_transform_and_precision_preflight', observations=rows,
                            summaries=summaries, source_evidence=sources, protocol_sha256=digest(args.protocol),
                            source_sha256=protocol['source_sha256'], optimizer_updates=0, model_fitted=False,
                            multiplier='Identity only; no learned or analytic inverse coefficients',
                            teacher_pressure_used=True, causal_candidate_inference=False, candidate_frontier_point=False,
                            timing_scope='Full 3-axis transform, inverse, typed pressure reconstruction and materialization only; excludes forcing, encoding, learned dynamics and velocity/tracer outputs',
                            cuda_initialized=torch.cuda.is_initialized(), cpu=protocol['cpu'], threads=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
