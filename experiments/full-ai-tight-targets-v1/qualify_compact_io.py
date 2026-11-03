"""Qualify compact physical codec/forcing and measure their complete wire cost.

The wire returns copies of its input state; it is not a simulation predictor.
"""
import argparse
import os
from pathlib import Path
import statistics
import time
from profile_frozen_components import ROOT, read, write, digest
from compact_state_codec import pack, unpack
from factorized_force import force_features, library
from fused_force_construction import force_features as reference_force
from numerics import fine_action
import numpy as np
import torch


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def force_flux(value):
    # Independent diagnostic divergence on compact normal-face components;
    # the omitted upper normal faces are all zero by construction.
    x, y, z = value[:, 0], value[:, 1], value[:, 2]
    from torch.nn import functional as F
    return (F.pad(x, (0, 1))[..., 1:] - x +
            F.pad(y, (0, 0, 0, 1))[:, :, 1:, :] - y +
            F.pad(z, (0, 0, 0, 0, 0, 1))[:, 1:, :, :] - z)


def run(args):
    if args.output.exists():
        raise ValueError('Fresh complete IO qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered source/artifact changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    norm = read(ROOT / reg['normalization']); scales = norm['scales']
    library()
    rows, sources = [], {}
    with torch.inference_mode():
        for row in read(ROOT / reg['bank'])['trajectories']:
            path = ROOT / row['receipt']
            if digest(path) != row['receipt_sha256']:
                raise ValueError('Consumed scene receipt changed')
            receipt = read(path); scene = receipt['scene']; nx, ny, nz = scene['counts']
            sources[str(path.relative_to(ROOT))] = digest(path)
            for step in reg['states']:
                spec = receipt['states'][step]; state = path.parent / spec['path']
                if digest(state) != spec['sha256']:
                    raise ValueError('Native current state changed')
                sources[str(state.relative_to(ROOT))] = digest(state)
                with np.load(state, allow_pickle=False) as archive:
                    fields = {name: archive[name].copy() for name in ('pressure', 'vx', 'vy', 'vz', 'tracer')}
                    rhs = torch.from_numpy(archive['rhs'].copy()).double().reshape(1, 1, nz, ny, nx)
                pressure, carrier = pack(fields, scene['counts'], scales)
                reconstructed = unpack(pressure, carrier, scene['counts'], scales)
                errors = {name: relative(reconstructed[name], torch.from_numpy(fields[name])) for name in fields}
                h = torch.tensor([1. / scene['cells_per_metre']], dtype=torch.float64)
                rho = torch.tensor([1000.], dtype=torch.float64)
                residual = relative(fine_action(reconstructed['pressure'].reshape(1, 1, nz, ny, nx), h, rho), rhs)
                if max(errors.values()) > reg['maximum_codec_relative_error'] or residual > reg['maximum_pressure_codec_residual']:
                    raise ValueError('Compact codec failed its precision control')
                for horizons in reg['horizons']:
                    expected = reference_force(scene, step, horizons, scales['velocity']).reshape(horizons, 3, nz+1, ny+1, nx+1)[:, :, :nz, :ny, :nx].contiguous()
                    actual = force_features(scene, step, horizons, scales['velocity'])
                    force_error = relative(actual, expected)
                    flux_error = relative(force_flux(actual.double()), force_flux(expected.double()))
                    if force_error > reg['maximum_force_relative_error'] or flux_error > reg['maximum_force_flux_relative_error']:
                        raise ValueError(f'Compact forcing formula failed qualification: {force_error}, {flux_error}')
                    def execute():
                        p, c = pack(fields, scene['counts'], scales)
                        force = force_features(scene, step, horizons, scales['velocity'])
                        outputs = [unpack(p, c, scene['counts'], scales) for unused in range(horizons)]
                        return outputs, force
                    for unused in range(reg['warmups']): execute()
                    times = []
                    for unused in range(reg['repeats']):
                        tick = time.perf_counter(); outputs, force = execute()
                        times.append((time.perf_counter() - tick) * 1000)
                    if any(not torch.equal(result[name], reconstructed[name]) for result in outputs for name in fields):
                        raise ValueError('Timed complete physical reconstruction changed')
                    rows.append(dict(identifier=row['identifier'], step=step, counts_xyz=scene['counts'], horizons=horizons,
                                     raw_complete_wire_block_ms=times, median_block_ms=statistics.median(times),
                                     maximum_block_ms=max(times), maximum_ms_per_output=max(times)/horizons,
                                     codec_relative_l2=errors, pressure_codec_equation_relative_l2=residual,
                                     force_relative_l2=force_error, force_flux_relative_l2=flux_error))
            print(row['identifier'], 'compact IO and forcing checks passed', flush=True)
    summaries = []
    for regime in ('small', '32cubed'):
        for horizons in reg['horizons']:
            selected = [r for r in rows if (max(r['counts_xyz']) <= 16) == (regime == 'small') and r['horizons'] == horizons]
            summaries.append(dict(regime=regime, horizons=horizons,
                                  median_block_ms=statistics.median(t for r in selected for t in r['raw_complete_wire_block_ms']),
                                  maximum_block_ms=max(r['maximum_block_ms'] for r in selected),
                                  maximum_ms_per_output=max(r['maximum_ms_per_output'] for r in selected),
                                  worst_codec_relative_l2=max(max(r['codec_relative_l2'].values()) for r in selected),
                                  worst_pressure_codec_equation_relative_l2=max(r['pressure_codec_equation_relative_l2'] for r in selected),
                                  worst_force_relative_l2=max(r['force_relative_l2'] for r in selected),
                                  worst_force_flux_relative_l2=max(r['force_flux_relative_l2'] for r in selected)))
    write(args.output, dict(status='passed_compact_codec_and_forcing_execution_only', observations=rows, summaries=summaries,
                            source_evidence=sources, source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
                            optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(), candidate_frontier_point=False,
                            model_predictions_generated=False, closed_zero_normal_velocity_walls_only=True,
                            timing_scope='Input encoding, prescribed forcing for all horizons and all complete identity-state reconstructions; no learned dynamics or pressure generation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
