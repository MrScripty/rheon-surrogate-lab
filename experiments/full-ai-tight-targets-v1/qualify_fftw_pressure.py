"""Finite execution/precision checks for an untrained double transform backend."""
import argparse
import ctypes
import os
from pathlib import Path
import statistics
import time
from profile_frozen_components import ROOT, read, write, digest
from preflight_global_pressure import bases, transform
from fftw_pressure_transform import DCTMultiplier
from numerics import fine_action
import numpy as np
import torch


def run(args):
    if args.output.exists():
        raise ValueError('Fresh transform qualification required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered inference source changed: ' + path)
    if digest(reg['library_path']) != reg['library_sha256']:
        raise ValueError('Registered serial FFTW binary changed')
    if read(ROOT / reg['preceding_preflight'])['status'] != 'completed_fixed_transform_and_precision_preflight':
        raise ValueError('Preceding numerical job must complete first')
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    generator = torch.Generator().manual_seed(reg['diagnostic_seed'])
    rows, setup, sources, owners = [], [], {}, {}
    with torch.inference_mode():
        try:
            for row in read(ROOT / reg['bank'])['trajectories']:
                receipt_path = ROOT / row['receipt']
                if digest(receipt_path) != row['receipt_sha256']:
                    raise ValueError('Consumed scene receipt changed')
                receipt = read(receipt_path); scene = receipt['scene']
                sources[str(receipt_path.relative_to(ROOT))] = digest(receipt_path)
                spatial = tuple(reversed(scene['counts']))
                axes = bases(spatial, torch.float64)
                h = torch.tensor([1. / scene['cells_per_metre']], dtype=torch.float64)
                rho = torch.tensor([1000.], dtype=torch.float64)
                for step in reg['states']:
                    spec = receipt['states'][step]; path = receipt_path.parent / spec['path']
                    if digest(path) != spec['sha256']:
                        raise ValueError('Native archive changed')
                    sources[str(path.relative_to(ROOT))] = digest(path)
                    with np.load(path, allow_pickle=False) as state:
                        truth = torch.from_numpy(state['pressure'].copy()).double().reshape(1, 1, *spatial)
                        rhs = torch.from_numpy(state['rhs'].copy()).double().reshape(1, 1, *spatial)
                    for horizons in reg['horizons']:
                        shape = (horizons, 1, *spatial)
                        if shape not in owners:
                            tick = time.perf_counter()
                            owners[shape] = DCTMultiplier(shape, reg['library_path'])
                            setup.append(dict(shape=shape, seconds=time.perf_counter() - tick))
                        owner = owners[shape]
                        value = truth.expand(shape).contiguous()
                        gains = torch.ones((1, 1, *spatial), dtype=torch.float64)
                        def execute():
                            output = owner(value, gains)
                            return (output - output[:, :, :1, :1, :1]).contiguous()
                        before = value.clone()
                        for unused in range(reg['warmups']): execute()
                        elapsed = []
                        for unused in range(reg['repeats']):
                            tick = time.perf_counter(); output = execute()
                            elapsed.append((time.perf_counter() - tick) * 1000)
                        residual = float((fine_action(output[:1], h, rho) - rhs).norm() / rhs.norm())
                        # Compare nontrivial finite diagonal multiplication,
                        # not only an identity round trip. These random
                        # diagnostic coefficients are neither fitted nor an
                        # analytic pressure inverse.
                        random_gain = torch.randn((1, 1, *spatial), generator=generator, dtype=torch.float64)
                        control = transform(transform(value, axes) * random_gain, axes, inverse=True)
                        actual = owner(value, random_gain)
                        relative = float((actual - control).norm() / control.norm().clamp_min(1e-30))
                        if not torch.equal(value, before):
                            raise ValueError('Out-of-place transform modified input')
                        if relative > reg['maximum_transform_relative_error'] or residual > reg['maximum_identity_control_residual']:
                            raise ValueError(f'Transform precision qualification failed: {relative}, {residual}')
                        rows.append(dict(identifier=row['identifier'], state=step, counts_xyz=scene['counts'], horizons=horizons,
                                         raw_complete_primitive_block_ms=elapsed, median_block_ms=statistics.median(elapsed),
                                         maximum_block_ms=max(elapsed), maximum_ms_per_output=max(elapsed) / horizons,
                                         identity_equation_relative_l2=residual, nontrivial_multiplier_relative_l2=relative,
                                         input_unchanged=True))
                print(row['identifier'], 'FFTW checks passed', flush=True)
        finally:
            for owner in owners.values(): owner.close()
    summaries = []
    for regime in ('small', '32cubed'):
        for horizons in reg['horizons']:
            selected = [r for r in rows if (max(r['counts_xyz']) <= 16) == (regime == 'small') and r['horizons'] == horizons]
            summaries.append(dict(regime=regime, horizons=horizons,
                                  median_block_ms=statistics.median(t for r in selected for t in r['raw_complete_primitive_block_ms']),
                                  maximum_block_ms=max(r['maximum_block_ms'] for r in selected),
                                  maximum_ms_per_output=max(r['maximum_ms_per_output'] for r in selected),
                                  worst_identity_equation_relative_l2=max(r['identity_equation_relative_l2'] for r in selected),
                                  worst_nontrivial_multiplier_relative_l2=max(r['nontrivial_multiplier_relative_l2'] for r in selected)))
    library = ctypes.CDLL(reg['library_path'])
    version = ctypes.string_at(ctypes.addressof(ctypes.c_char.in_dll(library, 'fftw_version'))).decode()
    write(args.output, dict(status='passed_transform_execution_and_precision_only', observations=rows, summaries=summaries,
                            setup=setup, source_evidence=sources, source_sha256=reg['source_sha256'],
                            protocol_sha256=digest(args.protocol), library_path=reg['library_path'],
                            library_sha256=reg['library_sha256'], library_version=version, thread_api_initialized=False,
                            inference_parameters_fitted=0, optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(),
                            all_plan_owners_closed=True, reference_pressure_used_offline=True, candidate_frontier_point=False,
                            timing_scope='Complete transform/multiplier/inverse, finite/alignment checks and pressure materialization only; excludes forcing and other model work'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
