"""Isolate frozen pressure and codec errors using consumed native RHS labels.

This is an offline teacher-input diagnostic, never causal candidate inference.
No pressure inverse, optimizer, or numerical reference solve is executed.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_frozen_components import read, write, digest
import numpy as np
import torch
from application_model_support import load_model
from temporal_train_displacement import inventory
from numerics import fine_action


def score(pressure, truth, rhs, scene):
    h = torch.tensor([1. / scene['cells_per_metre']], dtype=torch.float64)
    rho = torch.tensor([1000.], dtype=torch.float64)
    action = fine_action(pressure, h, rho)
    p, t = pressure - pressure.mean(), truth - truth.mean()
    return dict(equation_relative_l2=float((action - rhs).norm() / rhs.norm().clamp_min(1e-30)),
                mean_free_pressure_relative_l2=float((p - t).norm() / t.norm().clamp_min(1e-30)),
                pressure_absolute_rms=float((p - t).square().mean().sqrt()),
                finite=bool(torch.isfinite(pressure).all()))


def run(args):
    if args.output.exists():
        raise ValueError('Fresh offline diagnosis required')
    protocol = read(args.protocol)
    for path, expected in protocol['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered source changed: ' + path)
    profile = read(ROOT / protocol['completed_profile'])
    if profile['status'] != 'completed_diagnostic_only' or not profile['predictions_bitwise_unchanged']:
        raise ValueError('Finish the isolated profile before this numerical diagnostic')
    os.sched_setaffinity(0, {protocol['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    method = next(row for row in read(ROOT / protocol['models'])['methods'] if row['identifier'] == protocol['identifier'])
    network, norm, identity = load_model(method['model'])
    before = inventory(network)
    cases, sources = [], {}
    with torch.inference_mode():
        for row in read(ROOT / protocol['bank'])['trajectories']:
            path = ROOT / row['receipt']
            if digest(path) != row['receipt_sha256']:
                raise ValueError('Consumed scene receipt changed')
            receipt = read(path)
            sources[str(path.relative_to(ROOT))] = digest(path)
            scene = receipt['scene']
            shape = (1, 1, *reversed(scene['counts']))
            for step in protocol['states']:
                entry = receipt['states'][step]
                state = path.parent / entry['path']
                if digest(state) != entry['sha256']:
                    raise ValueError('Native field/RHS archive changed')
                sources[str(state.relative_to(ROOT))] = digest(state)
                with np.load(state, allow_pickle=False) as source:
                    truth = torch.from_numpy(source['pressure'].copy()).double().reshape(shape)
                    rhs = torch.from_numpy(source['rhs'].copy()).double().reshape(shape)
                if not bool(rhs.norm() > 0) or not bool(truth.norm() > 0):
                    raise ValueError('Registered nonzero reference diagnostic expected')
                q = rhs * (1000. * scene['cells_per_metre'])
                learned = network.pressure_core(q)
                # The deployed head normalizes/casts then unpacks in float32.
                # Preserve those two separate roundings and cell-zero gauge.
                scale = norm['scales']['pressure']
                def codec(value):
                    result = ((value / scale).float() * scale).double()
                    return result - result[:, :, :1, :1, :1]
                policies = dict(native_reference_control=truth,
                                native_reference_through_current_codec=codec(truth),
                                frozen_learned_pressure_on_native_rhs=learned,
                                frozen_learned_pressure_and_current_codec=codec(learned))
                values = {name: score(p, truth, rhs, scene) for name, p in policies.items()}
                if values['native_reference_control']['equation_relative_l2'] > protocol['maximum_reference_control_residual']:
                    raise ValueError('Teacher/operator units or precision failed qualification')
                cases.append(dict(identifier=row['identifier'], counts_xyz=scene['counts'],
                                  cells_per_metre=scene['cells_per_metre'], state=step, policies=values))
                print(row['identifier'], step, values['frozen_learned_pressure_on_native_rhs'], flush=True)
    if before != inventory(network) or torch.cuda.is_initialized():
        raise ValueError('Frozen diagnostic changed tensors or initialized CUDA')
    summaries = []
    for regime in ('small', '32cubed'):
        selected = [row for row in cases if (max(row['counts_xyz']) <= 16) == (regime == 'small')]
        for policy in cases[0]['policies']:
            worst = {field: max(row['policies'][policy][field] for row in selected)
                     for field in ('equation_relative_l2', 'mean_free_pressure_relative_l2', 'pressure_absolute_rms')}
            summaries.append(dict(regime=regime, policy=policy, cases=len(selected), worst=worst,
                                  equation_and_pressure_below_1e5=(worst['equation_relative_l2'] <= 1e-5 and worst['mean_free_pressure_relative_l2'] <= 1e-5)))
    write(args.output, dict(status='completed_offline_teacher_input_diagnostic', model=identity,
                            cases=cases, summaries=summaries, source_evidence=sources,
                            protocol_sha256=digest(args.protocol), source_sha256=protocol['source_sha256'],
                            teacher_future_rhs_used=True, causal_candidate_inference=False,
                            numerical_pressure_inverse_executed=False, optimizer_updates=0,
                            weights_unchanged=True, cuda_initialized=False, fresh_holdout_consumed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
