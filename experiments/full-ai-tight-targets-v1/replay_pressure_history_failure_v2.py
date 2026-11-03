"""Deterministic consumed-scene threshold replay; numerical overrides are OFFLINE only."""
import argparse
from pathlib import Path
import os
import time
import profile_frozen_components
from profile_frozen_components import ROOT, read, write, digest
from mixed_modal_sampler import load_model as load_base
from learned_pressure_history import LearnedPressureHistory
from recurrent_sampler_contiguous_v3 import library
import restored_recurrent_sampler as restored
from diagnose_causal_composition import FIELDS, inventory, measure
from evaluate_recurrent_sampler import identity
import numpy as np
import torch


def replay(args):
    if args.output.exists():
        raise ValueError('Fresh replay output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Registered replay evidence changed: ' + path)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    restored.library = library
    receipt_path = ROOT / reg['receipt']
    receipt = read(receipt_path)
    initial_step, stop = reg['initial_step'], reg['stop']
    if not 0 <= initial_step < stop <= 240:
        raise ValueError('Bounded actual history required')
    native_spec = receipt['states'][initial_step]
    initial = ROOT / reg['initial_state'] if reg.get('initial_state') else receipt_path.parent / native_spec['path']
    with np.load(initial, allow_pickle=False) as archive:
        current = {key: archive[key].copy() for key in FIELDS}
    if reg['pressure_role'] not in ('managed_learned', 'managed_learned_tensor', 'OFFLINE_numerical_oracle'):
        raise ValueError('Explicit pressure role required')
    override = None
    if reg['pressure_role'] == 'OFFLINE_numerical_oracle':
        from numerics import correction
        one = torch.ones(1, dtype=torch.float64)
        override = lambda q: correction(q, one, one, max_mode=None)
    if reg.get('seed_pressure_role') == 'OFFLINE_native_reference_pressure':
        source = receipt_path.parent / native_spec['path']
        if digest(source) != native_spec['sha256']:
            raise ValueError('Diagnostic current-pressure source changed')
        with np.load(source, allow_pickle=False) as archive:
            current['pressure'] = archive['pressure'].copy()
    base = load_base(reg['pair'], reg['library_path'])
    if reg['pressure_role'] == 'managed_learned_tensor':
        from sampler_precision_policies import TensorDCTPressure
        original = base.pressure_core
        original.close()
        base.pressure_core = TensorDCTPressure(original)
    model = LearnedPressureHistory(base, mode='reuse', pressure_map=override)
    before = inventory(model)
    args.output.mkdir(parents=True)
    snapshots = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for step in range(initial_step, stop):
                current = model.predict(current, receipt['scene'], step, 1)[0]
                if step + 1 in reg.get('save_own_prefix_steps', []):
                    path = args.output / f'own-state-{step+1:03d}.npz'
                    np.savez_compressed(path, **{k: v.numpy() for k, v in current.items()})
                    snapshots.append(dict(step=step+1, path=str(path.relative_to(ROOT)),
                                          sha256=digest(path), causal_own_history=True))
            spec = receipt['states'][stop]
            truth_path = receipt_path.parent / spec['path']
            if digest(truth_path) != spec['sha256']:
                raise ValueError('Endpoint reference changed')
            with np.load(truth_path, allow_pickle=False) as archive:
                truth = {key: archive[key].copy() for key in (*FIELDS, 'rhs')}
            result = measure(current, truth, receipt['scene'], read(ROOT / reg['normalization']))
            actual_identity = identity(current)
            if reg.get('expected_endpoint_identity') and actual_identity != reg['expected_endpoint_identity']:
                raise ValueError('Actual learned replay did not reproduce frozen endpoint bytes')
            if inventory(model) != before or torch.cuda.is_initialized():
                raise ValueError('Replay changed model or device')
        passed = all(v <= reg['maximum_error'] for v in result['relative'].values())
        write(args.output / 'result.json', dict(status='GREEN' if passed else 'RED',
            identifier=receipt['identifier'], pressure_role=reg['pressure_role'],
            initial_step=initial_step, stop=stop, actual_steps=stop-initial_step,
            metrics=result, maximum_error=reg['maximum_error'],
            output_identity=actual_identity, own_prefix_snapshots=snapshots,
            elapsed_s=time.perf_counter()-started, source_sha256=reg['source_sha256'],
            protocol_sha256=digest(args.protocol), optimizer_updates=0,
            diagnostic_only=True, independent_confirmation=False,
            numerical_oracle_used=override is not None, candidate_frontier_point=False,
            reference_used_only_for_endpoint_scoring=reg.get('seed_pressure_role') != 'OFFLINE_native_reference_pressure',
            seed_pressure_role=reg.get('seed_pressure_role', 'causal_own_history'), all_plan_owners_closed=True))
        print(('GREEN' if passed else 'RED'), receipt['identifier'],
              f'{initial_step}->{stop}', result['relative'], flush=True)
        return 0 if passed else 1
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parsed = parser.parse_args()
    parsed.output = parsed.output.resolve()
    raise SystemExit(replay(parsed))
