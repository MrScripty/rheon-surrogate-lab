"""Finite zero-update diagnosis of the completed full-state CPU generators.

Inclusive instrumentation costs are diagnostic, not candidate benchmark scores.
Only already-consumed development states are read. No teacher future is an input.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/full-ai-next-priorities-v1'
sys.path.insert(0, str(OLD))

import numpy as np
import torch
import bundle_model_fast_preparation as model_module
from application_model_support import load_model
from temporal_train_displacement import inventory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    if args.output.exists():
        raise ValueError('A fresh diagnostic output is required')
    protocol = read(args.protocol)
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Registered source changed: {name}')
    final = read(ROOT / protocol['preceding_publication'])
    if final['status'] != 'completed' or not all(x['reaped'] and x['returncode'] == 0 for x in final['commands']):
        raise ValueError('Preceding publication must be complete and reaped')
    if not read(ROOT / protocol['preceding_audit'])['completion_proven']:
        raise ValueError('Completed-study proof required')
    os.sched_setaffinity(0, {protocol['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    sources = {}
    for row in read(ROOT / protocol['bank'])['trajectories']:
        path = ROOT / row['receipt']
        if digest(path) != row['receipt_sha256']:
            raise ValueError('Development receipt changed')
        sources[str(path.relative_to(ROOT))] = digest(path)
        receipt = read(path)
        for step in protocol['starts']:
            spec = receipt['states'][step]
            state = path.parent / spec['path']
            if digest(state) != spec['sha256']:
                raise ValueError('Consumed development state changed')
            sources[str(state.relative_to(ROOT))] = digest(state)
            with np.load(state, allow_pickle=False) as values:
                fields = {name: values[name].copy() for name in ('pressure', 'vx', 'vy', 'vz', 'tracer')}
            cases.append(dict(identifier=row['identifier'], scene=receipt['scene'], step=step, fields=fields))
    registry = read(ROOT / protocol['models'])['methods']
    results = []
    for identifier in protocol['identifiers']:
        method = next(row for row in registry if row['identifier'] == identifier)
        print('Loading', identifier, flush=True)
        tick = time.perf_counter()
        network, norm, identity = load_model(method['model'])
        setup = time.perf_counter() - tick
        before = inventory(network)
        horizons = network.horizons
        names = ('fluid', 'local_update', 'displacement.context', 'displacement.curvature',
                 'viscous_head', 'flux', 'pressure_core', 'pressure_core.source',
                 'pressure_core.source.fine', 'momentum', 'network')
        records = []
        handles = []
        clocks, elapsed, counts = {}, {}, {}

        def begin(name):
            def hook(*unused):
                clocks.setdefault(name, []).append(time.perf_counter())
            return hook

        def finish(name):
            def hook(*unused):
                elapsed[name] = elapsed.get(name, 0.) + time.perf_counter() - clocks[name].pop()
                counts[name] = counts.get(name, 0) + 1
            return hook

        def observe_function(name, function):
            def observed(*positional, **keywords):
                tick = time.perf_counter()
                try:
                    return function(*positional, **keywords)
                finally:
                    elapsed[name] = elapsed.get(name, 0.) + time.perf_counter() - tick
                    counts[name] = counts.get(name, 0) + 1
            return observed

        def predict(case):
            return network.predict(case['fields'], case['scene'], case['step'], norm)

        original_displacement = model_module.displacement
        original_readout = model_module.convex_field_readout
        with torch.inference_mode():
            expected = [predict(case) for case in cases]
            for name in names:
                module = network if name == 'network' else network.get_submodule(name)
                handles.extend((module.register_forward_pre_hook(begin(name)), module.register_forward_hook(finish(name))))
            model_module.displacement = observe_function('displacement_function', original_displacement)
            model_module.convex_field_readout = observe_function('field_readout', original_readout)
            try:
                for case, reference in zip(cases, expected):
                    large = max(case['scene']['counts']) > 16
                    for repeat in range(protocol['repeats']):
                        clocks.clear(); elapsed.clear(); counts.clear()
                        tick = time.perf_counter()
                        output = predict(case)
                        full = time.perf_counter() - tick
                        if len(output) != horizons or any(not torch.equal(a[key], b[key]) for a, b in zip(output, reference) for key in a):
                            raise ValueError('Instrumentation changed physical predictions')
                        expected_calls = {name: horizons for name in names}
                        expected_calls.update(network=1, fluid=1, displacement_function=horizons, field_readout=horizons)
                        expected_calls['pressure_core.source'] = horizons if large else 0
                        expected_calls['pressure_core.source.fine'] = 2 * horizons if large else 0
                        if any(counts.get(name, 0) != number for name, number in expected_calls.items()):
                            raise ValueError(f'Unexpected component calls: {counts}; expected {expected_calls}')
                        if any(clocks.values()) or not math.isfinite(full):
                            raise ValueError('Unmatched instrumentation or nonfinite clock')
                        records.append(dict(identifier=case['identifier'], step=case['step'], counts_xyz=case['scene']['counts'],
                                            repeat=repeat, complete_profiled_ms=full * 1000,
                                            preparation_and_reconstruction_ms=(full - elapsed['network']) * 1000,
                                            inclusive_module_ms={name: elapsed.get(name, 0.) * 1000 for name in expected_calls},
                                            module_calls={name: counts.get(name, 0) for name in expected_calls}))
            finally:
                model_module.displacement = original_displacement
                model_module.convex_field_readout = original_readout
                for handle in handles:
                    handle.remove()
        if inventory(network) != before or torch.cuda.is_initialized():
            raise ValueError('Weights changed or CUDA initialized')
        summaries = []
        for regime in ('small', '32cubed'):
            selected = [r for r in records if (max(r['counts_xyz']) <= 16) == (regime == 'small')]
            medians = {name: statistics.median(r['inclusive_module_ms'][name] for r in selected) for name in records[0]['inclusive_module_ms']}
            summaries.append(dict(regime=regime, observations=len(selected),
                                  complete_profiled_median_block_ms=statistics.median(r['complete_profiled_ms'] for r in selected),
                                  inclusive_module_median_block_ms=medians,
                                  preparation_and_reconstruction_median_ms=statistics.median(r['preparation_and_reconstruction_ms'] for r in selected)))
        results.append(dict(identifier=identifier, horizons=horizons, model=identity, setup_seconds=setup,
                            summaries=summaries, observations=records, predictions_bitwise_unchanged=True,
                            weights_unchanged=True, optimizer_updates=0))
        print(identifier, json.dumps(summaries), flush=True)
        write(args.output.with_name(args.output.stem + '-progress.json'), dict(status='running', completed_models=len(results), results=results))
        del network
    write(args.output, dict(status='completed_diagnostic_only', results=results, source_evidence=sources,
                            protocol_sha256=digest(args.protocol), source_sha256=protocol['source_sha256'],
                            cpu=protocol['cpu'], threads=1, interop_threads=1, batch_size=1,
                            optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(),
                            predictions_bitwise_unchanged=True, weights_unchanged=True,
                            inclusive_times_not_additive=True, profile_overhead_is_not_benchmark_latency=True,
                            candidate_frontier_point=False, fresh_holdout_consumed=False))
    print('Completed frozen component diagnosis; no fitting or candidate qualification', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
