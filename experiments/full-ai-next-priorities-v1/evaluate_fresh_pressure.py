"""Evaluate frozen pressure candidates and attempt native trajectories without fallback."""
import argparse
import os
from pathlib import Path
import statistics
import time
import numpy as np
from study_support import ROOT, BASE, read, write, sha, load_checkpoint, equation_error
from fresh_pressure_data import cases
from native_client import NativeClient, FIELDS
from temporal_metrics import metrics


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh reserved evaluation output required')
    selection = read(args.selection)
    if selection['status'] != 'frozen_before_reserved_capture':
        raise ValueError('Selection must predate capture')
    frozen = next(row for row in selection['models'] if row['id'] == args.identifier)
    if sha(ROOT / frozen['entry_path']) != frozen['entry_sha256']:
        raise ValueError('Frozen model entry changed')
    bank = read(args.bank)
    if bank['selection_sha256'] != sha(args.selection):
        raise ValueError('Capture did not use this frozen selection')
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {23})
    tick = time.perf_counter()
    state = load_checkpoint(args.identifier)
    torch = state.torch
    torch.set_num_threads(1)
    setup = time.perf_counter() - tick
    inventory = state.data.helpers().state_inventory(state.network)
    predict = lambda sample: state.model.predict_pressure(state.network, sample, state.normalization)
    rows = []
    with torch.inference_mode():
        for identity, sample in cases(args.bank, torch):
            actual = predict(sample)
            if not bool(torch.isfinite(actual).all()) or float(actual[0, 0, 0, 0, 0]) != 0:
                raise ValueError('Finite pressure and cell-zero gauge required')
            reference = sample['reference'] - sample['reference'].mean()
            difference = actual - actual.mean() - reference
            rows.append(dict(identity, equation_relative_residual=equation_error(state, actual, sample),
                             reference_equation_relative_residual=equation_error(state, sample['reference'], sample),
                             pressure_relative_l2=float(torch.linalg.vector_norm(difference)) /
                             max(float(torch.linalg.vector_norm(reference)), 1e-30)))
    summaries = []
    for population in ('small', '32cubed_extension'):
        subset = [row for row in rows if row['population'] == population]
        values = np.array([row['equation_relative_residual'] for row in subset])
        summaries.append(dict(population=population, requests=len(subset),
                              median_equation_error=float(np.median(values)),
                              p95_equation_error=float(np.quantile(values, .95)),
                              worst_equation_error=float(values.max()),
                              passing_7e_4=int((values <= 7e-4).sum()),
                              worst_case=max(subset, key=lambda row: row['equation_relative_residual'])))
    write(args.output / 'pressure.json', dict(status='completed', identifier=args.identifier,
          rows=rows, summaries=summaries, bank_sha256=sha(args.bank), selection_sha256=sha(args.selection),
          model_entry=state.entry, source_sha256=sha(__file__), setup_seconds=setup,
          candidate_numerical_pressure_solve=False, optimizer_updates=0))
    print('Reserved prepared-request evaluation complete', args.identifier, summaries, flush=True)
    norm = read(BASE / 'temporal-training-view-02/normalization.json')
    episodes = []
    for row in bank['trajectories']:
        receipt_path = ROOT / row['receipt']
        receipt = read(receipt_path)
        scene = receipt['scene']
        directory = args.output / ('native-' + row['identifier'])
        tick = time.perf_counter()
        client = NativeClient(scene, directory)
        native_setup = time.perf_counter() - tick
        frames, clocks, diagnostics = [], [], []
        failure = None
        with np.load(receipt_path.parent / receipt['states'][0]['path'], allow_pickle=False) as arrays:
            initial_mass = float(arrays['tracer'].astype('f8').sum() / scene['cells_per_metre'] ** 3)
        try:
            for step in range(1, 65):
                tick = time.perf_counter()
                frame = client.advance(method='candidate', predict=predict)
                clocks.append(time.perf_counter() - tick)
                frames.append(frame)
                if not frame['frame']['advanced']:
                    failure = dict(attempted_step=step, native_frame=frame)
                    break
                if step in (12, 48, 64):
                    packet, physical = client.inspect()
                    spec = next(record for record in receipt['states'] if record['step'] == step)
                    path = receipt_path.parent / spec['path']
                    if sha(path) != spec['sha256']:
                        raise ValueError('Reserved state changed')
                    with np.load(path, allow_pickle=False) as arrays:
                        reference = {name: arrays[name].copy() for name in FIELDS}
                    case = receipt['cases'][step - 1]
                    path = receipt_path.parent / case['path']
                    if sha(path) != case['sha256']:
                        raise ValueError('Reserved equation changed')
                    with np.load(path, allow_pickle=False) as arrays:
                        rhs = arrays['rhs'].copy()
                    if packet['step'] != step:
                        raise ValueError('Native candidate timestamp mismatch')
                    diagnostics.append(dict(step=step, metrics=metrics(physical, reference, rhs, scene, norm, initial_mass)))
            custody = client.close()
        finally:
            if client.child.poll() is None:
                client.close()
        accepted = sum(bool(frame['frame']['advanced']) for frame in frames)
        result = dict(status='completed', identifier=row['identifier'], model=args.identifier,
                      counts_xyz=scene['counts'], accepted_steps=accepted, requested_steps=64,
                      completed_physical_interval=accepted == 64, failure=failure, frames=frames,
                      application_seconds=sum(clocks), per_attempt_seconds=clocks,
                      application_seconds_per_physical_second=sum(clocks) / (64 / 60) if accepted == 64 else None,
                      native_setup_seconds=native_setup, native_process=custody,
                      diagnostics=diagnostics, fallback=False, native_acceptance_gates_unchanged=True)
        write(directory / 'receipt.json', result)
        episodes.append({key: value for key, value in result.items() if key not in ('frames', 'failure', 'per_attempt_seconds', 'diagnostics')})
        print('Reserved native attempt', args.identifier, row['identifier'], 'accepted', accepted, 'of64', flush=True)
    if inventory != state.data.helpers().state_inventory(state.network):
        raise ValueError('Reserved evaluation changed frozen model weights')
    write(args.output / 'summary.json', dict(status='completed', identifier=args.identifier,
          prepared_requests=summaries, episodes=episodes, source_sha256=sha(__file__),
          bank_sha256=sha(args.bank), selection_sha256=sha(args.selection), cuda_initialized=torch.cuda.is_initialized(),
          native_accepted_complete_episodes=sum(row['completed_physical_interval'] for row in episodes),
          reserved_temporal_scenes_consumed=False, model_promotion=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifier', required=True)
    for name in ('selection', 'bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
