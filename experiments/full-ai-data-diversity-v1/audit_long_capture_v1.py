"""Audit actual long references, all temporal labels and late fine-state replay."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from fine_reference_client_v2 import FineReferenceClient
from capture_memory_telemetry_v1 import process_memory
from recurrent_sampler_native import dimensions

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
LABELS = ('rhs', 'native_input_residual', 'pressure_base')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh complete-reference audit directory required')
    reg = read(args.protocol)
    for name, h in reg['source_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('Frozen audit dependency changed: ' + name)
    bank_path = ROOT / reg['capture'] / 'bank.json'
    bank = read(bank_path)
    if bank['status'] not in ('completed', 'completed_with_reference_rejections'):
        raise ValueError('Actual terminal capture required; do not audit a live writer')
    if len(bank['completed']) + len(bank['reference_rejections']) != reg['expected_master_attempts']:
        raise ValueError('Incomplete registered pilot attempts')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', cases=[], native_processes=[], replayed_fine_updates=0,
        bank_sha256=digest(bank_path), source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), cpu_model_forward=False, learned_forwards=0,
        new_parameter_fits=0, reserved52_references_opened=0, full_generator_target_pass=False)

    def bounded():
        if (time.monotonic() - started > reg['maximum_wall_seconds']
                or process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes']):
            raise ValueError('Registered audit resource budget exceeded')

    try:
        import numerics
        for record in bank['completed']:
            path = ROOT / record['receipt']
            if digest(path) != record['receipt_sha256']:
                raise ValueError('Completed reference receipt changed')
            receipt = read(path)
            scene = receipt['scene']
            counts = tuple(scene['counts'])
            sizes = dimensions(counts)
            if (receipt['status'] != 'completed' or receipt['actual_reference_updates'] != 4800
                    or receipt['physical_duration_s'] != 20 or len(receipt['frames']) != 4800
                    or receipt['reference_dt_s'] != 1 / 240):
                raise ValueError('A full actual20second reference was not captured')
            if any(not f['frame']['advanced'] or not f['frame']['quality']['qualified']
                   or f['frame']['step'] != i for i, f in enumerate(receipt['frames'], 1)):
                raise ValueError('Fine trajectory contains skipped/rejected steps')
            states = {s['fine_step']: s for s in receipt['states']}
            if len(states) != len(receipt['states']):
                raise ValueError('Duplicate saved fine-state index')

            def load(step):
                spec = states[step]
                source = path.parent / spec['path']
                if digest(source) != spec['sha256'] or spec['physical_time_s'] != step / 240:
                    raise ValueError('Reference snapshot identity or time differs')
                with np.load(source, allow_pickle=False) as archive:
                    arrays = {n: archive[n].copy() for n in archive.files}
                if set(arrays) != set((*FIELDS, *LABELS)):
                    raise ValueError('Complete exact snapshot inventory required')
                for n, a in arrays.items():
                    d = spec['arrays'][n]
                    if (a.dtype.str != d['dtype'] or list(a.shape) != d['shape']
                            or hashlib.sha256(a.tobytes()).hexdigest() != d['sha256']
                            or not np.isfinite(a).all()):
                        raise ValueError('Typed complete snapshot array differs: ' + n)
                return arrays

            worst_equation = 0.
            first_mass = final_mass = None
            for step in sorted(states):
                bounded()
                arrays = load(step)
                for n, shape in zip(FIELDS, (tuple(reversed(counts)), *sizes)):
                    if arrays[n].size != math.prod(shape):
                        raise ValueError('Native staggered field dimensions differ')
                if step:
                    p = torch.from_numpy(arrays['pressure']).reshape(1, 1, *reversed(counts))
                    h = p.new_tensor([1 / scene['cells_per_metre']])
                    rho = p.new_tensor([1000.])
                    rhs = torch.from_numpy(arrays['rhs'])
                    error = float((numerics.fine_action(p, h, rho).flatten() - rhs).norm()
                                  / rhs.norm().clamp_min(1e-30))
                    worst_equation = max(worst_equation, error)
                if step == 0:
                    first_mass = float(arrays['tracer'].astype(np.float64).sum())
                if step == 4800:
                    final_mass = float(arrays['tracer'].astype(np.float64).sum())
            if worst_equation > reg['maximum_reference_equation_error']:
                raise ValueError('Saved pressure and matching fine RHS are inconsistent')
            windows = receipt['windows']
            starts = {w['start_fine_step'] for w in windows}
            observed = {(w['start_fine_step'], w['output_steps_per_s'], w['K']) for w in windows}
            expected = {(n, rate, k) for n in starts for rate in (30, 60, 120) for k in (1, 2, 3, 4)}
            if observed != expected or len(windows) != len(expected):
                raise ValueError('Missing or duplicate all12temporal targets')
            for w in windows:
                rate, k, start = w['output_steps_per_s'], w['K'], w['start_fine_step']
                if (w['output_dt_s'] != 1 / rate or w['block_duration_s'] != k / rate
                        or w['target_fine_steps'] != [start + j * (240 // rate) for j in range(1, k + 1)]
                        or any(n not in states for n in [start, *w['target_fine_steps']])):
                    raise ValueError('Temporal block does not advanceKwhole output intervals')

            replay_cases = []
            for start in reg['late_replay_starts']:
                bounded()
                fields = load(start)
                client = FineReferenceClient(scene, args.output / record['identifier'] / f'replay-{start}')
                try:
                    client.restore(start, {n: fields[n] for n in FIELDS})
                    checked = 0
                    for offset in range(1, 33):
                        bounded()
                        frame = client.advance()
                        result['replayed_fine_updates'] += 1
                        if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
                            raise ValueError('Late replay failed native stability')
                        if start + offset in states:
                            expected_state = load(start + offset)
                            packet, actual = client.inspect()
                            if packet['time_s'] != (start + offset) / 240:
                                raise ValueError('Late restart replay time differs')
                            if any(actual[n].tobytes() != expected_state[n].tobytes() for n in FIELDS):
                                raise ValueError('Late replay differs from uninterrupted physical trajectory')
                            for name, source in [('rhs', 'rhs'), ('pressure_base', 'base'), ('native_input_residual', 'native_input_residual')]:
                                if client.last_request[source].tobytes() != expected_state[name].tobytes():
                                    raise ValueError('Late replay changed fine pressure supervision')
                            checked += 1
                    replay_cases.append(dict(start=start, future_fine_updates=32,
                        compared_complete_states=checked, all_fields_and_labels_bitwise_equal=True))
                finally:
                    try:
                        custody = client.close()
                    finally:
                        if client.last_custody is not None:
                            result['native_processes'].append(client.last_custody)
                    if custody['returncode'] or not custody['graceful_protocol_complete']:
                        raise ValueError('Late replay native child failed to drain')
            result['cases'].append(dict(identifier=record['identifier'], split=receipt['split'],
                counts_xyz=list(counts), cells_per_metre=scene['cells_per_metre'],
                checked_state_files=len(states), checked_window_configurations=len(windows),
                worst_reference_equation_error=worst_equation,
                initial_tracer_sum=first_mass, final_tracer_sum=final_mass,
                late_replays=replay_cases, actual_fine_updates=4800,
                capture_elapsed_seconds=receipt['elapsed_seconds'], retained_state_bytes=receipt['retained_state_bytes']))
            write(args.output / 'results.json', result)
            print('Audited complete long reference and late replay:', record['identifier'], flush=True)
        if digest(bank_path) != result['bank_sha256'] or torch.cuda.is_initialized():
            raise ValueError('Capture changed during audit orCUDAwas initialized')
        result.update(status='passed_completed_reference_labels_windows_and_late_replay',
            all_four_pilots_valid=len(bank['completed']) == 4 and not bank['reference_rejections'],
            reference_rejections_preserved=bank['reference_rejections'],
            elapsed_seconds=time.monotonic() - started, memory_counters=process_memory())
    except BaseException as error:
        result.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
