"""Isolate pulse phase with stationary forcing and matched-impulse controls.

The interval-integrated gate is an explicit diagnostic forcing modification.
It fixes scalar pulse impulse, not the fluid integrator's event timing. No
result is added to the training bank or described as converged ground truth.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from long_reference_scene_v1 import make_long_scene, initial_state
from fine_reference_client_v2 import FineReferenceClient
from pulse_event_controls_v1 import event_packet, mean_pulse_gate
from continuous_force_program import evaluate
from capture_memory_telemetry_v1 import process_memory

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def hashes(state):
    return {n: hashlib.sha256(state[n].tobytes()).hexdigest() for n in FIELDS}


def error(a, b, names):
    delta = np.concatenate([(a[n].astype(np.float64) - b[n]).ravel() for n in names])
    baseline = np.concatenate([b[n].astype(np.float64).ravel() for n in names])
    absolute, norm = float(np.linalg.norm(delta)), float(np.linalg.norm(baseline))
    return dict(relative_l2=absolute / max(norm, 1e-30), absolute_l2=absolute,
                denominator_l2=norm, denominator_floor=1e-30)


def run(args):
    if args.output.exists():
        raise ValueError('Fresh pulse phase diagnostic required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    if read(ROOT / reg['fine480_qualification'])['status'] != 'passed480_long_reference_fine_clock_forces_restart_and_duration':
        raise ValueError('Qualified native240/480fine reference required')
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result = dict(status='running', runs=[], comparisons=[], native_processes=[], fine_updates=0,
        exact_event_packets=[], source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol), new_parameter_fits=0, learned_forwards=0,
        cpu_model_forward=False, reserved52_references_opened=0, full_generator_target_pass=False,
        reference_cpu_policy=reg['reference_cpu_policy'], native_threads=1,
        not_training_data=True, temporal_convergence_proven=False, latency_claim=False,
        scope='Stationary emitter, swirl0/fluctuation0; identical initial complete state and pulse duration/amplitude. Pulse onset is shifted inside one240fine interval. Compare original leftsampled gates against exact intervalmean gates on240and480clocks. Only scalar gate impulse is analytic; fluid advection/projection remain discrete.')
    outputs, retained_bytes = {}, 0
    try:
        for phase_index, phase in enumerate(reg['phases']):
            onset, end = (1 + phase) / 240, (3 + phase) / 240
            for factor in (4, 8):
                rate = 60 * factor
                for mode in ('left_sampled', 'interval_mean'):
                    label = f'phase-{phase_index}-r{rate}-{mode}'
                    scene = make_long_scene(reg['seed'], reg['counts_xyz'], reg['cells_per_metre'],
                        'irregular_pulse', 'stationary', duration_s=1, reference_substeps=factor)
                    scene.update(acceleration=reg['acceleration_m_per_s2'], radius=.22,
                                 swirl=0., fluctuation=0., center=[.5, .5, .5])
                    program = scene['forcing_program']
                    program.update(initial_center=[.5, .5, .5],
                        irregular_pulse_events=[dict(start=onset, end=end, amplitude=1.)])
                    for i, row in enumerate(scene['forcing_schedule']):
                        row['center'] = [.5, .5, .5]
                        row['pulse'] = (evaluate(program, i / rate)['pulse'] if mode == 'left_sampled'
                            else mean_pulse_gate(program, i / rate, (i + 1) / rate))
                    if factor == 4 and mode == 'left_sampled':
                        for output_rate in (30, 60, 120):
                            for k in (1, 2, 3, 4):
                                packet = event_packet(program, 0., k / output_rate,
                                                      [j / output_rate for j in range(1, k + 1)])
                                result['exact_event_packets'].append(dict(phase=phase,
                                    output_steps_per_s=output_rate, K=k, **packet))
                    initial = initial_state(scene, dict(seed=reg['initial_state_seed'], tracer_blobs=3,
                        initial_velocity_max_m_per_s=0., curl_z_modes=1))
                    directory = args.output / label
                    directory.mkdir()
                    client = FineReferenceClient(scene, directory / 'native')
                    states = {}
                    try:
                        client.restore(0, initial)
                        for fine in range(1, 32 * (rate // 240) + 1):
                            if (time.monotonic() - started > reg['maximum_wall_seconds']
                                    or process_memory()['peak_since_exec_bytes'] > reg['maximum_peak_rss_bytes']):
                                raise ValueError('Registered reference diagnostic resource bound exceeded')
                            frame = client.advance()
                            result['fine_updates'] += 1
                            if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
                                raise ValueError('Pulse phase reference failed native quality')
                            if fine % (rate // 240) == 0 and fine // (rate // 240) in reg['output_offsets240']:
                                offset = fine // (rate // 240)
                                packet, state = client.inspect()
                                if packet['time_s'] != fine / rate:
                                    raise ValueError('Matched physical endpoint differs')
                                states[offset] = state
                                path = directory / f'offset240-{offset:02d}.npz'
                                np.savez_compressed(path, **state)
                                retained_bytes += path.stat().st_size
                                if retained_bytes > reg['maximum_retained_state_bytes']:
                                    raise ValueError('Reference retained output bound exceeded')
                    finally:
                        try:
                            custody = client.close()
                        finally:
                            if client.last_custody is not None:
                                result['native_processes'].append(client.last_custody)
                        if custody['returncode'] or not custody['reaped'] or not custody['graceful_protocol_complete']:
                            raise ValueError('Reference child did not complete and drain')
                    impulse = sum(scene['forcing_schedule'][i]['pulse'] / rate
                                  for i in range(32 * (rate // 240)))
                    if not math.isclose(impulse, 2 / 240, rel_tol=0, abs_tol=1e-14):
                        raise ValueError('Matched scalar total pulse impulse was not retained')
                    if mode == 'interval_mean':
                        for offset in reg['output_offsets240']:
                            actual = sum(scene['forcing_schedule'][i]['pulse'] / rate
                                for i in range(offset * (rate // 240)))
                            exact = mean_pulse_gate(program, 0., offset / 240) * (offset / 240)
                            if not math.isclose(actual, exact, rel_tol=0, abs_tol=1e-14):
                                raise ValueError('Intervalmean controls lost prefix impulse')
                    outputs[phase_index, rate, mode] = states
                    result['runs'].append(dict(label=label, phase=phase, rate_per_s=rate, mode=mode,
                        onset_s=onset, end_s=end, total_pulse_integral_s=impulse,
                        state_hashes={str(n): hashes(s) for n, s in states.items()},
                        complete_fields_at_all_eight_requested_endpoints=True))
                    write(args.output / 'results.json', result)
            print('Compared pulse edge fraction:', phase, flush=True)
        for phase_index, phase in enumerate(reg['phases']):
            for a, b in [((240, 'left_sampled'), (480, 'left_sampled')),
                         ((240, 'interval_mean'), (480, 'interval_mean')),
                         ((240, 'left_sampled'), (240, 'interval_mean')),
                         ((480, 'left_sampled'), (480, 'interval_mean'))]:
                for offset in reg['output_offsets240']:
                    x, y = outputs[(phase_index, *a)][offset], outputs[(phase_index, *b)][offset]
                    result['comparisons'].append(dict(phase=phase, first=list(a), second=list(b),
                        physical_elapsed_s=offset / 240, errors={label: error(x, y, names)
                            for label, names in [('pressure', ('pressure',)),
                                ('velocity', ('vx', 'vy', 'vz')), ('tracer', ('tracer',))]}))
        identical = all(hashes(outputs[i, 240, 'left_sampled'][n]) == hashes(outputs[0, 240, 'left_sampled'][n])
                        for i in range(len(reg['phases'])) for n in reg['output_offsets240'])
        first, last = outputs[0, 240, 'interval_mean'][2], outputs[len(reg['phases']) - 1, 240, 'interval_mean'][2]
        if (not identical or hashes(first) == hashes(last)
                or result['fine_updates'] != reg['expected_native_advances'] or torch.cuda.is_initialized()):
            raise ValueError('Full population, phase ambiguity witness, or CPU-reference-only invariant failed')
        result.update(status='completed_stationary_matched_impulse_pulse_phase_diagnostic',
            all240_left_sampled_complete_outputs_identical_across_distinct_event_phases=True,
            intervalmean_outputs_distinguish_early_late_events=True,
            early_late_intervalmean_first_endpoint_difference={label: error(first, last, names)
                for label, names in [('pressure', ('pressure',)), ('velocity', ('vx', 'vy', 'vz')), ('tracer', ('tracer',))]},
            total_retained_state_bytes=retained_bytes, elapsed_seconds=time.monotonic() - started)
    except BaseException as exception:
        result.update(status='failed', error=repr(exception))
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
