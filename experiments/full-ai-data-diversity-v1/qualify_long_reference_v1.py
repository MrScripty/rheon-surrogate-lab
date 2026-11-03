"""Fine-state timing, force, legacy and restart qualification; no learned model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from support import ROOT, read, write, digest
from fine_reference_client_v1 import FineReferenceClient
from continuous_reference_client import ContinuousReferenceClient
from long_reference_scene_v1 import make_long_scene, initial_state
from force_program import PROFILES, MOTIONS
from recurrent_sampler_native import dimensions
from scheduled_force_fine_controls import force


def hashes(values):
    return {n: hashlib.sha256(v.tobytes()).hexdigest() for n, v in values.items()}


def run(args):
    if args.output.exists():
        raise ValueError('Fresh reference qualification directory required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered qualification source changed: ' + name)
    os.sched_setaffinity(0, {reg['host_cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    began = time.monotonic()
    result = dict(status='running', force_comparisons=[], legacy=[], restarts=[],
                  native_processes=[], fine_steps_executed=0,
                  new_parameter_fits=0, learned_forwards=0, cpu_model_forward=False,
                  reserved52_references_opened=0, full_generator_target_pass=False,
                  source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol))

    def bounded():
        if time.monotonic() - began > reg['maximum_wall_seconds']:
            raise TimeoutError('Registered reference qualification wall budget')

    def close(client):
        custody = client.close()
        result['native_processes'].append(custody)
        if not custody['reaped'] or custody['returncode'] or not custody['graceful_protocol_complete']:
            raise ValueError('Reference child did not complete and drain')

    def advance(client):
        bounded()
        frame = client.advance()
        result['fine_steps_executed'] += getattr(client, 'reference_substeps', 1) if isinstance(client, ContinuousReferenceClient) else 1
        if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
            raise ValueError('Native stability gate rejected reference advance')
        return frame

    try:
        for p, profile in enumerate(PROFILES):
            for m, motion in enumerate(MOTIONS):
                serial = p * 3 + m
                scene = make_long_scene(reg['seed'] + serial, [5, 7, 9],
                                        (10, 25, 50, 100)[p], profile, motion,
                                        duration_s=1, force_frequency_hz=(.5, 3., 9., 17.)[p])
                fine = FineReferenceClient(scene, args.output / f'forces-{serial:02d}')
                try:
                    before_packet, before = fine.inspect()
                    for step in (0, 1, 3, 4, 239):
                        shapes = dimensions(scene['counts'])[:3]
                        for label, value in [('zero', 0.), ('nonzero', .0125)]:
                            base = [torch.full(s, value, dtype=torch.float32) for s in shapes]
                            _, expected = force(scene, step, base)
                            actual = fine.preview_force(step, [v.numpy() for v in base])
                            if any(a.tobytes() != b.numpy().tobytes() for a, b in zip(actual, expected)):
                                raise ValueError('Long-reference force differs from independent evaluator')
                            result['force_comparisons'].append(dict(profile=profile, motion=motion,
                                fine_step=step, base=label, fields_bitwise_equal=True))
                    after_packet, after = fine.inspect()
                    if before_packet != after_packet or hashes(before) != hashes(after):
                        raise ValueError('Read-only forcing preview changed native state')
                finally:
                    close(fine)

                old = ContinuousReferenceClient(scene, args.output / f'legacy-old-{serial:02d}')
                new = FineReferenceClient(scene, args.output / f'legacy-new-{serial:02d}')
                try:
                    compared = 0
                    for fine_step in range(33):
                        if fine_step:
                            advance(new)
                        packet, values = new.inspect()
                        if packet['step'] != fine_step or packet['time_s'] != fine_step / 240:
                            raise ValueError('Fine state publication clock differs')
                        if fine_step % 4 == 0:
                            if fine_step:
                                advance(old)
                            old_packet, old_values = old.inspect()
                            if hashes(values) != hashes(old_values) or packet['time_s'] != old_packet['time_s']:
                                raise ValueError('Duration extension changed existing native arithmetic')
                            compared += 1
                    result['legacy'].append(dict(profile=profile, motion=motion,
                        complete_macro_states=compared, all_fields_bitwise_equal=True,
                        fine_states_checked=33))
                finally:
                    close(old)
                    close(new)

                original = FineReferenceClient(scene, args.output / f'restart-original-{serial:02d}')
                restarted = None
                try:
                    recipe = dict(seed=reg['seed'] ^ (serial + 981), tracer_blobs=2 + serial % 3,
                                  initial_velocity_max_m_per_s=.002 if serial % 2 else 0., curl_z_modes=1)
                    original.restore(0, initial_state(scene, recipe))
                    prefix = 3 if serial % 2 else 31
                    for _ in range(prefix):
                        advance(original)
                    packet, state = original.inspect()
                    restarted = FineReferenceClient(scene, args.output / f'restart-copy-{serial:02d}')
                    restarted.restore(prefix, state)
                    for offset in range(1, 33):
                        advance(original)
                        advance(restarted)
                        a, av = original.inspect()
                        b, bv = restarted.inspect()
                        if a != b or hashes(av) != hashes(bv):
                            raise ValueError('Fine restart changed future complete states')
                        for name in ('base', 'rhs', 'native_input_residual'):
                            if original.last_request[name].tobytes() != restarted.last_request[name].tobytes():
                                raise ValueError('Fine restart changed pressure supervision')
                    result['restarts'].append(dict(profile=profile, motion=motion,
                        restart_fine_index=prefix, compared_future_states=32,
                        all_fields_and_pressure_labels_bitwise_equal=True,
                        initial_state_recipe=recipe))
                finally:
                    close(original)
                    if restarted is not None:
                        close(restarted)
                write(args.output / 'results.json', result)
                print('Qualified long reference:', profile, motion, flush=True)

        scene = make_long_scene(reg['seed'] + 1337, [5, 5, 5], 25,
                                'continuous', 'stationary', duration_s=20)
        scene['acceleration'] = 0.
        client = FineReferenceClient(scene, args.output / 'quiescent20seconds')
        try:
            _, initial = client.inspect()
            identity = hashes(initial)
            for fine_step in range(1, 4801):
                advance(client)
                if fine_step % 240 == 0 or fine_step == 1:
                    packet, values = client.inspect()
                    if packet['time_s'] != fine_step / 240 or hashes(values) != identity:
                        raise ValueError('Full20second quiescent reference changed state or time')
            if packet['step'] != 4800 or packet['time_s'] != 20.:
                raise ValueError('Full declared reference duration not reached')
            result['quiescent20seconds'] = dict(actual_fine_updates=4800,
                final_time_s=packet['time_s'], all_sampled_states_bitwise_unchanged=True)
        finally:
            close(client)
        if torch.cuda.is_initialized():
            raise ValueError('CPU numerical reference qualification initializedCUDA')
        if len(result['force_comparisons']) != 120 or len(result['legacy']) != 12 or len(result['restarts']) != 12:
            raise ValueError('Incomplete registered reference qualification population')
        result.update(status='passed_long_reference_fine_clock_forces_restart_and_duration',
                      elapsed_seconds=time.monotonic() - began)
    except BaseException as error:
        result.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - began)
        raise
    finally:
        write(args.output / 'results.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
