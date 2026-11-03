"""Qualify actual fine-time forcing, baseline compatibility and state restart."""
import argparse
import copy
from pathlib import Path
import os
from support import ROOT, read, write, digest
from continuous_reference_client import ContinuousReferenceClient
from macro_native_client import MacroNativeClient
from continuous_force_program import frequency_scene, sample_scene, evaluate
from scheduled_force_fine_controls import force
from force_program import make_program, make_scene, PROFILES, MOTIONS
from recurrent_sampler_native import dimensions
import numpy as np
import torch


def equal(a, b):
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and
        a[k].tobytes() == b[k].tobytes() for k in a)


def run(args):
    if args.output.exists():
        raise ValueError('Fresh continuous reference qualification required')
    reg = read(args.protocol)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h:
            raise ValueError('Registered continuous reference source changed: '+p)
    os.sched_setaffinity(0, {reg['cpu']})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(reg['seed'])
    args.output.mkdir(parents=True)
    rows, owners, legacy, restarts = [], [], [], []
    serial = 0

    def close(client):
        result = client.close()
        owners.append(result)
        if not result['reaped'] or result['returncode'] != 0 or not result['graceful_protocol_complete']:
            raise ValueError('Every native reference child must complete and reap')

    def advance(client):
        frame = client.advance()
        if not frame['frame']['advanced'] or not frame['frame']['quality']['qualified']:
            raise ValueError('Actual qualified physical advance required')
        if len(frame['fine_frames']) != client.reference_substeps:
            raise ValueError('Every internal reference advance must execute')

    # This independent evaluator must recover the original sampled program at its saved times.
    old_program_checks = 0
    for index, profile in enumerate(PROFILES):
        for motion in MOTIONS:
            program, controls = make_program(reg['seed']+index, profile, motion, 240)
            for step, expected in enumerate(controls):
                actual = evaluate(program, step/60)
                if np.max(np.abs(np.asarray([*actual['center'], actual['pulse']]) -
                                 np.asarray([*expected['center'], expected['pulse']]))) > 2e-12:
                    raise ValueError('Continuous evaluator changed a coarse program sample')
                old_program_checks += 1
    with torch.inference_mode():
        for factor in reg['substeps']:
            for density in reg['densities']:
                for profile in PROFILES:
                    for motion in MOTIONS:
                        frequency = reg['frequencies_hz'][serial % len(reg['frequencies_hz'])]
                        scene = frequency_scene(reg['seed']+serial, [5, 7, 9], density,
                                                profile, motion, 8, frequency, frequency)
                        scene = sample_scene(scene, factor)
                        serial += 1
                        client = ContinuousReferenceClient(scene, args.output/f'force-{serial:03d}')
                        try:
                            before_packet, before = client.inspect()
                            for step in sorted({0, factor-1, factor, 8*factor-1}):
                                increments, _ = force(scene, step)
                                fixtures = [
                                    ('zero', [torch.zeros(s, dtype=torch.float32)
                                              for s in dimensions(scene['counts'])[:3]]),
                                    ('random', [torch.randn(s, dtype=torch.float32)*.01
                                                for s in dimensions(scene['counts'])[:3]]),
                                    ('canceling', [(-v).float() for v in increments])]
                                for kind, base in fixtures:
                                    _, merged = force(scene, step, base)
                                    expected = client.preview_force(step, [v.numpy() for v in base])
                                    if any(a.tobytes() != b.numpy().tobytes() for a, b in zip(expected, merged)):
                                        raise ValueError('Fine-time forcing bytes differ: '+str(
                                            (factor, density, profile, motion, step, kind)))
                                    rows.append(dict(substeps=factor, cells_per_metre=density,
                                        counts_xyz=scene['counts'], profile=profile, motion=motion,
                                        control_time_s=step/(60*factor), native_step=step,
                                        physical_frequency_parameter_hz=frequency,
                                        base=kind, all_fields_bitwise_equal=True))
                            after_packet, after = client.inspect()
                            if before_packet != after_packet or not equal(before, after):
                                raise ValueError('Forcing preview mutated state or time')
                        finally:
                            close(client)
            print('Qualified fine-time controls at reference factor', factor, flush=True)
        # New binary transport must also preserve the existing macro-held baseline exactly.
        for factor in reg['baseline_substeps']:
            for density in reg['densities']:
                scene = make_scene(2026102059, [8, 7, 5], density,
                                   'continuous_varying', 'oscillating', 8)
                scene['reference_substeps'] = factor
                old = MacroNativeClient(scene, args.output/f'baseline-old-{factor}-{density}')
                new = ContinuousReferenceClient(scene, args.output/f'baseline-new-{factor}-{density}')
                try:
                    for step in range(5):
                        if step:
                            advance(old)
                            advance(new)
                        op, oa = old.inspect()
                        npacket, na = new.inspect()
                        if not equal(oa, na) or op != npacket:
                            raise ValueError('Macro-held compatibility changed physical bytes or time')
                    legacy.append(dict(substeps=factor, cells_per_metre=density,
                                       complete_states=5, all_fields_bitwise_equal=True))
                finally:
                    close(old)
                    close(new)
        for factor in reg['restart_substeps']:
            for index, profile in enumerate(PROFILES):
                scene = frequency_scene(2026102061+index, [8, 7, 5],
                    (10, 25, 50, 100)[index], profile, MOTIONS[index % 3], 8, 75., 20.)
                scene = sample_scene(scene, factor)
                original = ContinuousReferenceClient(scene,
                    args.output/f'restart-original-{factor}-{index}')
                restored = None
                try:
                    for _ in range(4):
                        advance(original)
                    packet, state = original.inspect()
                    if packet['step'] != 4 or packet['native_step'] != 4*factor or packet['time_s'] != 4/60:
                        raise ValueError('Restart prefix must advance real physical time')
                    restored = ContinuousReferenceClient(scene,
                        args.output/f'restart-restored-{factor}-{index}')
                    restored.restore(4, state)
                    for step in range(5, 9):
                        advance(original)
                        advance(restored)
                        op, oa = original.inspect()
                        rp, ra = restored.inspect()
                        if not equal(oa, ra) or op != rp or op['native_step'] != step*factor:
                            raise ValueError('Fine-time control restart changed state or clock')
                    restarts.append(dict(substeps=factor, profile=profile,
                        complete_future_macro_states=4, actual_future_native_steps=4*factor,
                        all_fields_bitwise_equal=True))
                finally:
                    close(original)
                    if restored is not None:
                        close(restored)
        # A pulse invisible at the output boundary must still affect the evolved physical state.
        scene = frequency_scene(2026102067, [8, 7, 5], 25, 'regular_pulse',
                                'stationary', 1, 75., 1.)
        fine = sample_scene(scene, 4)
        center = fine['forcing_schedule'][0]['center']
        fine['forcing_schedule'] = [dict(center=center, pulse=v) for v in (0., 1., 0., 0.)]
        coarse = copy.deepcopy(fine)
        coarse['forcing_control_substeps'] = 1
        coarse['forcing_schedule'] = [dict(center=center, pulse=0.)]
        fc = ContinuousReferenceClient(fine, args.output/'interior-pulse-fine')
        cc = ContinuousReferenceClient(coarse, args.output/'interior-pulse-held')
        try:
            fp, fs = fc.inspect()
            cp, cs = cc.inspect()
            if fp != cp or not equal(fs, cs):
                raise ValueError('Pulse witness initial conditions must match exactly')
            advance(fc)
            advance(cc)
            fp, fs = fc.inspect()
            cp, cs = cc.inspect()
            velocity_difference = max(float(np.max(np.abs(fs[k].astype('f8')-cs[k])))
                                      for k in ('vx', 'vy', 'vz'))
            if fp != cp or fp['step'] != 1 or not velocity_difference > 0:
                raise ValueError('Interior forcing must affect the actual complete output state')
            witness = dict(output_timestep_s=1/60, substeps=4,
                interior_control_pulses=[0., 1., 0., 0.],
                coarse_sample_pulse=0., output_velocity_max_abs_difference=velocity_difference,
                real_physical_evolution=True)
        finally:
            close(fc)
            close(cc)
    if len(rows) != reg['expected_force_comparisons'] or torch.cuda.is_initialized():
        raise ValueError('Qualification population or device changed')
    write(args.output/'qualification.json', dict(status='passed_fine_time_external_control_qualification',
        force_comparisons=rows, old_program_sample_checks=old_program_checks,
        baseline_compatibility=legacy, restarts=restarts, interior_pulse_witness=witness,
        native_processes=owners, output_timestep_s=1/60,
        temporal_convergence_qualified=False, model_targets_verified=False, optimizer_updates=0,
        reserved_references_consumed=False, source_sha256=reg['source_sha256'],
        protocol_sha256=digest(args.protocol)))
    print('Qualified', len(rows), 'exact fine-time force cases, compatibility, restart and interior pulse', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
