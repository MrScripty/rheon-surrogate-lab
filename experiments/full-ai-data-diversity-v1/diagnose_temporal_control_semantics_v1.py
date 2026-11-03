"""Compare scalar forcing timing implied by interpolation with sampled controls.

No fluid fields, model forwards or fitting. This isolates known temporal input
semantics; force motion/swirl and nonlinear fluid pressure are not represented
by this scalar diagnostic.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from support import ROOT, read, write, digest
from temporal_control_packets_v1 import control_packet


def run(args):
    if args.output.exists():
        raise ValueError('Fresh temporal-control diagnostic required')
    reg = read(args.protocol)
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen control diagnostic changed')
    bank = read(ROOT / reg['bank_manifest'])
    args.output.mkdir(parents=True)
    rows, counts = [], Counter()
    for identifier, case in bank['masters'].items():
        scene = case['scene']
        for window in case['windows']:
            start, rate, k = (window[n] for n in ('start_fine_step', 'output_steps_per_s', 'K'))
            stride = 240 // rate
            packet = control_packet(scene, start, k * stride)
            strength = packet[:, 14]
            previous = scene['acceleration'] * scene['forcing_schedule'][start - 1]['pulse'] if start else 0.
            total_impulse = float((strength * packet[:, 17]).sum())
            whole_mean = float(strength.mean())
            for lead in range(1, k + 1):
                end = lead * stride
                alpha = lead / k
                final_sample = float(strength[end - 1])
                implied_pressure_strength = whole_mean if lead == k else previous + alpha * (whole_mean - previous)
                prefix_impulse = float((strength[:end] * packet[:end, 17]).sum())
                last_off_mean_on = final_sample == 0. and whole_mean > 0.
                cold_constant_ramp = (start == 0 and lead < k and np.all(strength == strength[0]) and strength[0] > 0)
                counts['physical_outputs'] += 1
                counts['last_off_whole_mean_on'] += int(last_off_mean_on)
                counts['cold_constant_force_pressure_ramp'] += int(cold_constant_ramp)
                counts['scalar_pressure_timing_mismatches'] += int(abs(implied_pressure_strength - final_sample) > 1e-12)
                counts['scalar_impulse_interpolation_mismatches'] += int(abs(alpha * total_impulse - prefix_impulse) > 1e-12)
                rows.append(dict(identifier=identifier, cells_per_metre=scene['cells_per_metre'], counts_xyz=scene['counts'],
                    profile=scene['forcing_program']['profile'], motion=scene['forcing_program']['motion'],
                    start_fine_step=start, output_steps_per_s=rate, K=k, lead=lead,
                    whole_mean_strength=whole_mean, native_last_sample_strength=final_sample,
                    scalar_pressure_interpolation_strength=implied_pressure_strength,
                    known_prefix_impulse=prefix_impulse, linear_interpolated_impulse=alpha * total_impulse,
                    last_sample_off_but_whole_mean_on=bool(last_off_mean_on), cold_constant_force_ramped_pressure=bool(cold_constant_ramp)))
    write(args.output / 'results.json', dict(status='completed_training_only_known_scalar_control_timing_diagnostic',
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol), counts=dict(counts), rows=rows,
        independent_training_masters=24, pressure_fields_read=0, fluid_model_forwards=0,
        learned_CPU_forwards=0, native_reference_advances=0, new_parameter_fits=0,
        exact_sampled240_controls_only=True, full_force_geometry_or_fluid_accuracy_claim=False,
        reference_clock_or_labels_changed=False, reserved52_references_opened=0,
        hypothesis='Pressure forcing is sampled at the last fine interval while velocity responds to cumulative forcing. A shared endpoint/interior decoder should distinguish these timing rules. Scalar guide alone does not predict nonlinear flow pressure or moving-force geometry.',
        full_generator_target_pass=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
