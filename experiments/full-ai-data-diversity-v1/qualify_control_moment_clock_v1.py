"""Exact-rational temporal integration control, not a fluid or model test."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time


def run(args):
    if args.output.exists():
        raise ValueError('Fresh temporal moment control required')
    reg = json.loads(args.protocol.read_text())
    for name, expected in reg['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Temporal moment control source changed')
    rows = []
    for rate in (30, 60, 120):
        for k in (1, 2, 3, 4):
            m = k * (240 // rate)
            for pattern in ('constant', 'first_pulse', 'middle_pulse', 'last_pulse', 'alternating', 'irregular'):
                force = {'constant': [Q(3, 7)] * m,
                    'first_pulse': [Q(int(r == 0)) for r in range(m)],
                    'middle_pulse': [Q(int(r == m // 2)) for r in range(m)],
                    'last_pulse': [Q(int(r == m - 1)) for r in range(m)],
                    'alternating': [Q((-1) ** r, 3) for r in range(m)],
                    'irregular': [Q(((r * r + 7 * r + 3) % 13) - 6, 11) for r in range(m)]}[pattern]
                for initial in (Q(0), Q(2, 5), Q(-3, 2)):
                    dt, displacement, velocity = Q(1, 240), Q(0), initial
                    for acceleration in force:
                        # Independent scalar loop follows transport-before-force
                        # ordering, rather than the candidate moment formula.
                        displacement += dt * velocity
                        velocity += dt * acceleration
                    guide_mean_velocity = initial + dt * sum(
                        Q(m - 1 - r, m) * a for r, a in enumerate(force))
                    guide_displacement = (m * dt) * guide_mean_velocity
                    if guide_displacement != displacement or m * dt != Q(k, rate):
                        raise ValueError('Whole horizon or discrete-force displacement differs')
                    if pattern == 'last_pulse' and (displacement != initial * m * dt or velocity == initial):
                        raise ValueError('Final force incorrectly moved tracer in the same fine update')
                    rows.append(dict(output_steps_per_s=rate, K=k, fine_intervals=m,
                        pattern=pattern, initial_velocity=str(initial), physical_duration_s=str(m * dt),
                        exact_displacement=str(displacement), exact_scalar_equality=True))
    result = dict(status='passed_exact_rational_transport_before_force_clock_control',
        created_unix=time.time(), source_sha256=reg['source_sha256'],
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(), rows=rows,
        cases=len(rows), fluid_fields_read=0, model_forwards=0, new_parameter_fits=0,
        reserved52_references_opened=0, full_generator_target_pass=False,
        limitation='Proves only temporal moment arithmetic for prescribed scalar acceleration and transport-before-force ordering. It does not qualify spatial fluid characteristics, learned projection/sampler, F32 rounding, native accuracy or GPU latency.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], result['cases'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
