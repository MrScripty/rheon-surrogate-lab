"""Evaluate known force programs in physical time, independently of saved frames."""
import copy
import math
import random
from force_program import make_program, make_scene, PROFILES, MOTIONS

SUBSTEPS = (1, 2, 4, 8, 16, 32, 64)


def evaluate(program, time_s):
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError('Finite nonnegative physical time required')
    center = program['initial_center']
    motion = program['motion']
    if motion == 'stationary':
        location = center
    elif motion == 'oscillating':
        location = [v + program['motion_amplitude'][a] * math.sin(
            math.tau * program['motion_frequency_hz'][a] * time_s + program['motion_phase'][a])
            for a, v in enumerate(center)]
    elif motion == 'reflected_line':
        location = [.03 + (.94 - abs(((v - .03 +
            program['motion_speed_domain_lengths_per_second'][a] * time_s) % 1.88) - .94))
            for a, v in enumerate(center)]
    else:
        raise ValueError('Known emitter motion required')
    profile = program['profile']
    if profile == 'continuous':
        pulse = 1.
    elif profile == 'continuous_varying':
        if program['version'] == 1:
            at = time_s * 4
            index = int(at)
            knots = program['continuous_variance_knots']
            if index + 1 >= len(knots):
                raise ValueError('Continuous variance knots exhausted')
            pulse = knots[index] + (at - index) * (knots[index + 1] - knots[index])
        else:
            pulse = 1 + program['variance'] * sum(
                weight * math.sin(math.tau * frequency * time_s + phase)
                for weight, frequency, phase in zip(program['variance_weights'],
                    program['variance_frequencies_hz'], program['variance_phases']))
    elif profile == 'regular_pulse':
        pulse = float((time_s * program['pulse_frequency_hz'] + program['pulse_phase']) % 1
                      < program['pulse_duty'])
    elif profile == 'irregular_pulse':
        pulse = next((e['amplitude'] for e in program['irregular_pulse_events']
                      if e['start'] <= time_s < e['end']), 0.)
    else:
        raise ValueError('Known forcing profile required')
    return dict(center=[round(v, 12) for v in location], pulse=round(pulse, 12))


def frequency_program(seed, profile, motion, steps, force_frequency_hz, motion_frequency_hz):
    if profile not in PROFILES or motion not in MOTIONS:
        raise ValueError('Registered force profile and emitter motion required')
    if not all(math.isfinite(v) and .01 <= v <= 100 for v in
               (force_frequency_hz, motion_frequency_hz)):
        raise ValueError('Explicit finite physical pilot frequency required')
    program, _ = make_program(seed, profile, motion, steps)
    rng = random.Random(seed ^ 0x3B796A1D)
    program.update(version=2, duration_s=steps/60,
        evaluation='Continuous-time external program sampled at each actual internal reference time',
        output_timestep_s=1/60, pulse_frequency_hz=force_frequency_hz,
        motion_frequency_hz=[motion_frequency_hz * v for v in (.5, .75, 1.)],
        variance_weights=[.2, .3, .5],
        variance_frequencies_hz=[force_frequency_hz * v for v in (.25, .5, 1.)],
        variance_phases=[rng.uniform(0., math.tau) for _ in range(3)],
        pulse_amplitude_is_bandlimited_before_rounding=profile in ('continuous', 'continuous_varying'),
        sharp_pulse_edges=profile in ('regular_pulse', 'irregular_pulse'),
        pulse_frequency_is_not_a_bandwidth_limit=profile in ('regular_pulse', 'irregular_pulse'),
        irregular_event_rate_scale_hz=force_frequency_hz if profile == 'irregular_pulse' else None)
    program['motion_speed_domain_lengths_per_second'] = [
        math.copysign(1.88 * motion_frequency_hz * scale, old)
        for old, scale in zip(program['motion_speed_domain_lengths_per_second'], (.5, .75, 1.))]
    if profile == 'irregular_pulse':
        events = []
        at = -rng.random()/force_frequency_hz
        while at < steps/60:
            end = at + rng.uniform(.15, .85)/force_frequency_hz
            events.append(dict(start=at, end=end, amplitude=rng.uniform(.2, 1.6)))
            at = end + rng.uniform(.15, .85)/force_frequency_hz
        program['irregular_pulse_events'] = events
    return program


def sample_scene(scene, factor):
    if factor not in SUBSTEPS or not 1 <= scene['steps'] <= 600:
        raise ValueError('Bounded reference clock required')
    result = copy.deepcopy(scene)
    result['reference_substeps'] = factor
    result['forcing_control_substeps'] = factor
    dt = 1/(60*factor)
    result['forcing_schedule'] = [evaluate(result['forcing_program'], step*dt)
                                  for step in range(scene['steps']*factor)]
    result['center'] = result['forcing_schedule'][0]['center']
    return result


def frequency_scene(seed, counts, density, profile, motion, steps, force_frequency_hz,
                    motion_frequency_hz):
    scene = make_scene(seed, counts, density, profile, motion, steps)
    scene['forcing_program'] = frequency_program(seed, profile, motion, steps,
                                                 force_frequency_hz, motion_frequency_hz)
    # Keep the spatial wave's temporal components explicit and below the force pilot maximum.
    scene['frequency'] = force_frequency_hz / 3
    return sample_scene(scene, 1)
