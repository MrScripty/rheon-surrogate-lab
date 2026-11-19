"""Deterministic known emitter controls: varied forcing and independent motion."""
import math
import random

PROFILES = ('continuous', 'continuous_varying', 'regular_pulse', 'irregular_pulse')
MOTIONS = ('stationary', 'oscillating', 'reflected_line')


def make_program(seed, profile, motion, steps=240):
    if profile not in PROFILES or motion not in MOTIONS or not 1 <= steps <= 600:
        raise ValueError('Registered force profile, motion and actual-step count required')
    rng = random.Random(seed ^ 0x1FCA7B29)
    center = [rng.uniform(0.03, 0.97) for _ in range(3)]
    amplitude = [rng.uniform(0.01, max(0.011, min(v-.02, .98-v))) for v in center]
    frequency = [rng.uniform(.05, 1.5) for _ in range(3)]
    phase = [rng.uniform(0., math.tau) for _ in range(3)]
    speed = [rng.choice((-1., 1.))*rng.uniform(.05, .8) for _ in range(3)]
    pulse_frequency, duty, pulse_phase = rng.uniform(.5, 5.), rng.uniform(.15, .85), rng.random()
    variance = rng.uniform(.1, .8)
    knots = [rng.uniform(1-variance, 1+variance) for _ in range(math.ceil(steps/15)+2)]
    events, time = [], rng.uniform(0., .1)
    while time < steps/60:
        end = time+rng.uniform(.05, .5)
        events.append(dict(start=time, end=end, amplitude=rng.uniform(.2, 1.6)))
        time = end+rng.uniform(.05, .75)
    frames = []
    for step in range(steps):
        time = step/60
        if motion == 'stationary':
            location = center
        elif motion == 'oscillating':
            location = [v+amplitude[a]*math.sin(math.tau*frequency[a]*time+phase[a]) for a,v in enumerate(center)]
        else:
            location = [.03+(.94-abs(((v-.03+speed[a]*time)%1.88)-.94)) for a,v in enumerate(center)]
        if profile == 'continuous':
            pulse = 1.
        elif profile == 'continuous_varying':
            at = step/15
            index = int(at)
            pulse = knots[index]+(at-index)*(knots[index+1]-knots[index])
        elif profile == 'regular_pulse':
            pulse = float((time*pulse_frequency+pulse_phase)%1 < duty)
        else:
            pulse = next((e['amplitude'] for e in events if e['start'] <= time < e['end']), 0.)
        frames.append(dict(center=[round(v, 12) for v in location], pulse=round(pulse, 12)))
    program = dict(version=1, seed=seed, profile=profile, motion=motion,
        initial_center=center, motion_amplitude=amplitude, motion_frequency_hz=frequency,
        motion_phase=phase, motion_speed_domain_lengths_per_second=speed,
        pulse_frequency_hz=pulse_frequency, pulse_duty=duty, pulse_phase=pulse_phase,
        variance=variance, continuous_variance_knots=knots, irregular_pulse_events=events,
        schedule_is_known_external_input=True, future_physical_state_used=False,
        normalized_position_rounding_digits=12, dt=1/60)
    return program, frames


def make_scene(seed, counts, density, profile, motion, steps=240):
    rng = random.Random(seed ^ 0x52D19E63)
    program, frames = make_program(seed, profile, motion, steps)
    scene = dict(seed=seed, counts=list(counts), cells_per_metre=density,
        steps=steps, warmup_steps=0, center=frames[0]['center'],
        acceleration=10**rng.uniform(math.log10(.02), math.log10(2.)),
        radius=rng.uniform(.1, .45), swirl=rng.uniform(-1, 1),
        fluctuation=0. if profile == 'continuous' else rng.uniform(0, .8),
        frequency=rng.uniform(0., 4.), sgs=False,
        forcing_program=program, forcing_schedule=frames)
    return scene
