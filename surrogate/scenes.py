"""Build a scene's full parameter set from a few defining values.

A scene is what the solver needs to run a simulation: grid, forcing blob and how
the blob moves and pulses over time. Everything except the grid follows from one
seed, so a dataset is defined by a short table of seeds and shapes, not by stored
scene files.

`scheduled_scene` reproduces, value for value, the scenes the lab recorded as its
"diversity" data (forcing profile and motion chosen per scene, a centre and pulse
for every output step). Changing anything here changes those scenes, so the tests
pin the result.
"""
import math
import random

PROFILES = ('continuous', 'continuous_varying', 'regular_pulse', 'irregular_pulse')
MOTIONS = ('stationary', 'oscillating', 'reflected_line')
OUTPUT_RATE_HZ = 60
MAX_STEPS = 600


def _forcing_program(seed, profile, motion, steps):
    """The forcing blob's centre (as a fraction of the box) and pulse strength at every output step."""
    rng = random.Random(seed ^ 0x1FCA7B29)
    center = [rng.uniform(0.03, 0.97) for _ in range(3)]
    amplitude = [rng.uniform(0.01, max(0.011, min(v - .02, .98 - v))) for v in center]
    frequency = [rng.uniform(.05, 1.5) for _ in range(3)]
    phase = [rng.uniform(0., math.tau) for _ in range(3)]
    speed = [rng.choice((-1., 1.)) * rng.uniform(.05, .8) for _ in range(3)]
    pulse_frequency, duty, pulse_phase = rng.uniform(.5, 5.), rng.uniform(.15, .85), rng.random()
    variance = rng.uniform(.1, .8)
    knots = [rng.uniform(1 - variance, 1 + variance) for _ in range(math.ceil(steps / 15) + 2)]
    events, time = [], rng.uniform(0., .1)
    while time < steps / OUTPUT_RATE_HZ:
        end = time + rng.uniform(.05, .5)
        events.append(dict(start=time, end=end, amplitude=rng.uniform(.2, 1.6)))
        time = end + rng.uniform(.05, .75)
    frames = []
    for step in range(steps):
        time = step / OUTPUT_RATE_HZ
        if motion == 'stationary':
            location = center
        elif motion == 'oscillating':
            location = [v + amplitude[a] * math.sin(math.tau * frequency[a] * time + phase[a])
                        for a, v in enumerate(center)]
        else:   # reflected_line: straight travel that bounces between the walls
            location = [.03 + (.94 - abs(((v - .03 + speed[a] * time) % 1.88) - .94)) for a, v in enumerate(center)]
        if profile == 'continuous':
            pulse = 1.
        elif profile == 'continuous_varying':
            at = step / 15
            index = int(at)
            pulse = knots[index] + (at - index) * (knots[index + 1] - knots[index])
        elif profile == 'regular_pulse':
            pulse = float((time * pulse_frequency + pulse_phase) % 1 < duty)
        else:   # irregular_pulse
            pulse = next((e['amplitude'] for e in events if e['start'] <= time < e['end']), 0.)
        frames.append(dict(center=[round(v, 12) for v in location], pulse=round(pulse, 12)))
    program = dict(version=1, seed=seed, profile=profile, motion=motion,
                   initial_center=center, motion_amplitude=amplitude, motion_frequency_hz=frequency,
                   motion_phase=phase, motion_speed_domain_lengths_per_second=speed,
                   pulse_frequency_hz=pulse_frequency, pulse_duty=duty, pulse_phase=pulse_phase,
                   variance=variance, continuous_variance_knots=knots, irregular_pulse_events=events,
                   schedule_is_known_external_input=True, future_physical_state_used=False,
                   normalized_position_rounding_digits=12, dt=1 / OUTPUT_RATE_HZ)
    return program, frames


def scheduled_scene(seed, counts, cells_per_metre, profile, motion, steps=240):
    """A scene with scheduled forcing. `counts` are cells along x, y, z."""
    if profile not in PROFILES or motion not in MOTIONS:
        raise ValueError(f'Profile must be one of {PROFILES} and motion one of {MOTIONS}.')
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f'A scheduled scene has 1 to {MAX_STEPS} steps.')
    rng = random.Random(seed ^ 0x52D19E63)
    program, frames = _forcing_program(seed, profile, motion, steps)
    return dict(seed=seed, counts=list(counts), cells_per_metre=cells_per_metre,
                steps=steps, warmup_steps=0, center=frames[0]['center'],
                acceleration=10 ** rng.uniform(math.log10(.02), math.log10(2.)),
                radius=rng.uniform(.1, .45), swirl=rng.uniform(-1, 1),
                fluctuation=0. if profile == 'continuous' else rng.uniform(0, .8),
                frequency=rng.uniform(0., 4.), sgs=False,
                forcing_program=program, forcing_schedule=frames)
