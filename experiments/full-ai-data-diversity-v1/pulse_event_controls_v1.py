"""Exact prescribed pulse edges and pulse-gate integrals, without field labels.

An interval-mean gate is not an exact full fluid update or an integral of a
moving, time-modulated spatial force. The initial diagnostic disables those
other force changes. Existing sampled reference archives are not modified.
"""
import math

from continuous_force_program import evaluate


def pulse_intervals(program, start, end):
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
        raise ValueError('Finite positive physical interval required')
    profile = program['profile']
    if profile == 'regular_pulse':
        frequency, phase, duty = (program[n] for n in ('pulse_frequency_hz', 'pulse_phase', 'pulse_duty'))
        if not frequency > 0 or not 0 < duty < 1:
            raise ValueError('Positive pulse frequency and bounded duty required')
        first = math.floor(start * frequency + phase) - 1
        last = math.ceil(end * frequency + phase) + 1
        rows = [((n - phase) / frequency, (n + duty - phase) / frequency, 1.)
                for n in range(first, last + 1)]
    elif profile == 'irregular_pulse':
        rows = [(e['start'], e['end'], e['amplitude']) for e in program['irregular_pulse_events']]
        if (any(not all(math.isfinite(v) for v in row) or not row[0] < row[1]
                or not 0 <= row[2] <= 2 for row in rows)
                or any(a[1] > b[0] for a, b in zip(rows, rows[1:]))):
            raise ValueError('Ordered finite nonoverlapping prescribed pulses required')
    elif profile in ('continuous', 'continuous_varying'):
        return []
    else:
        raise ValueError('Registered force profile required')
    return [row for row in rows if row[1] >= start and row[0] <= end]


def mean_pulse_gate(program, start, end):
    intervals = pulse_intervals(program, start, end)
    profile = program['profile']
    if profile == 'continuous':
        return 1.
    if profile == 'continuous_varying':
        if program['version'] not in (2, 3):
            raise ValueError('Analytic sinusoidal variance program required')
        value = 1.
        for weight, frequency, phase in zip(program['variance_weights'],
                program['variance_frequencies_hz'], program['variance_phases']):
            omega = math.tau * frequency
            average = ((math.cos(omega * start + phase) - math.cos(omega * end + phase))
                       / (omega * (end - start))) if omega else math.sin(phase)
            value += program['variance'] * weight * average
        return value
    integral = sum(amplitude * max(0., min(end, right) - max(start, left))
                   for left, right, amplitude in intervals)
    return integral / (end - start)


def event_packet(program, start_time_s, block_duration_s, requested_lead_times_s):
    end = start_time_s + block_duration_s
    intervals = pulse_intervals(program, start_time_s, end)
    if (not requested_lead_times_s or any(not math.isfinite(t) or not 0 < t <= block_duration_s
            for t in requested_lead_times_s)
            or list(requested_lead_times_s) != sorted(set(requested_lead_times_s))):
        raise ValueError('Ordered unique physical output leads inside this block required')
    events = []
    for left, right, amplitude in intervals:
        duration = right - left
        for at, before, after in [(left, 0., amplitude), (right, amplitude, 0.)]:
            if start_time_s < at <= end:
                offset = at - start_time_s
                events.append([offset, offset / block_duration_s, before, after, duration])
    events.sort(key=lambda row: row[0])
    return dict(schema='rheon.prescribed-pulse-events.v1',
        start_time_s=start_time_s, block_duration_s=block_duration_s,
        requested_lead_times_s=list(requested_lead_times_s),
        pulse_at_start=evaluate(program, start_time_s)['pulse'],
        columns=['offset_s', 'normalized_offset', 'pulse_before', 'pulse_after', 'pulse_duration_s'],
        events=events, edge_interval='start < edge <= end; start amplitude supplied separately',
        future_physical_fields_used=False,
        reference_semantics='Exact prescribed event times; sampled240reference labels remain discretized unless explicitly replaced by a separately qualified reference.')
