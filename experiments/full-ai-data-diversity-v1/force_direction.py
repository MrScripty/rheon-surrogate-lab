"""Canonical prescribed direction with the native scalar reduction order."""
import math


def direction_phase(seed):
    state, mask = seed, (1 << 64) - 1

    def next_value():
        nonlocal state
        state = (state + 0x9e3779b97f4a7c15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return ((value ^ (value >> 31)) >> 11) / (1 << 53)

    direction = [2 * next_value() - 1 for _ in range(3)]
    square_sum = 0.0
    for value in direction:
        square_sum = square_sum + value * value
    norm = max(math.sqrt(square_sum), 1e-12)
    return [value / norm for value in direction], next_value() * math.tau
