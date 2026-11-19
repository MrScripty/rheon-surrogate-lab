"""Exact stationary prefixes for the selected sampled-control reference.

Zero current pressure/velocity and no sampled force before a requested output
leave the sealed-box state unchanged. Analytic between-sample event times do
not change the existing left-sampled240 labels. This does not impose a learned
error tolerance or claim general causal invariance for moving states.
"""
import torch


def stationary_prefix_outputs(fields, packet, output_rate, k, outputs):
    still = torch.ones((), device='cuda', dtype=torch.bool)
    for name in ('pressure', 'vx', 'vy', 'vz'):
        still = still & torch.count_nonzero(fields[name]).eq(0)
    prefix = packet[:, 14].ne(0).cumsum(0)
    stride = 240 // output_rate
    result = []
    for lead, output in enumerate(outputs, 1):
        quiet = still & prefix[lead * stride - 1].eq(0)
        result.append({name: torch.where(quiet, fields[name], value) for name, value in output.items()})
    if len(result) != k:
        raise ValueError('Every requested physical output must be returned')
    return result
