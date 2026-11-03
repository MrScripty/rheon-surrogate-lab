"""Retain the original fitting stream and observe independently balanced depths."""
import hashlib

import numpy as np
import original_fit_bank as original

DEPTHS = None


def __getattr__(name):
    return getattr(original, name)


def depth_order(cases, exposures, seed):
    if exposures % 4:
        raise ValueError('Complete balanced depth groups required')
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.stack([rng.permutation(np.tile(np.arange(4), exposures // 4))
                     for _ in range(cases)]).astype('<i8')


def bind(cache, updates):
    import fine_tuning
    original.bind(cache, updates)
    stats = fine_tuning.STATS['fundamental_bank']
    count = len(original.SAMPLES)
    global DEPTHS
    DEPTHS = depth_order(count, updates // count, fine_tuning.SPEC['rollout_feedback']['depth_seed'])
    fine_tuning.STATS['rollout_feedback'] = dict(
        mode=fine_tuning.SPEC['rollout_feedback']['mode'],
        depth_order_sha256=hashlib.sha256(DEPTHS.tobytes()).hexdigest(),
        depth_exposures=[[0] * 4 for _ in range(count)], auxiliary_losses=0,
        original_sampling_order_sha256=stats['sampling_order_sha256'])


def one_update(network, optimizer, parents, indices, normalization, cache=None, strategy=None, prepared=None, **extras):
    import fine_tuning
    stats = fine_tuning.STATS['fundamental_bank']
    index = int(original.ORDER[stats['optimizer_examples']])
    depth = int(DEPTHS[index, stats['optimizer_exposures'][index]])
    fine_tuning.ACTIVE_DEPTH = depth
    try:
        result = original.one_update(network, optimizer, parents, indices, normalization, cache, strategy, prepared, **extras)
    finally:
        fine_tuning.ACTIVE_DEPTH = 0
    feedback = fine_tuning.STATS['rollout_feedback']
    feedback['depth_exposures'][index][depth] += 1
    feedback['auxiliary_losses'] += int(fine_tuning.SPEC['rollout_feedback']['mode'] != 'U')
    return result
