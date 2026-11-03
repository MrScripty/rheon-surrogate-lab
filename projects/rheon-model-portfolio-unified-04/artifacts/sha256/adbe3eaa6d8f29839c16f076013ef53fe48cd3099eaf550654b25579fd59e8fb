"""Repeat each frozen 1200-draw sample twice for matched one-stage encoder work."""
import copy
import hashlib
import original_fit_sampling
import numpy as np

STATS = {}


def order(parents, config):
    updates = config['training']['updates']
    if updates not in (1200,2400):
        raise ValueError('Registered architecture work budget required')
    compatible = copy.deepcopy(config)
    compatible['training']['updates'] = 1200
    values = original_fit_sampling.order(parents, compatible)
    factor = updates//1200
    values = np.repeat(values,factor,axis=0)
    STATS.clear();STATS.update(copy.deepcopy(original_fit_sampling.STATS))
    STATS.update(updates=updates,sampled_cells=STATS['sampled_cells']*factor,
        tier_exposures={k:v*factor for k,v in STATS['tier_exposures'].items()},
        frame_exposures={k:v*factor for k,v in STATS['frame_exposures'].items()},
        sample_order_sha256=hashlib.sha256(values.tobytes()).hexdigest(),sample_repeat_factor=factor,
        matched_fine_network_passes=updates*config['correction']['steps'])
    return values
