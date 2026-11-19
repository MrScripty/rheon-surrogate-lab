"""Equal tier exposure with explicit observed parent/frame work, no shared RNG."""
import math
import numpy as np

STATS = {}


def order(parents, config):
    if config['training']['batch_size'] != 1:
        raise ValueError('Serial single-case updates required')
    groups = {}
    for index, parent in enumerate(parents):
        if parent['entry']['split'] != 'train':
            raise ValueError('No validation parents enter fit sampling')
        size = max(parent['counts_xyz'])
        tier = 16 if size <= 16 else 32 if size <= 32 else 64
        groups.setdefault(tier, []).append(index)
    tiers = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
    updates = config['training']['updates']
    sequence = np.concatenate([rng.permutation(tiers) for _ in range(math.ceil(updates/len(tiers)))])[:updates]
    result = []
    exposures = [0]*len(parents)
    for tier in sequence:
        eligible = groups[int(tier)]
        parent = eligible[int(rng.integers(len(eligible)))]
        frame = int(rng.integers(len(parents[parent]['frames'])))
        result.append([(parent, frame)])
        exposures[parent] += 1
    STATS.update(tier_exposures={str(t):int((sequence == t).sum()) for t in tiers},
                 parent_exposures=exposures, updates=updates,
                 sampling='Equal spatial-tier exposure under a fixed optimizer-update budget; not matched forward-cell cost.')
    return np.asarray(result, dtype='<i8')
