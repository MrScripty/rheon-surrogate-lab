"""One finite streamed epoch over all24 masters and every all12window.

The managed entrypoint owns optimizer creation, input custody, budgets and
checkpoint selection. This module only executes its explicitly supplied epoch.
"""
import math
import random

import torch

from shared_temporal_fit_core_v2 import physical_block


def epoch(model, bank, normalization, optimizer, seed, bounded, progress=None):
    schedule = list(bank.windows)
    random.Random(seed).shuffle(schedule)
    rows = []
    for ordinal, window in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        item = bank.cuda_window(window)
        weight = window['scene_weight'] * window['window_weight'] / 12
        total = 0.
        for example in window['configurations']:
            loss, components = physical_block(model, item['current'], item['scene'],
                example, item['targets'], normalization)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError('Nonfinite training objective before update')
            (loss * weight).backward()
            total += value / 12
            del loss, components
            bounded()
        parameters = list(model.network.parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Every executed shared-network gradient must be finite')
        # Accumulate the norm in F64 so a finite F32 gradient cannot overflow
        # merely when its squares are added. The physical objective is unchanged.
        norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters))
        if not math.isfinite(norm):
            raise ValueError('Nonfinite F64 gradient norm')
        scale = min(1., 1. / max(norm, 1e-300))
        if scale < 1:
            for p in parameters:
                p.grad.mul_(scale)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError('Optimizer produced nonfinite parameters')
        rows.append(dict(identifier=window['identifier'], start_fine_step=window['start_fine_step'],
            blocks=12, physical_outputs=30, objective_before_update=total,
            unscaled_gradient_l2=norm, gradient_clip_multiplier=scale))
        del item
        bounded()
        if progress:
            progress(ordinal, len(schedule), total)
    return dict(updates=len(rows), rows=rows, all24scenes_every_selected_window_and_all12cases=True,
        objective_weights='equal physicalscene, equal within-scene window, equal cadence/K,1/K per physicalhead',
        gradient_clip_maximum_l2=1., gradient_norm_accumulation_dtype='float64')
