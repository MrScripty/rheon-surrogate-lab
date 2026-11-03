"""Accumulate a complete captured bank cycle before one clipped Adam update."""
import torch
import original_fit_bank as bank


def __getattr__(name):
    return getattr(bank, name)


def bind(cache, updates):
    import fine_tuning
    spec = fine_tuning.SPEC['fitting_bank']
    count = len(spec['grids']) * len(spec['modes_zyx'])
    bank.bind(cache, updates * count)
    fine_tuning.STATS['fundamental_bank'].update(
        optimizer_updates=0, aggregation='mean-complete-balanced-cycle')


def one_update(network, optimizer, parents, indices, normalization, cache=None,
               strategy=None, prepared=None, **extras):
    import fine_tuning
    from gradient_clipping import clip_l2_
    expected = {'analytical_sample', 'analytical_weight', 'supplemental_sample',
                'supplemental_weight', 'direction_sample', 'direction_weight',
                'current_sample', 'current_weight'}
    if (set(extras) != expected or prepared is not None or len(indices) != 1
            or any(extras[n] is not None for n in extras if n.endswith('_sample'))
            or any(extras[n] != 0 for n in extras if n.endswith('_weight'))):
        raise ValueError('A complete balanced bank cycle and no companion losses required')
    stats = fine_tuning.STATS['fundamental_bank']
    count = len(bank.SAMPLES)
    start = stats['optimizer_examples']
    cycle = bank.ORDER[start:start + count]
    if start % count or sorted(cycle.tolist()) != list(range(count)):
        raise ValueError('Exactly one complete balanced cycle required')
    network.train()
    optimizer.zero_grad(set_to_none=True)
    average = 0.
    for index in cycle:
        value = bank.loss(network, bank.SAMPLES[int(index)], normalization) / count
        if not bool(torch.isfinite(value)):
            raise ValueError('Nonfinite aggregate fitting loss')
        value.backward()
        average += float(value.detach())
        del value  # Release this graph before constructing the next case.
    gradient = clip_l2_(network.parameters(), 1.)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in network.parameters()):
        raise ValueError('Nonfinite aggregate fitted weights')
    stats['optimizer_examples'] += count
    for index in cycle:
        stats['optimizer_exposures'][int(index)] += 1
    stats['optimizer_updates'] += 1
    return average, float(gradient)


def observe(network, normalization, update):
    import fine_tuning
    if update % 10:
        return
    stats = fine_tuning.STATS['fundamental_bank']
    stats['trace'].append(dict(update=update, optimizer_updates=update,
        examples=stats['optimizer_examples'], exposures_per_case=update,
        measurements=bank.measure(network, normalization, bank.SAMPLES)))
