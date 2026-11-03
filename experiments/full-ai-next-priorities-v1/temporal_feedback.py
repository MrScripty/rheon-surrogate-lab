"""Differentiable physical reconstruction and deployed encoding between blocks."""
from temporal_features_v2 import packed_state, unpack_outputs


def next_input(predicted, counts, scales):
    if predicted.shape[0] != 1:
        raise ValueError('Registered temporal training uses batch size one')
    physical = unpack_outputs(predicted[:, -1:], counts, scales)[0]
    return packed_state(physical, counts, scales, predicted.device)
