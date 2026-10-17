"""Captured model variant: unchanged residual network plus learned coarse map.

The assembler captures the exact selected model as base_model.py. Physics,
preprocessing, recurrence and losses retain their existing owners. This module
owns only construction and evaluation of the new architectural addition.
"""
import copy
import base_model as base
from coarse import SpatialCoarseCorrection

ARCHITECTURE = {}


def __getattr__(name):
    return getattr(base, name)


def select(config):
    base.select(config)
    if config['correction']['activation'] != 'linear' or config['normalization'] != 'none':
        raise ValueError('spatial coarse experiment requires its linear no-normalization control')
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(base.ARCHITECTURE),
        spatial_coarse_correction={
            'schema': 'learned-spatial-coarse-v1', 'bins_zyx': [4, 4, 4],
            'restriction': 'fine-cell overlap averages',
            'mixing': 'dense bias-free 64-to-64 trainable matrix',
            'prolongation': 'linear cell-centre interpolation with clamped edges',
            'scale': '(longest cell count / 16)^2',
            'initialization': 'zeros; original network RNG stream unchanged',
            'inference_inverse_solver': False})


class PressureCorrectionNet(base.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        self.spatial_coarse = SpatialCoarseCorrection()

    def forward(self, values):
        return super().forward(values) + self.spatial_coarse(values)
