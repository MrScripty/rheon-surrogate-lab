"""Original fine network plus 63 learned low-frequency positive gains."""
import copy
import math
import torch
from torch import nn
import retained_domain_model

ARCHITECTURE = {}


def __getattr__(name):
    return getattr(retained_domain_model,name)


def select(config):
    retained_domain_model.select(config)
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(retained_domain_model.ARCHITECTURE))
    ARCHITECTURE['learned_coarse_revision'] = dict(
        basis='Orthonormal separable Neumann cosine modes 0..3, excluding DC',
        trainable_parameters=63, coefficients='Positive learned sigmoid gains; no analytical inverse eigenvalues',
        bound='max_axis_cells^2/2', initialization='Gain 1 on 16^3; all original neural tensors preserved',
        gate='Accept learned coarse proposal only when its actual fine-equation residual norm does not increase',
        placement='Learned coarse proposal before every fine neural pass',
        limitation='Gate controls coarse proposal only; final neural pass and trajectory still require independent evaluation')


class PressureCorrectionNet(retained_domain_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        self.coarse_logits = nn.Parameter(torch.full((63,),math.log(1/127),dtype=torch.float32))
