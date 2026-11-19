"""Pure neural pressure heads specialized to the registered small-grid regime."""
import json
from pathlib import Path
import torch
from torch import nn
import routed_base_model
from routed_descriptor import describe

ARCHITECTURE={};SPEC=None


def __getattr__(name):return getattr(routed_base_model,name)


def select(config):
    global SPEC
    SPEC=json.loads(Path(__file__).with_name('routing.json').read_text())
    if SPEC['maximum_small_axis_cells']!=16:raise ValueError('Registered primary regime required')
    routed_base_model.select(config)
    ARCHITECTURE.clear();ARCHITECTURE.update(describe(routed_base_model.ARCHITECTURE,SPEC))


class PressureCorrectionNet(routed_base_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        self.small_coarse_logits=nn.Parameter(torch.zeros(63,dtype=torch.float32))
        self.small_shape_head=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,63))

    def logits_for_shape(self,features,counts):
        if max(counts)<=SPEC['maximum_small_axis_cells']:
            return self.small_coarse_logits+self.small_shape_head(features)
        return self.coarse_logits+self.shape_head(features)
