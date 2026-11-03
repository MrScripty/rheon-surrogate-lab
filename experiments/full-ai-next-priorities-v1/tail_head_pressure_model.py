"""Pure learned final global block; unchanged parent fine CNN and global heads."""
import json
from pathlib import Path
import torch
from torch import nn
import parent_model
import tail_physical
from tail_head_descriptor import describe

ARCHITECTURE={}


def __getattr__(name):return getattr(parent_model,name)


def select(config):
    parent_model.select(config)
    calibration=json.loads(Path(__file__).with_name('calibration.json').read_text())
    ARCHITECTURE.clear();ARCHITECTURE.update(describe(parent_model.ARCHITECTURE,calibration))


class PressureCorrectionNet(parent_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        self.tail_logits=nn.Parameter(torch.zeros(511,dtype=torch.float32))
        self.tail_shape_head=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,511))
        nn.init.zeros_(self.tail_shape_head[-1].weight)
        nn.init.zeros_(self.tail_shape_head[-1].bias)


def predict_pressure(network,sample,normalization):
    pressure=parent_model.predict_pressure(network,sample,normalization)
    if max(pressure.shape[2:])>16:return pressure
    return tail_physical.append(network,sample,pressure,parent_model.owner)
