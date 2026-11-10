"""Shape-informed global branch; original C53 fine tensors remain unchanged at init."""
import copy
import torch
from torch import nn
import learned_coarse_model

ARCHITECTURE={}

def __getattr__(name):return getattr(learned_coarse_model,name)

def select(config):
    learned_coarse_model.select(config)
    ARCHITECTURE.clear();ARCHITECTURE.update(copy.deepcopy(learned_coarse_model.ARCHITECTURE))
    ARCHITECTURE['shape_informed_coarse']=dict(features='log(Nz/16), log(Ny/16), log(Nx/16)',
        network='3 -> 16 tanh -> 63 additive coarse logits; final layer initially zero',
        added_parameters=1135,head_learning_rate_multiplier=1000,
        fine_initialization='Exact completed v2-headshape CPU restore; matched continuation control',operator='Existing sealed constant-density equation')

class PressureCorrectionNet(learned_coarse_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        self.shape_head=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,63))
        nn.init.zeros_(self.shape_head[2].weight);nn.init.zeros_(self.shape_head[2].bias)
