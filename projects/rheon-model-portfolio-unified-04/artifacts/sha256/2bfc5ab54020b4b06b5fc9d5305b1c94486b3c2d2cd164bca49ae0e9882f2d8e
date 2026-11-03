"""Numerical-coarse training with a genuinely reduced fine network."""
import copy
import torch
import retained_domain_model
from channel_slice import half_width

ARCHITECTURE = {}
def __getattr__(name): return getattr(retained_domain_model,name)

def select(config):
    retained_domain_model.select(config)
    import json
    from pathlib import Path
    ARCHITECTURE.clear(); ARCHITECTURE.update(copy.deepcopy(retained_domain_model.ARCHITECTURE))
    ARCHITECTURE.update(encoder_widths=[4,8,12,16],bottleneck_width=24,decoder_widths=[16,12,8,4],
        activation='linear',initialization='Exact deterministic channel-sliced named checkpoint; all retained channels trainable',
        numerical_training_method=json.loads(Path(__file__).with_name('hybrid-method.json').read_text()))

class PressureCorrectionNet(retained_domain_model.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        pruned,_ = half_width(self,torch)
        self._modules = pruned._modules
        self.requires_grad_(True)
        if sum(p.numel() for p in self.parameters()) != 92560: raise ValueError('Reduced inventory changed')
