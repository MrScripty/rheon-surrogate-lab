"""Same two-channel architecture for zero-feature and explicit-spacing fits."""
import copy
import json
from pathlib import Path
import retained_domain_model

ARCHITECTURE = {}


def __getattr__(name):
    return getattr(retained_domain_model,name)


def select(config):
    retained_domain_model.select(config)
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(retained_domain_model.ARCHITECTURE))
    method = json.loads(Path(__file__).with_name('spacing-method.json').read_text())
    ARCHITECTURE.update(input_channels=2,channels=['dimensionless_laplacian_rhs','spacing_modulated_rhs'],
        explicit_spacing_ablation=method,
        preprocessing='q/train_q_RMS; second channel q/train_q_RMS times log2(h/(1/8 m)) or zero',
        initialization='Original C53 first-convolution column and all other tensors preserved; added column exactly zero')
