"""Keep neural parameters and loss fixed; declare the numerical pass explicitly."""
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
    method = json.loads(Path(__file__).with_name('hybrid-method.json').read_text())
    ARCHITECTURE['numerical_training_method'] = method
    ARCHITECTURE['method_comparison']['sampling'] = 'Same expanded-domain tier sampler and fixed optimizer-update budget'
