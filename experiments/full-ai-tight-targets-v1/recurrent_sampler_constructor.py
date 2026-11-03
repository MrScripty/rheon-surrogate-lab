"""Keep frozen tensor version counters even inside an inference caller context."""
import torch
from recurrent_sampler_model import RecurrentSamplerPrototype


def construct(library_path,initialization='neutral',seed=2026102051):
    with torch.inference_mode(False),torch.no_grad():
        return RecurrentSamplerPrototype(library_path,initialization,seed)
