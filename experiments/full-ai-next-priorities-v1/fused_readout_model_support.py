"""Create an inference execution variant with exactly the supplied neural state."""
import torch
from bundle_model_fused_readout import FusedBundleNet
from temporal_train_displacement import inventory
from fused_field_readout import library
from study_support import HERE,read,sha


def wrap(network,norm,identity,assets):
    library()
    before=inventory(network)
    with torch.random.fork_rng(devices=[]):
        variant=FusedBundleNet(network.horizons,network.width,network.architecture,
            forcing_scale=network.forcing_scale,increment_scales=network.increment_scales,
            scales=network.scales,asset_directory=assets).eval()
    variant.load_state_dict(network.state_dict(),strict=True)
    if inventory(variant)!=before:raise ValueError('Execution variant must preserve every neural tensor')
    return variant,norm,dict(identity,execution_variant='fused-fixed-field-readout-v1',
        execution_source_sha256=sha(HERE/'bundle_model_fused_readout.py'),execution_loader_sha256=sha(__file__),
        no_neural_weights_changed=True,inference_only=True)
