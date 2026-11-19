"""Create an inference execution variant with exactly the supplied neural state."""
import torch
from bundle_model_fast_preparation import FastPreparationBundleNet
from temporal_train_displacement import inventory
from fused_field_readout import library
from study_support import HERE,read,sha


def wrap(network,norm,identity,assets):
    library()
    from fused_force_construction import library as forcing_library
    forcing_library()
    before=inventory(network)
    with torch.random.fork_rng(devices=[]):
        variant=FastPreparationBundleNet(network.horizons,network.width,network.architecture,
            forcing_scale=network.forcing_scale,increment_scales=network.increment_scales,
            scales=network.scales,asset_directory=assets).eval()
    variant.load_state_dict(network.state_dict(),strict=True)
    if inventory(variant)!=before:raise ValueError('Execution variant must preserve every neural tensor')
    return variant,norm,dict(identity,execution_variant='fused-readout-fast-preparation-v1',
        execution_source_sha256=sha(HERE/'bundle_model_fast_preparation.py'),execution_loader_sha256=sha(__file__),
        no_neural_weights_changed=True,inference_only=True)
