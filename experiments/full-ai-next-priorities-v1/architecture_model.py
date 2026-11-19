"""Two predeclared C107 architecture variants; preserve native convolution classes."""
import copy
import json
import math
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import student_original

ARCHITECTURE = {}
SPEC = None


def __getattr__(name):
    return getattr(student_original, name)


def select(config):
    global SPEC
    SPEC = json.loads(Path(__file__).with_name('architecture-variant.json').read_text())
    compatible = copy.deepcopy(config)
    if compatible['schema'] == 'rheon-water-residual-experiment-v12':
        compatible['schema'] = 'rheon-water-residual-experiment-v11'
        compatible['correction']['steps'] = 2
    student_original.select(compatible)
    import baseline_model
    baseline_model.CORRECTION = copy.deepcopy(config['correction'])
    ARCHITECTURE.clear()
    ARCHITECTURE.update(copy.deepcopy(student_original.ARCHITECTURE))
    ARCHITECTURE.update(correction=copy.deepcopy(config['correction']), architecture_variant=SPEC)
    if SPEC['variant'] == 'redistributed':
        ARCHITECTURE.update(encoder_widths=[3,8,16,20], bottleneck_width=28, decoder_widths=[20,16,8,3])
    else:
        ARCHITECTURE['learned_coarse_revision'].update(
            basis='Orthonormal separable cosine modes 0..7 excluding DC', trainable_parameters=511)
        ARCHITECTURE['shape_informed_coarse'].update(network='3 -> 64 tanh -> 511 additive learned logits', added_parameters=33471)


def transform(network):
    if SPEC['variant'] == 'redistributed':
        def convolution(layer, outputs, inputs):
            if layer.bias is not None or layer.groups != 1:
                raise ValueError('Bias-free dense C107 convolutions required')
            multiplicity = Counter(inputs)
            weight = layer.weight.detach()[outputs][:, inputs].clone()
            weight /= weight.new_tensor([multiplicity[i] for i in inputs])[None,:,None,None,None]
            layer.weight = nn.Parameter(weight)
            layer.in_channels = len(inputs)
            layer.out_channels = len(outputs)

        def block(module, inputs, width):
            selected = inputs
            for layer in module.layers:
                if isinstance(layer, nn.Conv3d):
                    outputs = [i % layer.out_channels for i in range(width)]
                    convolution(layer, outputs, selected)
                    selected = outputs
                elif any(True for _ in layer.parameters()):
                    raise ValueError('Parameter-free normalization required')
            return selected

        selected = [0]
        skips = []
        for module,width in zip(network.encoders,[3,8,16,20]):
            selected = block(module, selected, width)
            skips.append(selected)
        original_coarse = network.bottleneck.layers[0].out_channels
        selected = block(network.bottleneck, selected, 28)
        for layer in network.global_pool.modules():
            if isinstance(layer, nn.Conv3d):
                convolution(layer, selected, selected)
        for module,skip,width in zip(network.decoders,reversed(skips),[20,16,8,3]):
            next_original = module.layers[0].out_channels
            selected = block(module, selected+[original_coarse+i for i in skip], width)
            original_coarse = next_original
        convolution(network.head,[0],selected)
    elif SPEC['variant'] == 'one-stage-modes8':
        old_logits = network.coarse_logits.detach()
        old_head = network.shape_head
        # Constant initial high-mode gain. No eigenvalues or analytical inverse.
        logits = old_logits.new_full((511,), math.log(.125/(128-.125)))
        mapped = [z*64+y*8+x-1 for z in range(4) for y in range(4) for x in range(4) if z+y+x]
        logits[mapped] = old_logits
        network.coarse_logits = nn.Parameter(logits)
        head = nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,511)).to(old_logits.device)
        with torch.no_grad():
            head[0].weight[:16].copy_(old_head[0].weight)
            head[0].bias[:16].copy_(old_head[0].bias)
            head[2].weight.zero_();head[2].bias.zero_()
            head[2].weight[mapped,:16] = old_head[2].weight
            head[2].bias[mapped] = old_head[2].bias
        network.shape_head = head
    else:
        raise ValueError('Registered architecture variant required')
    return network


class PressureCorrectionNet(student_original.PressureCorrectionNet):
    def __init__(self):
        super().__init__()
        transform(self)
