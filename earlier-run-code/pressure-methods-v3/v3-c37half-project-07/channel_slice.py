import copy
import hashlib
def half_width(network, torch):
    """Keep each stage's first half; decoder concat groups retain their offsets."""
    result = copy.deepcopy(network)
    witness = []
    def convolution(layer, outputs, inputs, name):
        if layer.bias is not None or layer.groups != 1:
            raise ValueError('Only captured bias-free dense convolutions qualified')
        old = layer.weight.detach()
        values = old[outputs][:, inputs].clone()
        layer.weight = torch.nn.Parameter(values, requires_grad=False)
        layer.out_channels = len(outputs)
        layer.in_channels = len(inputs)
        if not torch.equal(layer.weight, old[outputs][:, inputs]):
            raise ValueError('Channel slice differs')
        witness.append(dict(name=name, original_shape=list(old.shape), shape=list(values.shape),
            input_indices=inputs, output_indices=outputs,
            sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest()))
    def block(module, inputs, name):
        selected = inputs
        for index, layer in enumerate(module.layers):
            if isinstance(layer, torch.nn.Conv3d):
                outputs = list(range(layer.out_channels//2))
                convolution(layer, outputs, selected, name+f'.layers.{index}')
                selected = outputs
            elif any(True for _ in layer.parameters()):
                raise ValueError('Unexpected parameterized non-convolution layer')
        return selected
    selected = [0]
    skips = []
    for index, encoder in enumerate(result.encoders):
        selected = block(encoder, selected, f'encoders.{index}')
        skips.append(selected)
    selected = block(result.bottleneck, selected, 'bottleneck')
    for name, layer in result.global_pool.named_modules():
        if isinstance(layer, torch.nn.Conv3d):
            convolution(layer, selected, selected, 'global_pool.'+name)
    original_coarse = network.bottleneck.layers[0].out_channels
    for index, (decoder, skip) in enumerate(zip(result.decoders, reversed(skips))):
        selected = block(decoder, selected+[original_coarse+i for i in skip], f'decoders.{index}')
        original_coarse = network.decoders[index].layers[0].out_channels
    convolution(result.head, [0], selected, 'head')
    return result.eval(), witness
