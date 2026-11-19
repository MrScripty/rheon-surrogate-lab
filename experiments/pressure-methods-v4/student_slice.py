"""Explicit intermediate-width channel selection, preserving decoder concat groups."""
import copy
import torch

def slice_width(network,numerator,denominator):
    if (numerator,denominator)not in [(3,4),(1,2)]:raise ValueError('Predeclared student widths required')
    out=copy.deepcopy(network);slices=[]
    def conv(layer,outputs,inputs,name):
        if layer.bias is not None or layer.groups!=1:raise ValueError('Captured bias-free dense convolution required')
        prior=layer.weight.detach();values=prior[outputs][:,inputs].clone()
        layer.weight=torch.nn.Parameter(values);layer.out_channels=len(outputs);layer.in_channels=len(inputs)
        slices.append(dict(name=name,input_indices=inputs,output_indices=outputs,shape=list(values.shape)))
    def block(module,inputs,name):
        selected=inputs
        for i,layer in enumerate(module.layers):
            if isinstance(layer,torch.nn.Conv3d):
                outputs=list(range(layer.out_channels*numerator//denominator))
                conv(layer,outputs,selected,name+f'.layers.{i}');selected=outputs
            elif any(True for _ in layer.parameters()):raise ValueError('Unexpected parameterized block module')
        return selected
    selected=[0];skips=[]
    for i,module in enumerate(out.encoders):selected=block(module,selected,f'encoders.{i}');skips.append(selected)
    selected=block(out.bottleneck,selected,'bottleneck')
    for name,layer in out.global_pool.named_modules():
        if isinstance(layer,torch.nn.Conv3d):conv(layer,selected,selected,'global_pool.'+name)
    old_coarse=network.bottleneck.layers[0].out_channels
    for i,(module,skip)in enumerate(zip(out.decoders,reversed(skips))):
        selected=block(module,selected+[old_coarse+j for j in skip],f'decoders.{i}')
        old_coarse=network.decoders[i].layers[0].out_channels
    conv(out.head,[0],selected,'head')
    return out,slices
