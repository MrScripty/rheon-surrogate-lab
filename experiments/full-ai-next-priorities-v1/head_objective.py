"""Learn numerical teacher labels using a neural gain predictor; no pressure inverse."""
import torch
from torch.nn import functional as F


def head_name(name):return name=='coarse_logits'or name.startswith('shape_head.')


def freeze_fine(network):
    names=[]
    for name,parameter in network.named_parameters():
        parameter.requires_grad_(head_name(name))
        if parameter.requires_grad:names.append(name)
    if len(names)!=5:raise ValueError('All five original C107 head tensors must own gradients')
    return names


def loss(network,features,target):
    predicted=F.logsigmoid(network.coarse_logits+network.shape_head(features))
    squared=(predicted-target).square()
    return squared.mean()+.1*squared.max(dim=1).values.mean()
