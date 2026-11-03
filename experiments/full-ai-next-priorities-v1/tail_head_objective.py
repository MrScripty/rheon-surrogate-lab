"""Fit only the final neural tail, using valid modes of offline geometry labels."""
import itertools
import torch
from torch.nn import functional as F


def head_name(name):return name=='tail_logits'or name.startswith('tail_shape_head.')


def freeze_fine(network):
    names=[]
    for name,parameter in network.named_parameters():
        parameter.requires_grad_(head_name(name))
        if parameter.requires_grad:names.append(name)
    if len(names)!=5:raise ValueError('All five new tail parameter tensors must own gradients')
    return names


def loss(network,features,target):
    modes=torch.tensor([row for row in itertools.product(range(8),repeat=3)if sum(row)],device=features.device)
    counts=torch.round(features.exp()*16)
    valid=(modes.unsqueeze(0)<counts.unsqueeze(1)).all(2)
    predicted=F.logsigmoid(network.tail_logits+network.tail_shape_head(features))
    squared=(predicted-target).square()*valid
    return (squared.sum(1)/valid.sum(1)).mean()+.1*squared.max(1).values.mean()
