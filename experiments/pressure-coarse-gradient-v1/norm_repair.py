"""Experimental L2 clipping repair; not installed into any training recipe.

Ordinary finite Torch clipping retains its exact path and bits. Only a rejected
nonfinite reduction with finite float32 gradient components may use scaled
float32 arithmetic. Actual nonfinite components or an unrepresentable final
norm still reject before changing a gradient. No wider network/optimizer dtype.
"""
import math

import torch

ORIGINAL_CLIP = torch.nn.utils.clip_grad_norm_


@torch.no_grad()
def clip_l2_(parameters, max_norm=1.0):
    parameters = list(parameters)
    if not parameters or not math.isfinite(max_norm) or max_norm<=0:
        raise ValueError('Positive finite L2 clip limit and nonempty parameters required')
    gradients = [parameter.grad for parameter in parameters]
    if any(value is None or value.dtype!=torch.float32 or value.device!=parameters[0].device
           for value in gradients):
        raise ValueError('Colocated present float32 training gradients required')
    try:
        return ORIGINAL_CLIP(parameters,max_norm,error_if_nonfinite=True)
    except RuntimeError as error:
        ordinary = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value)for value in gradients]))
        if bool(torch.isfinite(ordinary))or not all(bool(torch.isfinite(value).all())for value in gradients):
            raise
        maximum = torch.stack([value.abs().max()for value in gradients]).max()
        if not bool(maximum>0):
            raise
        scaled = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(value/maximum)for value in gradients]))
        total = maximum*scaled
        if not bool(torch.isfinite(total)):
            raise error
        coefficient = (max_norm/(total+1e-6)).clamp(max=1.0)
        for value in gradients:
            value.mul_(coefficient)
        return total
