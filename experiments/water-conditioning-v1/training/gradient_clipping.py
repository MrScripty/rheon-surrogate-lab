"""Overflow-only float32 L2 gradient clipping for pressure training.

Ordinary finite Torch clipping retains its exact path and bits. Only a rejected
nonfinite reduction with finite float32 gradient components may use scaled
float32 arithmetic. Actual nonfinite components or an unrepresentable final
norm still reject before changing a gradient. No wider network/optimizer dtype.
"""
import math

import torch

ORIGINAL_CLIP = torch.nn.utils.clip_grad_norm_


@torch.no_grad()
def clip_l2_(parameters, max_norm=1.0, *, on_repair=None):
    parameters = list(parameters)
    if not math.isfinite(max_norm) or max_norm<=0:
        raise ValueError('Positive finite L2 clip limit required')
    try:
        return ORIGINAL_CLIP(parameters,max_norm,error_if_nonfinite=True)
    except RuntimeError as error:
        if not (str(error).startswith("The total norm of order 2.0 for gradients from")
                and "is non-finite" in str(error)):
            raise
        gradients = [parameter.grad for parameter in parameters]
        if not gradients or any(value is None or value.dtype!=torch.float32
                or value.device!=parameters[0].device for value in gradients):
            raise
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
        if on_repair is not None:
            on_repair(dict(ordinary_float32_norm=str(float(ordinary)),
                repaired_float32_norm=float(total), coefficient=float(coefficient),
                maximum_absolute=float(maximum), scaled_float32_norm=float(scaled)))
        return total
