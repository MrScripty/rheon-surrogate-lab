"""Coupled correction with the original ordered CUDA RHS readout forward."""
import torch
from coupled_macrostep_v1 import CoupledMacroBackbone as OriginalCoupledMacroBackbone


class _OrderedReadout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, core, scale, denominator):
        ctx.multiplier = scale / denominator
        return core.affine_ops.ordered_three([a.contiguous(), b.contiguous(), c.contiguous()], scale, denominator)

    @staticmethod
    def backward(ctx, gradient):
        value = gradient * ctx.multiplier
        return value, value, value, None, None, None


class CoupledMacroBackbone(OriginalCoupledMacroBackbone):
    def rhs(self, fields, context):
        parts = [self.linear(fields, context['shape'], 'flux', 0, component) for component in range(3)]
        return _OrderedReadout.apply(*parts, self.model, context['h']*context['h'], context['dt'])
