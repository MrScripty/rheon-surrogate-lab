"""Division-form execution of managed learned gains, with no analytical scales."""
import hashlib
import torch
from torch import nn
from modal_basis import bases
from refined_pressure_sampler import load_model as restored_model


class LearnedDivisionPressure(nn.Module):
    def __init__(self, source, policy):
        super().__init__()
        if policy not in ('reciprocal_division', 'negative_log_division'):
            raise ValueError('Only the two predeclared learned-weight representations allowed')
        self.model = source.model
        self.identity = source.identity
        self.policy = policy
        self.axes = {}
        self.divisors = {}
        self.encoding = dict(policy=policy, optimizer_updates=0,
                             analytic_eigenvalues_or_inverse_scales_used=False,
                             exact_raw_learned_parameters_retained=True,
                             derived_divisor_inventory={})

    def forward(self, q):
        if q.dtype != torch.float64 or q.device.type != 'cpu' or q.ndim != 5 or q.shape[:2] != (1, 1) or not bool(torch.isfinite(q).all()):
            raise ValueError('Finite complete CPU double pressure input required')
        shape = tuple(q.shape[2:])
        key = 'g' + '_'.join(map(str, shape))
        if shape not in self.axes:
            self.axes[shape] = bases(shape)
        if key not in self.divisors:
            if self.policy == 'reciprocal_division':
                gains = self.model.gains(key).detach().clone()
                gains[0, 0, 0] = 1.
                divisor = gains.reciprocal()
            else:
                divisor = (-self.model.log_gains[key].detach()).exp()
            divisor[0, 0, 0] = 1.
            if not bool(torch.isfinite(divisor).all() and (divisor > 0).all()):
                raise ValueError('Learned positive finite divisor encoding required')
            self.divisors[key] = divisor
            self.encoding['derived_divisor_inventory'][key] = dict(
                dtype='<f8', shape=list(divisor.shape),
                sha256=hashlib.sha256(divisor.numpy().tobytes()).hexdigest())
        z, y, x = self.axes[shape]
        modes = torch.einsum('kz,bzyx->bkyx', z, q[:, 0])
        modes = torch.einsum('ly,bkyx->bklx', y, modes)
        modes = torch.einsum('mx,bklx->bklm', x, modes)
        modes = modes / self.divisors[key]
        modes[:, 0, 0, 0] = 0.
        result = torch.einsum('mx,bklm->bklx', x, modes)
        result = torch.einsum('ly,bklx->bkyx', y, result)
        result = torch.einsum('kz,bkyx->bzyx', z, result)[:, None]
        return result - result[:, :, :1, :1, :1]

    def close(self):
        self.axes = {}
        self.divisors = {}


def load_model(pair, library_path, pressure_pair, policy):
    model = restored_model(pair, library_path, pressure_pair)
    try:
        source = model.pressure_core
        source.close()
        model.pressure_core = LearnedDivisionPressure(source, policy)
        model.asset_identity['learned_pressure_division_encoding'] = model.pressure_core.encoding
        model.frozen_versions = {name: value._version for name, value in model.named_parameters()}
        return model
    except BaseException:
        model.close()
        raise
