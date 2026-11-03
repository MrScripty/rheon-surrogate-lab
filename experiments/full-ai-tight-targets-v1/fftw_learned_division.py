"""Fixed FFTW coordinates divided by derived learned values, never eigenvalues."""
import hashlib
import torch
from torch import nn
from fftw_pressure_transform_v2 import MeasuredDCTMultiplier


class FFTWLearnedDivision(nn.Module):
    def __init__(self, source, library_path):
        super().__init__()
        self.model = source.model
        self.identity = source.identity
        self.library_path = library_path
        self.owners = {}
        self.divisors = {}
        self.encoding = dict(kind='FFTW learned negative-log division',
                             analytic_eigenvalues_or_inverse_scales_used=False,
                             optimizer_updates=0, derived_divisor_inventory={})

    def forward(self, q):
        if q.dtype != torch.float64 or q.device.type != 'cpu' or q.ndim != 5 or q.shape[:2] != (1, 1) or not q.is_contiguous() or not bool(torch.isfinite(q).all()):
            raise ValueError('Complete finite contiguous CPU double learned pressure input required')
        shape = tuple(q.shape)
        key = 'g' + '_'.join(map(str, shape[2:]))
        if shape not in self.owners:
            self.owners[shape] = MeasuredDCTMultiplier(shape, self.library_path)
        if key not in self.divisors:
            divisor = (-self.model.log_gains[key].detach()).exp().reshape(1, 1, *shape[2:]).contiguous()
            divisor[:, :, 0, 0, 0] = 1.
            if not bool(torch.isfinite(divisor).all() and (divisor > 0).all()):
                raise ValueError('Finite positive derived learned divisor required')
            self.divisors[key] = divisor
            self.encoding['derived_divisor_inventory'][key] = dict(dtype='<f8', shape=list(divisor.shape),
                sha256=hashlib.sha256(divisor.numpy().tobytes()).hexdigest())
        owner = self.owners[shape]
        coefficients = torch.empty_like(q)
        output = torch.empty_like(q)
        owner._execute(0, q, coefficients)
        coefficients.div_(self.divisors[key])
        coefficients[:, :, 0, 0, 0] = 0.
        owner._execute(1, coefficients, output)
        output.mul_(1 / (8 * owner.volume))
        return output - output[:, :, :1, :1, :1]

    def close(self):
        for owner in self.owners.values():
            owner.close()
        self.owners = {}
        self.divisors = {}
