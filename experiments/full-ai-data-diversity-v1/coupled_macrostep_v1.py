"""Causal grouped backbone and differentiable restored learned projection.

All spatial weights and modal responses come from managed-restored models.
No native stencil, inverse, future field or supervised RHS enters this module.
"""
import torch
from grouped_macrostep_backbone_v1 import GroupedMacroBackbone
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


class RestoredLinear:
    """Batched autograd execution of the existing ordered learned CSR maps."""
    def __init__(self, model):
        self.model = model
        self.indices = {}

    def __call__(self, fields, shape, name, head, selected_channel=None, sign=1):
        if not fields or any(v.device.type != 'cuda' or v.dtype not in (torch.float32, torch.float64) for v in fields):
            raise ValueError('Restored learned maps require CUDA fields')
        batch = fields[0].shape[:-3]
        if any(v.shape[:-3] != batch for v in fields): raise ValueError('Matching leading batch dimensions required')
        rows, channels, offsets, weights = self.model.sparse[name]
        result = torch.zeros((*batch, *shape), dtype=torch.float64, device='cuda')
        for index in range(rows[head], rows[head + 1]):
            c = channels[index]
            if selected_channel is not None and c != selected_channel: continue
            field = fields[c]
            key = (tuple(field.shape[-3:]), tuple(shape), tuple(offsets[index]))
            if key not in self.indices:
                xyz = [(torch.arange(shape[2-a], device='cuda') + offsets[index][a]).clamp(0, field.shape[-1-a]-1) for a in range(3)]
                self.indices[key] = (xyz[2][:, None, None], xyz[1][None, :, None], xyz[0][None, None, :])
            result = result + (sign * weights[index]) * field[(..., *self.indices[key])].double()
        return result


def wall(value, component):
    axis = value.ndim - 1 - component
    index = torch.arange(value.shape[axis], device='cuda')
    keep = (index > 0) & (index < value.shape[axis]-1)
    shape = [1] * value.ndim; shape[axis] = -1
    return torch.where(keep.reshape(shape), value, torch.zeros_like(value))


class _AnchoredAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, delta):
        return torch.where(delta == 0., base, base + delta)

    @staticmethod
    def backward(ctx, gradient):
        # Algebraic addition has derivative1also at zero. The forward branch
        # preserves the baseline's signed zeros, rather than disabling learning.
        return gradient, gradient


class _RestoredPressure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, core, counts):
        if q.device.type != 'cuda' or q.dtype != torch.float64 or q.shape[:2] != (1, 1):
            raise ValueError('One full double CUDA learned pressure input required')
        ctx.axes = core.geometry_cache[counts][1]
        ctx.divisor = core.divisors['g' + '_'.join(map(str, q.shape[2:]))]
        return core.pressure(q, counts)

    @staticmethod
    def backward(ctx, gradient):
        # Transpose of the restored linear map, including its output gauge.
        # Basis matrices and response are the same frozen learned inputs used
        # by forward; no physical inverse coefficients are constructed.
        value = gradient.reshape(gradient.shape[-3:]).clone()
        origin = torch.zeros_like(value); origin[0, 0, 0] = value.sum()
        value = value - origin
        for dim, basis in enumerate(ctx.axes):
            value = (value.movedim(dim, -1) @ basis.T).movedim(-1, dim)
        value = value / ctx.divisor
        mask = torch.ones_like(value); mask[0, 0, 0] = 0.
        value = value * mask
        for dim in (2, 1, 0):
            value = (value.movedim(dim, -1) @ ctx.axes[dim]).movedim(-1, dim)
        return value.reshape_as(gradient), None, None


class CoupledMacroBackbone(GroupedMacroBackbone):
    def __init__(self, *args):
        super().__init__(*args)
        self.linear = RestoredLinear(self.model)

    @torch.no_grad()
    def context(self, fields, scene, start):
        if set(fields) != set(FIELDS) or any(v.device.type != 'cuda' for v in fields.values()):
            raise ValueError('Complete current CUDA physical state required')
        counts = tuple(scene['counts']); shape = tuple(reversed(counts)); core = self.model
        core.prepare_geometry(counts)
        h, rho = 1. / scene['cells_per_metre'], 1000.
        external = self.controls.generate_from_packet(counts, self.packet(scene, start, 1, 8).to('cuda'))
        current = [fields[n].reshape(s) for n, s in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        features = [dict(pressure=previous.flatten(), **{n: v.flatten() for n, v in zip(FIELDS[1:], current)})]
        offset = 0
        for stage, width in enumerate((2, 2, 3, 1)):
            dt = width / (60. * 8.); core.configure_transport(h, dt)
            addition = external[offset]
            for j in range(1, width): addition = [a + b for a, b in zip(addition, external[offset+j])]
            offset += width
            advected = core.transport(current, counts, dt/h)
            entering = core.affine_ops.three(advected[:3], addition, counts)
            provisional = core.gather_ops.three(core, entering, entering, counts, 'viscosity', (.001/rho*dt/h**2))
            if stage == 3:
                return dict(counts=counts, shape=shape, h=h, rho=rho, dt=dt, previous=previous,
                            provisional=provisional, tracer=advected[3], features=features)
            rhs = core.gather_ops.ordered_rhs(core, provisional, shape, 'flux', 0, h*h, dt)
            history = core.gather_ops.three(core, [previous[0, 0]], None, counts, 'momentum', (h*h)*(1/(rho*h)), multiply_only=True, as_double=True)
            request = core.gather_ops.difference(core, history, rhs, shape, 'flux', 0, -1., rho/h)
            correction = core.pressure(request.reshape(1, 1, *shape), counts)
            pressure = core.affine_ops.binary('sum_gauge', previous, correction)
            velocity = core.gather_ops.three(core, [pressure[0, 0]], provisional, counts, 'momentum', dt*(1/(rho*h)))
            features.append(dict(pressure=pressure.flatten(), **{n: v.flatten() for n, v in zip(FIELDS[1:], [*velocity, advected[3]])}))
            current, previous = [*velocity, advected[3]], pressure
        raise AssertionError('Final fine stage context must be returned')

    def pressure(self, q, counts):
        return _RestoredPressure.apply(q, self.model, counts)

    def rhs(self, fields, context):
        value = torch.zeros((*fields[0].shape[:-3], *context['shape']), dtype=torch.float64, device='cuda')
        for component in range(3):
            part = self.linear(fields, context['shape'], 'flux', 0, component)
            value = value + (context['h'] * context['h']) * part
        return value / context['dt']

    def emit(self, context, velocity_delta=None, tracer_delta=None, float_carriers=True):
        counts = context['counts']; shape = context['shape']
        h, dt, rho, previous = (context[n] for n in ('h', 'dt', 'rho', 'previous'))
        provisional = context['provisional']
        if velocity_delta is not None:
            provisional = [wall(_AnchoredAdd.apply(v.double(), d), c) for c, (v, d) in enumerate(zip(provisional, velocity_delta))]
        rhs = self.rhs(provisional, context)
        history = [wall(self.linear([previous[0, 0]], s, 'momentum', c) * ((h*h)*(1/(rho*h))), c)
                   for c, s in enumerate(dimensions(counts)[:3])]
        action = self.linear(history, shape, 'flux', 0, sign=-1)
        correction = self.pressure(((rhs-action)*(rho/h)).reshape(1, 1, *shape), counts)
        pressure = previous + correction
        pressure = pressure - (previous.flatten()[0] + correction.flatten()[0])
        velocities = [wall(v.double() + self.linear([pressure[0, 0]], s, 'momentum', c)*(dt*(1/(rho*h))), c)
                      for c, (v, s) in enumerate(zip(provisional, dimensions(counts)[:3]))]
        tracer = context['tracer'].double()
        if tracer_delta is not None: tracer = _AnchoredAdd.apply(tracer, tracer_delta)
        if float_carriers:
            velocities = [v.float() for v in velocities]; tracer = tracer.float()
        return dict(pressure=pressure.contiguous().flatten(), **{n: v.contiguous().flatten() for n, v in zip(FIELDS[1:], [*velocities, tracer])})

    def response(self, velocity_delta, context):
        """Unrounded affine correction response; teacher-free and CUDA-only."""
        rhs = self.rhs(velocity_delta, context)
        if rhs.ndim != 3: raise ValueError('One correction direction required by exact learned pressure forward')
        p = self.pressure((rhs*(context['rho']/context['h'])).reshape(1, 1, *context['shape']), context['counts'])
        velocity = [wall(d + self.linear([p[0, 0]], s, 'momentum', c)*(context['dt']*(1/(context['rho']*context['h']))), c)
                    for c, (d, s) in enumerate(zip(velocity_delta, dimensions(context['counts'])[:3]))]
        return p, velocity

    @torch.no_grad()
    def predict_zero(self, fields, scene, start, horizons):
        if horizons not in (1, 2, 3, 4): raise ValueError('One through four complete physical outputs required')
        outputs = []
        for lead in range(horizons):
            context = self.context(fields, scene, start+lead)
            fields = self.emit(context)
            outputs.append(fields)
        return outputs
