"""CUDA-only full-state execution of the restored learned divisor prototype.

CPU work is limited to typed artifact loading and fixed geometry preparation.
All state-dependent learned stages execute on CUDA. No physical inverse or
native simulation function is used. This version has no optimizer.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from support import ROOT, read, digest
from recurrent_sampler_assets import source_tensors, csr
from recurrent_sampler_native import dimensions
from restored_modal_core import RestoredModalCore
from modal_basis import bases
from legacy_exact_forcing import schedule
from scheduled_force_tables import force
import json

FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')


def prescribed_inputs(scene, start, horizons, device='cuda'):
    """Known scene/time controls only; includes host construction and upload."""
    if horizons not in (1, 2, 3, 4) or not 0 <= start < start+horizons <= scene['steps']:
        raise ValueError('Bounded complete control block required')
    inputs = schedule(json.dumps(scene, sort_keys=True, separators=(',', ':')))
    return [[v.to(device=device) for v in force(inputs, start+lead)[0]]
            for lead in range(horizons)]


class CudaDivisorGenerator(nn.Module):
    def __init__(self, pair='runs/full-ai-tight-targets-v1/sampler-head-pair-01',
                 pressure_pair='runs/full-ai-tight-targets-v1/mixed-modal-pair-11'):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA inference required; CPU fallback is forbidden')
        self.device = torch.device('cuda', torch.cuda.current_device())
        tensors, self.asset_identity = source_tensors()
        self.sparse = {}
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor.to(self.device))
            rows, channels, offsets, weights = csr(tensor)
            self.sparse[name] = (rows.tolist(), channels.tolist(), offsets.tolist(),
                                 weights.to(self.device))
        pair = ROOT/pair
        receipt, restored = read(pair/'receipt.json'), read(pair/'restore/receipt.json')
        if (receipt['status'] != 'completed' or receipt['optimizer_updates'] != 440
                or not all(receipt['artifact_parity'].values())
                or restored['optimizer_updates'] != 0 or not restored['full_cpu_probe_bitwise_verified']):
            raise ValueError('Existing exact managed CUDA fit and historical CPU restore required')
        for key in ('training_run_id', 'restore_run_id'):
            terminal = read(pair/('terminal-'+receipt[key]+'.json'))
            if terminal['state'] != 'completed' or terminal['custody_phase'] != 'drained':
                raise ValueError('Completed managed owner custody required')
        archive = pair/'restore/weights.npz'
        if digest(archive) != restored['outputs']['weights.npz']['sha256']:
            raise ValueError('Transport checkpoint changed')
        changes = {}
        with np.load(archive, allow_pickle=False) as source:
            if set(source.files) != {'mid_gate', 'mix'}:
                raise ValueError('Exact 48-parameter learned head required')
            for name in source.files:
                value = source[name]
                spec = restored['final_inventory'][name]
                if (value.dtype.str != '<f8' or list(value.shape) != spec['shape']
                        or hashlib.sha256(value.tobytes()).hexdigest() != spec['sha256']):
                    raise ValueError('Typed learned head identity changed')
                # Same uniform deployment quantization as the CPU prototype.
                raw = torch.from_numpy(value.copy())
                quantized = (raw/(2.**-40)).round()*(2.**-40)
                changes[name] = float((raw-quantized).abs().max())
                self.register_buffer(name, quantized.to(self.device))
        # Constructor loads/checks artifacts only; it performs no CPU forward.
        loaded = RestoredModalCore(ROOT/pressure_pair, library_path=None)
        self.pressure_model = loaded.model.to(self.device)
        self.asset_identity.update(transport_pair=str(pair.relative_to(ROOT)),
            transport_weights_sha256=digest(archive), transport_quantization_step=2.**-40,
            transport_quantization_changes=changes, pressure=loaded.identity,
            inference_device='cuda', numerical_pressure_solver=False,
            analytical_physical_inverse=False, optimizer_updates=0)
        loaded.close()
        self.index_cache, self.geometry_cache, self.divisors = {}, {}, {}
        self.eval().requires_grad_(False)

    def prepare_geometry(self, counts):
        """Fixed coordinates and basis only; report separately from steady state."""
        counts = tuple(counts)
        if counts in self.geometry_cache:
            return
        shape = tuple(reversed(counts))
        key = 'g'+'_'.join(map(str, shape))
        if key not in self.pressure_model.log_gains:
            raise ValueError('No restored learned pressure geometry; no fallback')
        coordinate_sets = []
        for carrier in dimensions(counts):
            z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64, device=self.device)
                                      for n in carrier), indexing='ij')
            coordinate_sets.append(torch.stack((x, y, z)))
        # This is a fixed transform basis, identical to the original CPU basis.
        axes = tuple(value.to(self.device) for value in bases(shape))
        self.geometry_cache[counts] = (coordinate_sets, axes)
        gains = self.pressure_model.gains(key)
        self.divisors[key] = torch.where(gains != 0, gains.reciprocal(), torch.ones_like(gains))

    def linear(self, fields, target_shape, name, head, selected_channel=None, sign=1):
        rows, channels, offsets, weights = self.sparse[name]
        result = torch.zeros(target_shape, dtype=torch.float64, device=self.device)
        for index in range(rows[head], rows[head+1]):
            channel = channels[index]
            if selected_channel is not None and channel != selected_channel:
                continue
            field = fields[channel]
            offset = tuple(offsets[index])
            key = (tuple(field.shape), tuple(target_shape), offset)
            if key not in self.index_cache:
                xyz = [(torch.arange(target_shape[2-a], device=self.device)+offset[a])
                       .clamp(0, field.shape[2-a]-1) for a in range(3)]
                self.index_cache[key] = (xyz[2][:, None, None], xyz[1][None, :, None], xyz[0][None, None, :])
            sample = field[self.index_cache[key]].double()
            result = result + (sign*weights[index])*sample
        return result

    @staticmethod
    def interpolate(field, query, clip=True):
        dims = tuple(reversed(field.shape))
        lo, hi, fractions = [], [], []
        for axis, size in enumerate(dims):
            position = query[axis].clamp(0, size-1)
            lower = position.floor().long()
            lo.append(lower); hi.append((lower+1).clamp_max(size-1)); fractions.append(position-lower)
        flat = field.flatten().double()
        result = torch.zeros_like(query[0])
        lower = torch.full_like(result, float('inf')); upper = -lower
        for bz in range(2):
            for by in range(2):
                for bx in range(2):
                    bits = (bx, by, bz)
                    index = [hi[d] if bits[d] else lo[d] for d in range(3)]
                    value = flat[index[0]+dims[0]*(index[1]+dims[1]*index[2])]
                    weight = torch.ones_like(result)
                    for d in range(3):
                        weight = weight*(fractions[d] if bits[d] else 1-fractions[d])
                    result = result+weight*value
                    if clip:
                        lower = torch.minimum(lower, value); upper = torch.maximum(upper, value)
        return torch.minimum(torch.maximum(result, lower), upper) if clip else result

    @staticmethod
    def wall(value, component):
        selection = [slice(None)]*3
        selection[2-component] = [0, -1]
        value[tuple(selection)] = 0.
        return value

    def transport(self, fields, counts, alpha):
        coordinates = self.geometry_cache[counts][0]
        outputs = []
        for kind, field in enumerate(fields):
            base = coordinates[kind]
            point = torch.stack([self.linear(fields[:3], field.shape, 'point', kind*3+a) for a in range(3)])
            offset = torch.tensor([0. if kind == a else .5 for a in range(3)],
                                  dtype=torch.float64, device=self.device)[:, None, None, None]
            query = (base+offset)+(alpha*point)*self.mid_gate[kind, :, None, None, None]
            sampled = []
            for component in range(3):
                phase = torch.tensor([0. if component == a else .5 for a in range(3)],
                                     dtype=torch.float64, device=self.device)[:, None, None, None]
                sampled.append(self.interpolate(fields[component], query-phase))
            delta = []
            for axis in range(3):
                value = torch.zeros_like(sampled[0])
                for component in range(3):
                    value = value+self.mix[kind, axis, component]*sampled[component]
                delta.append((alpha*value).clamp(-1, 1))
            output = self.interpolate(field, base+torch.stack(delta), clip=False).float()
            # Mirror the original CPU scalar tail clipping policy.
            tail = field.shape[-1] % 4
            if tail:
                clipped = self.interpolate(field, base+torch.stack(delta), clip=True).float()
                output[..., -tail:] = clipped[..., -tail:]
            outputs.append(self.wall(output, kind) if kind < 3 else output)
        return outputs

    def pressure(self, q, counts):
        z, y, x = self.geometry_cache[counts][1]
        key = 'g'+'_'.join(map(str, q.shape[2:]))
        value = torch.einsum('kz,bzyx->bkyx', z, q[:, 0])
        value = torch.einsum('ly,bkyx->bklx', y, value)
        value = torch.einsum('mx,bklx->bklm', x, value)
        value = value/self.divisors[key]
        value[:, 0, 0, 0] = 0.
        value = torch.einsum('mx,bklm->bklx', x, value)
        value = torch.einsum('ly,bklx->bkyx', y, value)
        value = torch.einsum('kz,bkyx->bzyx', z, value)[:, None]
        return value-value[:, :, :1, :1, :1]

    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons, external):
        counts = tuple(scene['counts']); shape = tuple(reversed(counts))
        if (horizons not in (1, 2, 3, 4) or scene['cells_per_metre'] not in (8, 16)
                or scene.get('sgs', False) or counts not in self.geometry_cache):
            raise ValueError('Prepared registered reference-regime geometry required')
        if set(fields) != set(FIELDS) or len(external) != horizons:
            raise ValueError('Complete current state and prescribed input block required')
        for name, expected_shape in [('pressure', shape), *zip(FIELDS[1:], dimensions(counts))]:
            value = fields[name]
            if (value.device != self.device or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(expected_shape) or not value.is_contiguous()):
                raise ValueError('Exact native dtype and contiguous CUDA state required')
        for block in external:
            if len(block) != 3 or any(v.device != self.device or v.dtype != torch.float64
                                    or tuple(v.shape) != size for v, size in zip(block, dimensions(counts)[:3])):
                raise ValueError('Complete CUDA double prescribed velocities required')
        current = [fields[name].reshape(size) for name, size in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        dt, h, rho = 1/60, 1/scene['cells_per_metre'], 1000.
        results = []
        for lead in range(horizons):
            advected = self.transport(current, counts, dt/h)
            entering = [(value.double()+addition).float() for value, addition in zip(advected[:3], external[lead])]
            provisional = [self.wall((entering[c].double()+self.linear(entering, size, 'viscosity', c)
                                      *(.001/rho*dt/h**2)).float(), c)
                           for c, size in enumerate(dimensions(counts)[:3])]
            rhs = torch.zeros(shape, dtype=torch.float64, device=self.device)
            for channel in range(3):
                rhs = rhs+(h*h)*self.linear(provisional, shape, 'flux', 0, channel)
            rhs = rhs/dt
            history = []
            scale = (h*h)*(1/(rho*h))
            for component, size in enumerate(dimensions(counts)[:3]):
                history.append(self.wall(self.linear([previous[0, 0]], size, 'momentum', component)*scale, component))
            action = self.linear(history, shape, 'flux', 0, sign=-1)
            residual = rhs-action
            correction = self.pressure((residual*(rho/h)).reshape(1, 1, *shape), counts)
            pressure = previous+correction
            pressure = pressure-pressure[:, :, :1, :1, :1]
            velocity = [self.wall((provisional[c].double()+self.linear([pressure[0, 0]], size, 'momentum', c)
                                   *(dt*(1/(rho*h)))).float(), c)
                        for c, size in enumerate(dimensions(counts)[:3])]
            results.append(dict(pressure=pressure.contiguous().flatten(),
                vx=velocity[0].flatten(), vy=velocity[1].flatten(), vz=velocity[2].flatten(), tracer=advected[3].flatten()))
            current, previous = [*velocity, advected[3]], pressure
        return results

    def close(self):
        self.index_cache.clear(); self.geometry_cache.clear(); self.divisors.clear()
