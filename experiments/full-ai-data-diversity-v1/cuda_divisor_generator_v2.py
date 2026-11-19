"""Graph-safe CUDA port: fixed carrier phases uploaded during setup only."""
import torch
from cuda_divisor_generator_v1 import CudaDivisorGenerator


class CudaDivisorGeneratorV2(CudaDivisorGenerator):
    def __init__(self, pair, pressure_pair):
        super().__init__(pair, pressure_pair)
        self.phases = tuple(torch.tensor([0. if kind == axis else .5 for axis in range(3)],
            dtype=torch.float64, device=self.device)[:, None, None, None] for kind in range(4))
        self.asset_identity['carrier_phase_setup'] = 'Fixed phase constants uploaded once before capture; no state-dependent CPU model stage'

    @staticmethod
    def wall(value, component):
        selection = [slice(None)]*3
        selection[2-component] = 0
        value[tuple(selection)] = 0.
        selection[2-component] = -1
        value[tuple(selection)] = 0.
        return value

    def transport(self, fields, counts, alpha):
        coordinates = self.geometry_cache[counts][0]
        outputs = []
        for kind, field in enumerate(fields):
            base = coordinates[kind]
            point = torch.stack([self.linear(fields[:3], field.shape, 'point', kind*3+a) for a in range(3)])
            offset = self.phases[kind]
            query = (base+offset)+(alpha*point)*self.mid_gate[kind, :, None, None, None]
            sampled = []
            for component in range(3):
                phase = self.phases[component]
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
