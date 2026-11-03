"""Complete explicit-time generator integrating the untrained residual subnet.

Fine groups retain the qualified frozen step. Each corrected coarse group uses
one coarse prediction and one learned correction projection. K whole outputs
are sequential; joint endpoint/interior generation remains a separate track.
"""
import math
import torch

from temporal_group_projector_v2 import TemporalGroupProjector
from temporal_frozen_history_v1 import TemporalFrozenBackbone as FineBackbone
from temporal_residual_network_v1 import TemporalResidual, features, physical_deltas, PARAMETERS
from temporal_control_packets_v1 import control_packet
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


class TemporalResidualGenerator(TemporalGroupProjector):
    def __init__(self, *args):
        super().__init__(*args)
        self.network = TemporalResidual().eval().requires_grad_(False)
        self.asset_identity.update(temporal_residual_prototype=True,
            new_residual_parameters=PARAMETERS, residual_optimizer_updates=0,
            default_groups='One whole coarse interval at30/60outputs/s; preserved fine steps at120outputs/s',
            learned_pressure_calls_per_group='1forfine group;2forcorrected coarse group',
            true_joint_temporal_generation=False, accuracy_and_latency_require_qualification=True)

    def state_dict(self):
        result = dict(super().state_dict())
        result.update({'temporal_residual.' + n: v for n, v in self.network.state_dict().items()})
        return result

    @torch.inference_mode()
    def predict(self, fields, scene, start_fine, output_rate, k, groups=None):
        rate = 60 * scene['reference_substeps']
        if (rate not in (240, 480) or output_rate not in (30, 60, 120)
                or type(k) is not int or k not in (1, 2, 3, 4) or scene.get('sgs', False)):
            raise ValueError('Explicit registered physical cadence andKrequired')
        stride = rate // output_rate
        if groups is None:
            groups = (1,) * stride if output_rate == 120 else (stride,)
        groups = tuple(groups)
        if not groups or any(type(n) is not int or not 1 <= n <= 32 for n in groups) or sum(groups) != stride:
            raise ValueError('Group widths must partition a complete physical output interval')
        counts = tuple(scene['counts'])
        if set(fields) != set(FIELDS):
            raise ValueError('Complete current native state required')
        for name, size in [('pressure', tuple(reversed(counts))), *zip(FIELDS[1:], dimensions(counts))]:
            value = fields[name]
            if (value.device.type != 'cuda' or not value.is_contiguous()
                    or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(size)):
                raise ValueError('Exact contiguous native CUDA fields required')
        packet = torch.from_numpy(control_packet(scene, start_fine, k * stride)).to('cuda')
        return self.predict_packet(fields, scene, k, groups, packet)

    @torch.inference_mode()
    def predict_packet(self, fields, scene, k, groups, packet):
        if all(n == 1 for n in groups):
            return FineBackbone.predict_packet(self, fields, scene, k, groups, packet)
        offset, outputs = 0, []
        for _ in range(k):
            for width in groups:
                if width == 1:
                    fields = FineBackbone.predict_packet(self, fields, scene, 1, (1,), packet.narrow(0, offset, 1))[0]
                else:
                    phase = self.phase_packet(fields, scene, width, packet.narrow(0, offset, width))
                    delta = physical_deltas(self.network(*features(phase)), phase)
                    fields = self.emit(phase, *delta)
                offset += width
            outputs.append(fields)
        return outputs
