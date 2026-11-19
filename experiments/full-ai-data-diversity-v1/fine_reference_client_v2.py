"""Owned fine-state offline reference, including full 20-second trajectories.

The wire clock remains the qualified native 1/(60*factor) clock. Dataset output
cadence is independent: callers retain fine indices j*240/output_rate. No learned
model runs here; the CPU spectral inverse supplies offline reference pressure.
"""
import math
import json
import time

import numpy as np
import torch

from support import BASE
from bounded_native_client import BoundedNativeClient
from continuous_reference_client import ContinuousReferenceClient
from macro_native_client import MacroNativeClient


class FineReferenceClient(BoundedNativeClient):
    def write_json(self, value):
        if not self._wire_pending:
            return BoundedNativeClient.write_json(self, value)
        if value is not self.scene:
            raise ValueError('The first packet must declare its control cadence')
        header = {k: v for k, v in value.items() if k not in ('forcing_program', 'forcing_schedule')}
        controls = value.get('forcing_schedule')
        raw = None
        if controls is not None:
            raw = np.asarray([[*frame['center'], frame['pulse']] for frame in controls], dtype='<f8')
            count = value['steps'] * value.get('forcing_control_substeps', 1)
            if raw.shape != (count, 4) or not 1 <= count <= 38400 or not np.isfinite(raw).all():
                raise ValueError('Complete bounded finite external control sequence required')
            if not ((raw[:, :3] >= 0).all() and (raw[:, :3] <= 1).all()
                    and (raw[:, 3] >= 0).all() and (raw[:, 3] <= 2).all()):
                raise ValueError('External control values outside the native contract')
            header['forcing_binary_controls'] = count
        packet = (json.dumps(header, separators=(',', ':'))+'\n').encode()
        if len(packet) > 65536:
            raise ValueError('Bounded scene header required')
        self._wire_pending = False
        self.child.stdin.write(packet)
        if raw is not None:
            self.child.stdin.write(memoryview(np.ascontiguousarray(raw)).cast('B'))
        self.child.stdin.flush()

    preview_force = MacroNativeClient.preview_force

    def __init__(self, scene, directory, **kwargs):
        self.reference_substeps = scene.get('reference_substeps', 4)
        control_factor = scene.get('forcing_control_substeps', self.reference_substeps)
        count = scene['steps'] * self.reference_substeps
        if (self.reference_substeps not in (1, 2, 4, 8, 16, 32, 64)
                or control_factor not in (1, self.reference_substeps)
                or not 1 <= scene['steps'] <= 1200 or count > 38400):
            raise ValueError('Declared bounded native reference clock required')
        if scene.get('reference_substeps') != self.reference_substeps:
            raise ValueError('Wire reference clock must be explicit')
        self.reference_dt_s = 1 / (60 * self.reference_substeps)
        self.reference_rate_per_s = 60 * self.reference_substeps
        self.maximum_fine_steps = count
        self._fine_step = 0
        self._wire_pending = True
        build = kwargs.pop('build', BASE / 'long-reference-build-01')
        super().__init__(scene, directory, build=build, **kwargs)

    def inspect(self):
        packet, values = super().inspect()
        if (packet['step'] != self._fine_step
                or packet['time_s'] != self._fine_step / self.reference_rate_per_s):
            raise ValueError('Fine native state timestamp differs')
        packet.update(native_step=self._fine_step,
                      reference_dt_s=self.reference_dt_s,
                      reference_rate_per_s=self.reference_rate_per_s)
        return packet, values

    def restore(self, step, state):
        if not isinstance(step, int) or not 0 <= step <= self.maximum_fine_steps:
            raise ValueError('Bounded integer fine restart index required')
        result = super().restore(step, state)
        self._fine_step = step
        self.last_request = None
        return result

    def advance(self, method='spectral', predict=None):
        if method != 'spectral' or predict is not None:
            raise ValueError('This client is an offline spectral reference only')
        if self._fine_step >= self.maximum_fine_steps:
            raise ValueError('Declared reference interval is complete')
        import numerics
        self.write_json(dict(kind='advance', native=False))
        packet = self.read_json()
        cells = math.prod(self.scene['counts'])
        if (packet.get('kind') != 'predict' or packet['cells'] != cells
                or packet['counts_xyz'] != self.scene['counts']
                or packet['cell_size_m'] != 1 / self.scene['cells_per_metre']
                or packet['rho'] != 1000. or packet['dt'] != self.reference_dt_s):
            raise ValueError('Native fine pressure request contract differs')
        shape = (1, 1, *reversed(self.scene['counts']))
        base = torch.from_numpy(self.read_array(cells, '<f8').reshape(shape))
        residual = torch.from_numpy(self.read_array(cells, '<f8').reshape(shape))
        h = torch.tensor([1 / self.scene['cells_per_metre']], dtype=torch.float64)
        rho = torch.tensor([1000.], dtype=torch.float64)
        with torch.inference_mode():
            rhs = residual + numerics.fine_action(base, h, rho)
            tick = time.perf_counter()
            pressure = base + numerics.correction(residual, h, rho, None)
            pressure = pressure - pressure[:, :, :1, :1, :1].clone()
            solve_seconds = time.perf_counter() - tick
        value = np.ascontiguousarray(pressure.numpy(), dtype='<f8')
        self.child.stdin.write(memoryview(value).cast('B'))
        self.child.stdin.flush()
        self.last_request = dict(
            base=base.numpy().reshape(-1).copy(),
            native_input_residual=residual.numpy().reshape(-1).copy(),
            rhs=rhs.numpy().reshape(-1).copy(),
            reference_dt_s=self.reference_dt_s,
            force_time_s=self._fine_step * self.reference_dt_s,
            predict_seconds=solve_seconds,
        )
        frame = self.read_json()
        if frame.get('kind') != 'frame':
            raise ValueError('Native fine reference frame required')
        if frame['frame']['advanced']:
            self._fine_step += 1
        if frame['frame']['step'] != self._fine_step:
            raise ValueError('Native reference fine index differs')
        self.last_frame = frame
        return frame
