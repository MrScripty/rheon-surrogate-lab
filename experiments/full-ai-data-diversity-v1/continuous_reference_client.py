"""Owned offline reference with bounded binary fine-time external controls."""
from support import BASE
from bounded_native_client import BoundedNativeClient
from macro_native_client import MacroNativeClient
from continuous_force_program import SUBSTEPS
import json
import numpy as np


class ContinuousReferenceClient(MacroNativeClient):
    def __init__(self, scene, directory, **kwargs):
        self.reference_substeps = scene.get('reference_substeps', 1)
        control_factor = scene.get('forcing_control_substeps', 1)
        if self.reference_substeps not in SUBSTEPS or control_factor not in (1, self.reference_substeps):
            raise ValueError('Explicit reference timestep and control cadence required')
        if not 1 <= scene['steps'] <= 600:
            raise ValueError('Bounded output count required')
        self._wire_pending = True
        self.reference_dt_s = 1/(60*self.reference_substeps)
        self._macro_step = 0
        self.last_native_frames = []
        BoundedNativeClient.__init__(self, scene, directory,
            build=BASE/'continuous-reference-build-01', **kwargs)

    def write_json(self, value):
        if not self._wire_pending:
            return super().write_json(value)
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
