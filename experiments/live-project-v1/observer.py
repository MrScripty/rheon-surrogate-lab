"""Atomic live observation; published previews never establish scientific custody."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np


def atomic(path, value):
    raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('Live publication exceeds 2 MiB')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Observer:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.preview = None
        self.lock = threading.Lock()

    def publish(self, phase, message, completed=0, total=None, state='running', preview=None):
        with self.lock:
            self.sequence += 1
            value = {'schema_version': 1, 'producer': 'Rheon Surrogate Lab', 'sequence': self.sequence,
                     'updated_at': datetime.now(timezone.utc).isoformat(), 'state': state,
                     'phase': phase, 'message': message, 'progress': {'completed': completed, 'total': total}}
            atomic(self.root/'activity.json', value)
            if preview is not None:
                self.preview = preview
            if self.preview is not None:
                atomic(self.root/'frame.json', {**self.preview, 'schema_version': 1, 'session': self.session,
                                               'sequence': self.sequence, 'state': state, 'phase': phase})

    def frame(self, capture, trajectory, index, pressure=None, pressure_label=None):
        directory = Path(capture)/trajectory['trajectory_id']
        tracer = np.frombuffer((directory/f'frame-{index:03d}-tracer.bin').read_bytes(), dtype='<f4')
        if pressure is None:
            pressure = np.frombuffer((directory/f'frame-{index:03d}-pressure.bin').read_bytes(), dtype='<f8')
        pressure = np.asarray(pressure, dtype='<f8').reshape(-1)
        counts = trajectory.get('counts_xyz', [16]*3)
        cells = int(np.prod(counts))
        if tracer.shape != (cells,) or pressure.shape != (cells,) or not np.isfinite(tracer).all() or not np.isfinite(pressure).all():
            raise ValueError('Incomplete/nonfinite external frame')
        return {'label': trajectory['trajectory_id'], 'frame': index,
                'time_s': (index+1)*trajectory['dt_s'], 'counts_xyz': counts,
                'extents_m': trajectory['extents_m'], 'tracer': tracer.tolist(), 'pressure': pressure.tolist(),
                'pressure_label': pressure_label or 'Native reference pressure (Pa); independent verification pending'}


class Watch:
    """One joinable observation thread; never owns or changes the producer."""
    def __init__(self, poll):
        self.poll = poll
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='live-project-observer')

    def _run(self):
        while not self.stop.is_set():
            try:
                self.poll()
            except (OSError, ValueError, KeyError) as error:
                self.error = str(error)
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
