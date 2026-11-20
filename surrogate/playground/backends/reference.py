"""Rheon's own solver as a playground backend: the lab's reference probe, one CPU thread.

The probe holds a scene whose own forcing is switched off (acceleration 0). Each step the playground adds the blobs'
velocity to the state, restores that state into the probe, advances one output step with Rheon's conjugate-gradient
pressure solve and reads the state back. It is the reference everything else is measured against, and slow.
"""
from ...fields import FluidState
from ...scenes import scheduled_scene
from ...solver import RheonProbe, probe_path
from .base import Backend, Unsupported, limit_cpu_threads, zero_state

GRIDS = ((9, 11, 11), (15, 6, 8), (16, 16, 16), (16, 24, 40), (32, 24, 48), (32, 32, 32), (64, 64, 64))


class ReferenceBackend(Backend):
    id = 'reference'
    label = 'Rheon reference solver (exact solver) · 1 CPU thread'
    kind = 'solver'
    device = 'cpu'
    forcing = 'every force added to the state before the step'

    def __init__(self):
        super().__init__()
        self._probe = None

    @classmethod
    def grids(cls):
        return GRIDS

    @classmethod
    def available(cls):
        try:
            probe_path()
        except FileNotFoundError as error:
            return False, str(error)
        return True, ''

    def _open(self, spec):
        ok, reason = self.available()
        if not ok:
            raise Unsupported(reason)
        self.close()
        limit_cpu_threads()
        # The probe takes whole cells per metre.
        scene = scheduled_scene(1, list(spec.counts), int(spec.cells_per_metre), 'continuous', 'stationary', steps=2)
        scene['acceleration'] = 0.0
        self._probe = RheonProbe(scene)
        self.state = zero_state(spec.counts, tracer=spec.tracer)

    def fields(self):
        return self.state

    def _set_tracer(self, tracer):
        self.state['tracer'] = tracer

    def _advance(self):
        s = self.state
        if self.forces:
            s = dict(s)
            for name, part in zip('uvw', self.force_increment()):
                s[name] = s[name] + part.float()
        # The probe's own scene never advances past its first step: every call restores at step 0.
        self._probe.restore(0, FluidState(u=s['u'], v=s['v'], w=s['w'], tracer=s['tracer'], pressure=s['pressure']))
        self._probe.advance_native()
        _, state = self._probe.state()
        self.state = dict(u=state.u, v=state.v, w=state.w, pressure=state.pressure, tracer=state.tracer)

    def close(self):
        if self._probe is not None:
            self._probe.close()
            self._probe = None
