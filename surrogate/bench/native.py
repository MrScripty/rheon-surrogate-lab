"""Rheon's own solver as a benchmark subject: one CPU thread, conjugate-gradient pressure solve."""
from ..solver import RheonProbe, probe_for
from .runner import Subject, Unsupported


class NativeRheon(Subject):
    id = 'rheon-native-pcg'
    label = 'Rheon solver · conjugate gradients · 1 CPU thread'
    attributes = {'kind': 'solver', 'device': 'cpu'}

    def __init__(self):
        self._probe = self._seconds = None

    def begin(self, case):
        try:
            binary = probe_for(case.parameters)
        except NotImplementedError as error:
            raise Unsupported(str(error)) from error
        self._probe = RheonProbe(case.parameters, binary, density=case.scene.density.item())

    def advance(self, state, forcing, scene, step):
        self._probe.restore(step, state.map(lambda tensor: tensor[0]))
        self._seconds = self._probe.advance_native()['frame_seconds']
        return [self._probe.state()[1].map(lambda tensor: tensor.unsqueeze(0))]

    def own_seconds(self):
        return self._seconds

    def end(self):
        if self._probe is not None:
            self._probe.close()
            self._probe = self._seconds = None
