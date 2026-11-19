"""The subjects the benchmark can score that need nothing but this package."""
import torch

from ..checkpoint import load_stepper
from .runner import Subject


class Persistence(Subject):
    """Returns the state it was given. Any useful model must beat this error."""
    id = 'persistence'
    label = 'No change (persistence)'
    attributes = {'kind': 'baseline'}

    def advance(self, state, forcing, scene, step):
        return [state]


class LearnedStepper(Subject):
    """A trained DivergenceFreeStepper checkpoint, run on `device`."""

    def __init__(self, path, device, identity=None):
        self.model, saved = load_stepper(path, device)
        self.device = torch.device(device)
        self.outputs_per_pass = self.model.outputs_per_pass
        self.id = identity or saved['name']
        self.label = saved['label']
        self.attributes = {'kind': 'learned', 'device': self.device.type,
                           'parameters': sum(p.numel() for p in self.model.parameters()),
                           'training_updates': saved['updates'], 'training_commit': saved['git_commit']}

    @torch.no_grad()
    def advance(self, state, forcing, scene, step):
        outputs = self.model(state.to(self.device, torch.float32),
                             tuple(f.to(self.device, torch.float32) for f in forcing), scene.to(self.device))
        # .cpu() waits for the device to finish, so the runner's timer sees the whole pass.
        return [output.to('cpu') for output in outputs]
