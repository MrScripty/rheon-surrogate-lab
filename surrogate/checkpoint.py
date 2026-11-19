"""Save and load a stepper with what is needed to rebuild and identify it.

A checkpoint is one .pt file: weights, the model configuration, a name and label,
the number of optimiser updates, and the Git commit of the training code.
"""
import torch

from .models import DivergenceFreeStepper, StepperConfig

FORMAT = 'rheon_surrogate_stepper_v1'


def save_stepper(path, model, *, name, label, updates, git_commit, training=None):
    torch.save({'format': FORMAT, 'config': model.config.to_json(), 'weights': model.state_dict(),
                'name': name, 'label': label, 'updates': updates, 'git_commit': git_commit,
                'training': training or {}}, path)


def load_stepper(path, device='cpu'):
    """Return (model in eval mode on `device`, the saved dictionary)."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError(f'{path} is not a {FORMAT} checkpoint.')
    model = DivergenceFreeStepper(StepperConfig(**saved['config']))
    model.load_state_dict(saved['weights'])
    return model.to(device).eval(), saved
