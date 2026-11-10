"""Train a DivergenceFreeStepper on its own rollouts.

    python -m surrogate.train <config.json> <output directory> --device cuda

Each update rolls the model forward several passes from a recorded state, feeding
it its own outputs, and compares every output with the recording. Small
divergence-free noise is added to the starting velocity so the model also sees
slightly wrong inputs, as it will in a long rollout. The number of passes grows
on a schedule.

The output directory receives model.pt (averaged weights, what the benchmark
loads), resume.pt, and log.jsonl with one line per report.
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from . import data
from .bench.metrics import relative_error
from .bench.results import git_identity
from .checkpoint import save_stepper
from .fields import curl, remove_mean, seal_potential
from .models import DivergenceFreeStepper, StepperConfig


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    label: str
    train_banks: list
    development_banks: list
    # Which of a bank's trajectories are used, by the split label the bank gives them.
    train_splits: list = field(default_factory=lambda: ['train'])
    development_splits: list = field(default_factory=lambda: ['development'])
    model: dict = field(default_factory=dict)          # StepperConfig fields except the measured scales
    updates: int = 20000
    # (first update, passes per rollout): the rollout grows as training proceeds.
    rollout_schedule: list = field(default_factory=lambda: [[0, 1], [2000, 2], [6000, 4], [12000, 8]])
    cells_per_batch: int = 262144                      # batch size is this over the grid's cell count, at least 1
    learning_rate: float = 3e-4
    warmup_updates: int = 500
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    input_noise: float = 0.01                          # starting-velocity noise, as a fraction of its RMS
    averaging_decay: float = 0.999                     # weight averaging for the saved model
    tracer_weight: float = 1.0
    pressure_weight: float = 0.1
    report_every: int = 100
    evaluate_every: int = 1000
    evaluation_outputs: int = 32
    seed: int = 0

    def passes_at(self, update):
        return max(passes for first, passes in self.rollout_schedule if first <= update)


def _grid_units(scene):
    return scene.timestep / scene.spacing, scene.timestep ** 2 / (scene.density * scene.spacing ** 2)


def _mean_square(*tensors):
    return sum(tensor.double().square().sum() for tensor in tensors) / sum(tensor.numel() for tensor in tensors)


def measure_scales(trajectories, generator, samples=64):
    """Typical field magnitudes in grid units, from random recorded transitions.

    They become the model's fixed input and output scales, so they are measured
    once, before training, from training data only.
    """
    pairs = data.windows(trajectories, 1)
    chosen = [pairs[i] for i in torch.randperm(len(pairs), generator=generator)[:samples].tolist()]
    totals = dict.fromkeys(('velocity', 'velocity_change', 'tracer', 'tracer_change', 'pressure', 'force'), 0.0)
    for selection in chosen:
        start, forcing, future, scene = data.batch([selection], 1)
        to_grid, pressure_to_grid = (value.item() for value in _grid_units(scene))
        after = future.map(lambda tensor: tensor[:, 0])
        totals['velocity'] += _mean_square(after.u, after.v, after.w) * to_grid ** 2
        totals['velocity_change'] += _mean_square(after.u - start.u, after.v - start.v, after.w - start.w) * to_grid ** 2
        totals['tracer'] += _mean_square(after.tracer)
        totals['tracer_change'] += _mean_square(after.tracer - start.tracer)
        totals['pressure'] += _mean_square(remove_mean(after.pressure)) * pressure_to_grid ** 2
        totals['force'] += _mean_square(*forcing) * to_grid ** 2
    scales = {f'{name}_scale': math.sqrt(float(total) / len(chosen)) for name, total in totals.items()}
    silent = [name for name, value in scales.items() if not value > 0.0]
    if silent:
        raise ValueError(f'The sampled training data has no signal for: {", ".join(silent)}.')
    return scales


def add_velocity_noise(state, fraction, generator):
    """Add divergence-free, wall-respecting noise sized as a fraction of each sample's velocity RMS."""
    if fraction <= 0:
        return state
    z, y, x = state.grid
    batch = state.u.shape[0]
    shapes = ((z + 1, y + 1, x), (z + 1, y, x + 1), (z, y + 1, x + 1))
    potential = (torch.randn(batch, *shape, generator=generator).to(state.u) for shape in shapes)
    noise = curl(*seal_potential(*potential))
    rms = lambda *tensors: torch.sqrt(sum(t.square().flatten(1).sum(1) for t in tensors)
                                      / sum(t[0].numel() for t in tensors))
    gain = (fraction * rms(state.u, state.v, state.w) / rms(*noise).clamp_min(1e-30)).reshape(-1, 1, 1, 1)
    return replace(state, u=state.u + gain * noise[0], v=state.v + gain * noise[1], w=state.w + gain * noise[2])


# A field counts as "at rest" below this fraction of its typical magnitude; relative errors
# are measured against at least that much, so a still fluid cannot dominate the loss.
REST_FRACTION = 1e-3


def _relative_square(predicted, reference, rest):
    """Per-sample squared error over squared size, with `rest` (per sample) as the smallest size per value."""
    error = sum((p - r).square().flatten(1).sum(1) for p, r in zip(predicted, reference))
    size = sum(r.square().flatten(1).sum(1) for r in reference)
    values = sum(r[0].numel() for r in reference)
    return (error / (size + rest ** 2 * values)).mean()


def rollout_loss(model, start, forcing, future, scene, config):
    """Mean over outputs of squared relative errors: velocity, tracer and mean-free pressure."""
    k = model.outputs_per_pass
    passes = forcing[0].shape[1] // k
    to_grid, pressure_to_grid = _grid_units(scene)
    scales = model.config
    rest_velocity = REST_FRACTION * scales.velocity_scale / to_grid          # back to m/s, per sample
    rest_pressure = REST_FRACTION * scales.pressure_scale / pressure_to_grid  # back to Pa, per sample
    rest_tracer = REST_FRACTION * scales.tracer_scale
    state, total = start, 0.0
    for p in range(passes):
        outputs = model(state, tuple(f[:, p * k:(p + 1) * k] for f in forcing), scene)
        for offset, output in enumerate(outputs):
            target = future.map(lambda tensor: tensor[:, p * k + offset])
            total = total + _relative_square((output.u, output.v, output.w), (target.u, target.v, target.w),
                                             rest_velocity)
            total = total + config.tracer_weight * _relative_square((output.tracer,), (target.tracer,), rest_tracer)
            total = total + config.pressure_weight * _relative_square(
                (remove_mean(output.pressure),), (remove_mean(target.pressure),), rest_pressure)
        state = outputs[-1]
    return total / (passes * k)


class WindowSampler:
    """Draws batches of windows that share a grid shape, sized to a cell budget."""

    def __init__(self, trajectories, cells_per_batch, seed):
        self.trajectories, self.cells_per_batch = trajectories, cells_per_batch
        self.random = random.Random(seed)
        self._by_intervals = {}

    def draw(self, intervals):
        if intervals not in self._by_intervals:
            groups = {}
            for window in data.windows(self.trajectories, intervals):
                groups.setdefault(window[0].grid, []).append(window)
            if not groups:
                raise ValueError(f'No training trajectory has {intervals} consecutive saved intervals.')
            self._by_intervals[intervals] = groups
        groups = self._by_intervals[intervals]
        # Pick the grid in proportion to its number of windows, then fill the batch from it.
        grid = self.random.choices(list(groups), weights=[len(group) for group in groups.values()])[0]
        size = max(1, self.cells_per_batch // (grid[0] * grid[1] * grid[2]))
        return data.batch(self.random.choices(groups[grid], k=size), intervals)


@torch.no_grad()
def evaluate(model, trajectories, outputs, device):
    """Mean final relative velocity error of own-history rollouts from each trajectory's first long-enough start."""
    k = model.outputs_per_pass
    intervals = (outputs // k) * k
    errors = []
    for trajectory in trajectories:
        starts = trajectory.starts(intervals)
        if not starts:
            continue
        start, forcing, future, scene = data.batch([(trajectory, starts[0])], intervals)
        state, scene = start.to(device), scene.to(device)
        for p in range(intervals // k):
            state = model(state, tuple(f[:, p * k:(p + 1) * k].to(device) for f in forcing), scene)[-1]
        target = future.map(lambda tensor: tensor[:, -1].to(device))
        errors.append(relative_error((state.u, state.v, state.w), (target.u, target.v, target.w)))
    return sum(errors) / len(errors) if errors else None


def train(config, output, device, resume=False, repository=data.REPOSITORY):
    """Run or continue a training run; returns the path of the saved model. Bank paths are relative to `repository`."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    identity = git_identity()
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    training = [t for t in data.load_banks(config.train_banks, repository) if t.split in config.train_splits]
    development = [t for t in data.load_banks(config.development_banks, repository)
                   if t.split in config.development_splits]
    if not training:
        raise ValueError(f'The training banks contain no trajectory with split {config.train_splits}.')
    shared = {t.id for t in training} & {t.id for t in development}
    if shared:
        raise ValueError(f'{len(shared)} trajectories are in both the training and development sets, '
                         f'e.g. {sorted(shared)[0]}.')

    resume_path = output / 'resume.pt'
    saved = torch.load(resume_path, map_location='cpu', weights_only=True) if resume else None
    scales = saved['scales'] if saved else measure_scales(training, generator)
    model = DivergenceFreeStepper(StepperConfig(**config.model, **scales)).to(device)
    averaged = DivergenceFreeStepper(model.config).to(device)
    averaged.load_state_dict(model.state_dict())
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    first = 0
    if saved:
        model.load_state_dict(saved['weights'])
        averaged.load_state_dict(saved['averaged'])
        optimiser.load_state_dict(saved['optimiser'])
        first = saved['update']
    sampler = WindowSampler(training, config.cells_per_batch, config.seed + first)
    (output / 'config.json').write_text(json.dumps({**asdict(config), **identity, 'scales': scales}, indent=1) + '\n')

    def learning_rate(update):
        if update < config.warmup_updates:
            return config.learning_rate * (update + 1) / config.warmup_updates
        progress = (update - config.warmup_updates) / max(1, config.updates - config.warmup_updates)
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def save(update):
        save_stepper(output / 'model.pt', averaged, name=config.name, label=config.label, updates=update,
                     git_commit=identity['git_commit'], training={'git_dirty': identity['git_dirty']})
        torch.save({'weights': model.state_dict(), 'averaged': averaged.state_dict(),
                    'optimiser': optimiser.state_dict(), 'update': update, 'scales': scales}, resume_path)

    started, running = time.perf_counter(), []
    with open(output / 'log.jsonl', 'a') as log:
        for update in range(first, config.updates):
            passes = config.passes_at(update)
            start, forcing, future, scene = sampler.draw(passes * model.outputs_per_pass)
            start = add_velocity_noise(start, config.input_noise, generator).to(device)
            for group in optimiser.param_groups:
                group['lr'] = learning_rate(update)
            loss = rollout_loss(model, start, tuple(f.to(device) for f in forcing), future.to(device),
                                scene.to(device), config)
            if not torch.isfinite(loss):
                raise RuntimeError(f'The loss stopped being finite at update {update}.')
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimiser.step()
            with torch.no_grad():
                for mean, current in zip(averaged.parameters(), model.parameters()):
                    mean.lerp_(current, 1 - config.averaging_decay)
            running.append(loss.item())
            done = update + 1
            if done % config.report_every == 0 or done == config.updates:
                line = {'update': done, 'loss': sum(running) / len(running), 'passes': passes,
                        'learning_rate': learning_rate(update), 'seconds': time.perf_counter() - started}
                running = []
                if done % config.evaluate_every == 0 or done == config.updates:
                    line['development_velocity_error'] = evaluate(averaged.eval(), development,
                                                                  config.evaluation_outputs, device)
                    save(done)
                log.write(json.dumps(line) + '\n')
                log.flush()
                print(json.dumps(line), flush=True)
    return output / 'model.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', required=True, help='cuda or cpu; there is no default so the choice is explicit')
    parser.add_argument('--resume', action='store_true', help='continue from resume.pt in the output directory')
    arguments = parser.parse_args()
    config = TrainingConfig(**json.loads(arguments.config.read_text()))
    print(train(config, arguments.output, arguments.device, arguments.resume))


if __name__ == '__main__':
    main()
