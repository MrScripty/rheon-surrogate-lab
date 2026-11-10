"""The training loop runs end to end on a tiny synthetic bank. This checks the machinery, not learning quality."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from surrogate import train
from surrogate.bench.subjects import LearnedStepper
from surrogate.fields import divergence
from surrogate.tests.support import sealed_state
from surrogate.tests.test_data import write_bank


def config(**changes):
    settings = dict(name='unit-run', label='Unit run', train_banks=['bank.json'], development_banks=['bank.json'],
                    model={'outputs_per_pass': 1, 'widths': [6, 8], 'blocks_per_level': 1}, updates=4,
                    rollout_schedule=[[0, 1], [2, 2]], cells_per_batch=240, warmup_updates=2, report_every=2,
                    evaluate_every=4, evaluation_outputs=2, input_noise=0.05)
    return train.TrainingConfig(**{**settings, **changes})


class Training(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        write_bank(self.root, dense_states=8)
        quiet = contextlib.redirect_stdout(io.StringIO())   # train() prints each report line
        quiet.__enter__()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(quiet.__exit__, None, None, None)

    def train(self, settings, resume=False):
        return train.train(settings, self.root / 'run', 'cpu', resume, repository=self.root)

    def test_runs_saves_a_loadable_model_and_logs_each_report(self):
        output = self.root / 'run'
        path = self.train(config())
        lines = [json.loads(line) for line in (output / 'log.jsonl').read_text().splitlines()]
        self.assertEqual([line['update'] for line in lines], [2, 4])
        self.assertEqual([line['passes'] for line in lines], [1, 2])     # the rollout grew on schedule
        self.assertTrue(all(line['loss'] > 0 for line in lines))
        self.assertIsNotNone(lines[-1]['development_velocity_error'])
        saved = json.loads((output / 'config.json').read_text())
        self.assertRegex(saved['git_commit'], r'^[0-9a-f]{40}$')
        self.assertGreater(saved['scales']['velocity_scale'], 0)
        subject = LearnedStepper(path, 'cpu')
        self.assertEqual((subject.id, subject.attributes['training_updates']), ('unit-run', 4))
        self.assertEqual(subject.model.config.velocity_scale, saved['scales']['velocity_scale'])

    def test_resume_continues_from_the_saved_update(self):
        output = self.root / 'run'
        self.train(config())
        self.train(config(updates=6), resume=True)
        updates = [json.loads(line)['update'] for line in (output / 'log.jsonl').read_text().splitlines()]
        self.assertEqual(updates, [2, 4, 6])

    def test_a_bank_without_training_trajectories_is_refused(self):
        (self.root / 'empty.json').write_text(json.dumps({'trajectories': []}))
        with self.assertRaisesRegex(ValueError, 'no trajectory with split'):
            self.train(config(train_banks=['empty.json']))

    def test_a_trajectory_may_not_be_both_fitted_and_used_for_development(self):
        with self.assertRaisesRegex(ValueError, 'both the training and development sets'):
            self.train(config(development_splits=['train', 'development']))


class Pieces(unittest.TestCase):
    def test_noise_is_divergence_free_sealed_and_sized_to_the_velocity(self):
        state = sealed_state(3, (5, 6, 7), dtype=torch.float32)
        noisy = train.add_velocity_noise(state, 0.1, torch.Generator().manual_seed(0))
        added = (noisy.u - state.u, noisy.v - state.v, noisy.w - state.w)
        self.assertLess(divergence(*added).abs().max().item(), 1e-4)
        self.assertEqual(added[0][..., 0].abs().max().item(), 0.0)
        rms = lambda *t: torch.sqrt(sum(x.square().flatten(1).sum(1) for x in t) / sum(x[0].numel() for x in t))
        ratio = rms(*added) / rms(state.u, state.v, state.w)
        self.assertTrue(torch.allclose(ratio, torch.full((3,), 0.1), rtol=1e-4))
        self.assertIs(train.add_velocity_noise(state, 0.0, None), state)

    def test_rollout_schedule_picks_the_latest_stage_reached(self):
        schedule = config(rollout_schedule=[[0, 1], [10, 2], [30, 8]])
        self.assertEqual([schedule.passes_at(update) for update in (0, 9, 10, 29, 30, 99)], [1, 1, 2, 2, 8, 8])

    def test_loss_is_zero_for_a_perfect_rollout_and_one_per_field_for_a_zero_prediction(self):
        from surrogate.fields import Scene

        class Fixed(torch.nn.Module):
            outputs_per_pass = 1
            # Scales that put the "at rest" level at a thousandth of this test's unit-sized fields:
            # dt/h for velocity and dt^2/(rho h^2) for pressure, for the scene below.
            config = train.StepperConfig(velocity_scale=16 / 60, pressure_scale=256 / 3600 / 1000)

            def __init__(self, answers):
                super().__init__()
                self.answers = iter(answers)

            def forward(self, state, forcing, scene):
                return [next(self.answers)]

        future = sealed_state(2, (4, 4, 4), seed=3, dtype=torch.float32).map(
            lambda tensor: torch.stack((tensor, 0.5 * tensor), dim=1))
        start = sealed_state(2, (4, 4, 4), dtype=torch.float32)
        forcing = tuple(torch.zeros(2, 2, *shape) for shape in ((4, 4, 5), (4, 5, 4), (5, 4, 4)))
        one = torch.ones(2)
        scene = Scene(spacing=one / 16, timestep=one / 60, density=1000 * one, viscosity=1e-3 * one)
        settings = config(tracer_weight=1.0, pressure_weight=1.0)
        steps = [future.map(lambda tensor: tensor[:, index]) for index in range(2)]
        self.assertLess(train.rollout_loss(Fixed(steps), start, forcing, future, scene, settings).item(), 1e-12)
        zeros = [step.map(torch.zeros_like) for step in steps]
        self.assertAlmostEqual(train.rollout_loss(Fixed(zeros), start, forcing, future, scene, settings).item(),
                               3.0, places=3)


if __name__ == '__main__':
    unittest.main()
