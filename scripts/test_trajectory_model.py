"""Behavior checks against real Rheon kernels, using an independent 8³ box."""
import unittest

import numpy as np
import torch

from trajectory_experiment import compare, rollout
from trajectory_model import ModalPressure


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = ModalPressure(8)
        states = []
        for seed in (100, 101):
            _, trajectory = rollout(8, "jacobi", seed, 4, capture=True)
            states.extend(trajectory)
        cls.model.fit(torch.from_numpy(np.stack([s["rhs"] for s in states])),
                      torch.from_numpy(np.stack([s["pressure"] for s in states])))

    def test_independent_rollout_matches_reference(self):
        result, *_ = compare(8, 997, 10, self.model)
        self.assertLess(result["max_relative_velocity_l2"], 1e-5)
        self.assertLess(result["max_relative_tracer_l2"], 1e-5)
        self.assertLess(result["max_ai_divergence"], 1e-5)

    def test_zero_linearity_gauge_and_nonlocal_response(self):
        with torch.inference_mode():
            zero = torch.zeros(512, dtype=torch.float64)
            self.assertEqual(torch.count_nonzero(self.model(zero)).item(), 0)
            x = zero.clone()
            x[17], x[18] = 1, -1
            pressure = self.model(x)
            self.assertEqual(pressure[0].item(), 0)
            self.assertGreater(pressure[-1].abs().item(), 1e-3)
            torch.testing.assert_close(self.model(3*x), 3*pressure)

    def test_untrained_model_is_rejected(self):
        with self.assertRaises(RuntimeError):
            rollout(8, "learned", 997, 1, ModalPressure(8))

    def test_nonfinite_training_is_rejected(self):
        model = ModalPressure(8)
        with self.assertRaises(ValueError):
            model.fit(torch.full((2, 512), float("nan"), dtype=torch.float64),
                      torch.ones((2, 512), dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
