"""Focused arithmetic checks for the new observation code, no training."""
import unittest

import torch

import diagnose


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        diagnose.model.configure_cpu()

    def test_spatial_range_separates_constant_channels(self):
        observed = diagnose.stats(torch.tensor([[[[1., 1.]]], [[[8., 8.]]]]))
        self.assertEqual(observed["spatial_range_max"], 0)
        self.assertEqual(observed["max"] - observed["min"], 7)

    def test_copied_stages_match_exact_f32_forward(self):
        torch.manual_seed(20261014)
        network = diagnose.model.PressureCorrectionNet().eval()
        values = torch.linspace(-1, 1, 10 * 16**3).reshape(1, 10, 16, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(network(values), diagnose.copied_forward(network, values)))

    def test_fd_detects_nonconstant_direction_and_gauge_null(self):
        raw = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2) / 10
        objective = lambda v: diagnose.model.gauge_correction(v, 2).square().sum()
        actual = torch.autograd.grad(objective(raw.requires_grad_()), raw)[0]
        direction = diagnose.directions(raw.shape, torch.float64)
        expected = float((actual * direction).sum())
        result = diagnose.difference(objective, raw.detach(), direction, [1e-4, 1e-6])
        for value in result:
            self.assertAlmostEqual(value["central_difference"], expected, places=7)
        constant = diagnose.difference(objective, raw.detach(), torch.ones_like(raw), [1e-4])[0]
        self.assertAlmostEqual(constant["central_difference"], 0, places=8)

    def test_scoped_hooks_remove_all_handles(self):
        network = diagnose.model.PressureCorrectionNet().eval()
        hooks = diagnose.Stages(network)
        with torch.no_grad():
            network(torch.ones(1, 10, 16, 16, 16))
        self.assertIn("bottleneck.layers.0:pre_conv", hooks.values)
        self.assertIn("global_pool:output", hooks.values)
        self.assertIn("head:pre_conv", hooks.values)
        hooks.close()
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in network.modules()))


if __name__ == "__main__":
    unittest.main()
