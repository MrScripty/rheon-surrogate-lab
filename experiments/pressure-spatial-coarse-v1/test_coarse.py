"""Independent interpolation, global response and autodiff checks."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).with_name('overlay')))
from coarse import SpatialCoarseCorrection, restriction, prolongation


class CoarseTests(unittest.TestCase):
    def test_volume_restriction_and_interpolation_preserve_constants(self):
        for n in (1, 3, 7, 16, 33, 40):
            r = restriction(n, 'cpu', torch.float64)
            p = prolongation(n, 'cpu', torch.float64)
            torch.testing.assert_close(r.sum(1), torch.ones(4, dtype=torch.float64))
            torch.testing.assert_close(r.sum(0) * n / 4, torch.ones(n, dtype=torch.float64))
            torch.testing.assert_close(p.sum(1), torch.ones(n, dtype=torch.float64))

    def test_zero_initialization_preserves_rng_and_output(self):
        before = torch.random.get_rng_state().clone()
        layer = SpatialCoarseCorrection()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for shape in ((1, 3, 7), (9, 11, 33), (16, 16, 16)):
            self.assertEqual(torch.count_nonzero(layer(torch.randn(1, 1, *shape))).item(), 0)

    def test_far_field_response_survives_gauge_and_is_linear(self):
        layer = SpatialCoarseCorrection()
        with torch.no_grad():
            layer.weight[63, 0] = 1
        x = torch.zeros(1, 1, 16, 16, 16)
        x[:, :, :4, :4, :4] = 1
        out = layer(x)
        self.assertGreater(float((out[0, 0, -1, -1, -1] - out[0, 0, 0, 0, 0]).detach()), .9)
        torch.testing.assert_close(layer(2 * x), 2 * out)

    def test_input_and_weight_gradients(self):
        layer = SpatialCoarseCorrection().double()
        with torch.no_grad():
            layer.weight.normal_(std=.001)
        values = torch.randn(1, 1, 2, 3, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(layer, (values,), fast_mode=True))
        layer(values).square().sum().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertGreater(float(layer.weight.grad.abs().max()), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
