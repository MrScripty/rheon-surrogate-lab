import unittest

import torch

from surrogate.fields import divergence
from surrogate.models import DivergenceFreeStepper, StepperConfig, UNet3d
from surrogate.tests.support import forcing, scene, sealed_state

SMALL = dict(widths=(8, 12, 16), blocks_per_level=1)


def randomise_heads(model, seed=7):
    """Heads start at zero; give them weights so outputs differ from the input."""
    generator = torch.Generator().manual_seed(seed)
    for head in (model.potential_x, model.potential_y, model.potential_z, model.tracer_change, model.pressure):
        with torch.no_grad():
            head.weight.copy_(0.1 * torch.randn(head.weight.shape, generator=generator, dtype=head.weight.dtype))


class Trunk(unittest.TestCase):
    def test_keeps_grid_shape_for_odd_small_and_flat_grids(self):
        trunk = UNet3d(3, widths=(4, 6, 8, 8), blocks_per_level=1)
        for grid in ((5, 6, 7), (1, 3, 9), (16, 16, 16), (2, 2, 2)):
            output = trunk(torch.randn(2, 3, *grid))
            self.assertEqual(tuple(output.shape), (2, 4, *grid))

    def test_every_cell_can_influence_every_other_cell(self):
        trunk = UNet3d(1, widths=(4, 6), blocks_per_level=1)
        for block in list(trunk.encoders) + list(trunk.decoders) + [trunk.bottleneck]:
            for residual in block:
                torch.nn.init.normal_(residual.second.weight, std=0.1)
        field = torch.zeros(1, 1, 24, 24, 24, requires_grad=True)
        trunk(field)[0, 0, 0, 0, 0].backward()
        self.assertGreater(field.grad[0, 0, -1, -1, -1].abs().item(), 0.0)


class Stepper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def advance(self, grid=(5, 6, 7), outputs=2, batch=2):
        model = DivergenceFreeStepper(StepperConfig(outputs_per_pass=outputs, **SMALL)).double()
        randomise_heads(model)
        state = sealed_state(batch, grid)
        return model, state, model(state, forcing(batch, outputs, grid), scene(batch))

    def test_untrained_model_returns_its_input(self):
        model = DivergenceFreeStepper(StepperConfig(outputs_per_pass=3, **SMALL)).double()
        state = sealed_state(1, (4, 5, 6))
        outputs = model(state, forcing(1, 3, (4, 5, 6)), scene(1))
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertTrue(torch.allclose(output.u, state.u, atol=1e-12))
            self.assertTrue(torch.equal(output.tracer, state.tracer))
            self.assertEqual(output.pressure.abs().max().item(), 0.0)

    def test_outputs_stay_divergence_free_and_sealed_with_arbitrary_weights(self):
        _, state, outputs = self.advance()
        for output in outputs:
            self.assertGreater((output.u - state.u).abs().max().item(), 1e-6)   # it did change
            self.assertLess(divergence(output.u, output.v, output.w).abs().max().item(), 1e-9)
            for wall in (output.u[..., 0], output.u[..., -1], output.v[..., 0, :], output.v[..., -1, :],
                         output.w[..., 0, :, :], output.w[..., -1, :, :]):
                self.assertLess(wall.abs().max().item(), 1e-12)

    def test_total_tracer_is_conserved_and_pressure_is_mean_free(self):
        _, state, outputs = self.advance()
        for output in outputs:
            self.assertGreater((output.tracer - state.tracer).abs().max().item(), 1e-6)
            drift = (output.tracer.sum(dim=(-3, -2, -1)) - state.tracer.sum(dim=(-3, -2, -1))).abs().max()
            self.assertLess(drift.item(), 1e-9)
            self.assertLess(output.pressure.mean(dim=(-3, -2, -1)).abs().max().item(), 1e-6)

    def test_batch_entries_do_not_affect_each_other(self):
        model, state, outputs = self.advance(batch=2)
        first = state.map(lambda tensor: tensor[:1])
        grid = state.grid
        alone = model(first, tuple(f[:1] for f in forcing(2, 2, grid)), scene(1))
        self.assertTrue(torch.allclose(alone[1].u, outputs[1].u[:1], atol=1e-10))

    def test_forcing_must_cover_every_output(self):
        model = DivergenceFreeStepper(StepperConfig(outputs_per_pass=2, **SMALL)).double()
        with self.assertRaisesRegex(ValueError, 'cover 2 output intervals'):
            model(sealed_state(1, (4, 4, 4)), forcing(1, 1, (4, 4, 4)), scene(1))

    def test_default_model_is_a_few_million_parameters(self):
        count = sum(p.numel() for p in DivergenceFreeStepper().parameters())
        self.assertGreater(count, 2_000_000)
        self.assertLess(count, 8_000_000)


if __name__ == '__main__':
    unittest.main()
