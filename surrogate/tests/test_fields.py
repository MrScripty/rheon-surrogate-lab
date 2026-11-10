import unittest

import torch

from surrogate.fields import (FluidState, check_staggering, curl, divergence, faces_as_cell_channels,
                              seal_potential)
from surrogate.tests.support import random_potential, sealed_state

GRID = (5, 6, 7)   # odd and non-cubic on purpose


class FieldOperators(unittest.TestCase):
    def test_curl_of_any_potential_is_divergence_free(self):
        potential = random_potential(2, GRID, torch.Generator().manual_seed(3))
        u, v, w = curl(*potential)
        self.assertEqual(tuple(u.shape[-3:]), (5, 6, 8))
        self.assertEqual(tuple(v.shape[-3:]), (5, 7, 7))
        self.assertEqual(tuple(w.shape[-3:]), (6, 6, 7))
        self.assertLess(divergence(u, v, w).abs().max().item(), 1e-12)

    def test_sealed_potential_has_no_flow_through_walls(self):
        potential = random_potential(2, GRID, torch.Generator().manual_seed(4))
        u, v, w = curl(*seal_potential(*potential))
        walls = (u[..., :, :, 0], u[..., :, :, -1], v[..., :, 0, :], v[..., :, -1, :],
                 w[..., 0, :, :], w[..., -1, :, :])
        for wall in walls:
            self.assertEqual(wall.abs().max().item(), 0.0)
        self.assertGreater(u.abs().max().item(), 0.1)   # the interior is not zeroed

    def test_seal_potential_leaves_its_input_untouched(self):
        potential = random_potential(1, GRID, torch.Generator().manual_seed(5))
        before = [component.clone() for component in potential]
        seal_potential(*potential)
        for component, original in zip(potential, before):
            self.assertTrue(torch.equal(component, original))

    def test_divergence_counts_net_outflow(self):
        u = torch.zeros(1, 2, 2, 3)
        v = torch.zeros(1, 2, 3, 2)
        w = torch.zeros(1, 3, 2, 2)
        u[0, 0, 0, 1] = 2.0   # flow from cell x=0 into cell x=1
        result = divergence(u, v, w)
        self.assertEqual(result[0, 0, 0, 0].item(), 2.0)
        self.assertEqual(result[0, 0, 0, 1].item(), -2.0)
        self.assertEqual(result.abs().sum().item(), 4.0)

    def test_cell_channels_keep_every_face_value(self):
        state = sealed_state(1, GRID)
        channels = faces_as_cell_channels(state.u, state.v, state.w)
        self.assertEqual(tuple(channels.shape), (1, 6, *GRID))
        self.assertTrue(torch.equal(channels[0, 0], state.u[0, :, :, :-1]))
        self.assertTrue(torch.equal(channels[0, 1], state.u[0, :, :, 1:]))
        self.assertTrue(torch.equal(channels[0, 5], state.w[0, 1:, :, :]))

    def test_check_staggering_names_the_wrong_field(self):
        state = sealed_state(1, GRID)
        check_staggering(state)
        wrong = FluidState(u=state.u, v=state.u, w=state.w, tracer=state.tracer, pressure=state.pressure)
        with self.assertRaisesRegex(ValueError, 'v has grid shape'):
            check_staggering(wrong)


if __name__ == '__main__':
    unittest.main()
