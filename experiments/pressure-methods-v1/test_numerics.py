"""Independent stencil/eigenmode, nullspace and transfer checks."""
import math
import unittest
import torch
import numerics


class NumericalQualification(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.h = torch.tensor([.125], dtype=torch.float64)
        self.rho = torch.tensor([1000.], dtype=torch.float64)

    def test_full_inverse_on_odd_thin_and_single_cell_grids(self):
        rng = torch.Generator().manual_seed(2026101801)
        for shape in [(7, 5, 9), (1, 3, 8), (2, 1, 1), (1, 1, 1)]:
            with self.subTest(shape=shape):
                p = torch.randn((1, 1, *shape), dtype=torch.float64, generator=rng)
                p -= p[:, :, :1, :1, :1].clone()
                rhs = numerics.fine_action(p, self.h, self.rho)
                got = numerics.correction(rhs, self.h, self.rho, None)
                torch.testing.assert_close(got, p, rtol=2e-12, atol=2e-12)

    def test_low_mode_eigenpair_and_unrepresented_mode(self):
        shape = (7, 9, 11)
        for mode, represented in [((3, 2, 1), True), ((0, 0, 4), False)]:
            factors = [torch.cos(math.pi*k*(torch.arange(n, dtype=torch.float64)+.5)/n)
                       for n, k in zip(shape, mode)]
            p = torch.einsum('z,y,x->zyx', *factors)[None, None]
            eigenvalue = sum(4*math.sin(math.pi*k/(2*n))**2 for n, k in zip(shape, mode))
            rhs = p * eigenvalue * (self.h/self.rho).item()
            torch.testing.assert_close(numerics.fine_action(p, self.h, self.rho), rhs,
                                       rtol=1e-12, atol=1e-18)
            got = numerics.correction(rhs, self.h, self.rho)
            expected = p-p[:, :, :1, :1, :1].clone() if represented else torch.zeros_like(p)
            torch.testing.assert_close(got, expected, rtol=2e-12, atol=2e-12)

    def test_constant_pressure_is_null_and_zero_rhs_is_exact(self):
        p = torch.ones((1, 1, 3, 5, 1), dtype=torch.float64)
        self.assertEqual(float(numerics.fine_action(p, self.h, self.rho).abs().max()), 0.)
        self.assertEqual(float(numerics.correction(p*0, self.h, self.rho).abs().max()), 0.)

    def test_incompatible_rhs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            numerics.correction(torch.ones((1, 1, 3, 3, 3), dtype=torch.float64), self.h, self.rho)

    def test_spacing_density_scaling_and_batch(self):
        p = torch.arange(105, dtype=torch.float64).reshape(1, 1, 3, 5, 7)
        p = torch.cat((p, -p))
        h = torch.tensor([.125, .0625], dtype=torch.float64)
        rho = torch.tensor([1000., 500.], dtype=torch.float64)
        rhs = numerics.fine_action(p, h, rho)
        torch.testing.assert_close(numerics.correction(rhs, h, rho, None), p, rtol=1e-12, atol=1e-11)

    def test_partial_correction_is_idempotent(self):
        rng = torch.Generator().manual_seed(17)
        p = torch.randn((1, 1, 7, 9, 11), generator=rng, dtype=torch.float64)
        rhs = numerics.fine_action(p, self.h, self.rho)
        delta = numerics.correction(rhs, self.h, self.rho)
        remaining = rhs-numerics.fine_action(delta, self.h, self.rho)
        torch.testing.assert_close(numerics.correction(remaining, self.h, self.rho),
                                   torch.zeros_like(delta), rtol=0, atol=1e-12)

    def test_invalid_physics_and_nonfinite_rejected(self):
        rhs = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float64)
        with self.assertRaises(ValueError):
            numerics.correction(rhs, -self.h, self.rho)
        rhs[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            numerics.correction(rhs, self.h, self.rho)


if __name__ == '__main__':
    unittest.main()
