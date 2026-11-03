"""Check the residual-to-pressure bound against independent sealed-grid modes."""
import math
from pathlib import Path
import sys
import unittest
import torch

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-stability-learning-v1'))
from learned_iteration import equation_quality, stopping_policy


class SealedGrid:
    ARCHITECTURE = {'schema': 'rheon-physical-residual-pressure-v1'}

    @staticmethod
    def full_operator(pressure, h, rho):
        result = torch.zeros_like(pressure)
        for axis in (2, 3, 4):
            left = [slice(None)]*5
            right = [slice(None)]*5
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            difference = pressure[tuple(left)]-pressure[tuple(right)]
            result[tuple(left)] += difference
            result[tuple(right)] -= difference
        return result*(h/rho).reshape(-1, 1, 1, 1, 1)


class PressureBound(unittest.TestCase):
    original = {'schema': 'rheon-learned-stopping-policy-v1', 'relative_residual': 1e-6,
                'absolute_residual': 1e-12, 'predicted_divergence': 1e-7}
    bounded = {**original, 'schema': 'rheon-learned-stopping-policy-v2',
               'mean_free_pressure_relative_error': 1e-6}

    def sample(self):
        return {'cell_size_m': torch.tensor([.125], dtype=torch.float64),
                'rho': torch.tensor([1000.], dtype=torch.float64),
                'dt': torch.tensor([1/60], dtype=torch.float64)}

    def test_long_wave_error_can_pass_equation_and_fail_pressure_bound(self):
        j = torch.arange(17, dtype=torch.float64)+.5
        low = torch.cos(math.pi*j/17).reshape(1, 1, 1, 1, 17).expand(1, 1, 2, 3, 17)
        high = torch.cos(16*math.pi*j/17).reshape_as(low[:, :, :1, :1]).expand_as(low)
        exact = low+10*high
        prediction = exact-1e-4*low
        sample = self.sample()
        rhs = SealedGrid.full_operator(exact, sample['cell_size_m'], sample['rho'])
        old = equation_quality(SealedGrid, prediction, sample, rhs, self.original)
        new = equation_quality(SealedGrid, prediction, sample, rhs, self.bounded)
        actual = float(torch.linalg.vector_norm(prediction-exact)/torch.linalg.vector_norm(exact))
        self.assertTrue(old['equation_passed'])
        self.assertFalse(new['equation_passed'])
        self.assertFalse(new['pressure_error_bound_passed'])
        self.assertGreater(actual, self.bounded['mean_free_pressure_relative_error'])
        self.assertGreaterEqual(new['mean_free_pressure_relative_error_bound'], actual)

    def test_bound_covers_independent_dense_neumann_solution(self):
        # Dense eigen-decomposition is test-only, independent of the runtime bound.
        generator = torch.Generator().manual_seed(2026103001)
        shape = (3, 4, 5)
        n = math.prod(shape)
        sample = self.sample()
        basis = torch.eye(n, dtype=torch.float64).reshape(n, 1, *shape)
        matrix = SealedGrid.full_operator(basis, sample['cell_size_m'], sample['rho']).reshape(n, n).T
        eigen, vectors = torch.linalg.eigh(matrix)
        for scale in (1e-3, 1e-5, 1e-7):
            exact = torch.randn((1, 1, *shape), generator=generator, dtype=torch.float64)
            exact -= exact.mean()
            rhs = (matrix@exact.flatten()).reshape_as(exact)
            prediction = exact+scale*torch.randn(exact.shape, generator=generator, dtype=torch.float64)
            error = (rhs-SealedGrid.full_operator(prediction, sample['cell_size_m'], sample['rho'])).flatten()
            inverse = vectors[:, 1:]@((vectors[:, 1:].T@error)/eigen[1:])
            result = equation_quality(SealedGrid, prediction, sample, rhs, self.bounded)
            self.assertLessEqual(float(torch.linalg.vector_norm(inverse)),
                                 result['mean_free_pressure_error_bound_pa_l2']*(1+1e-10))
            actual = float(torch.linalg.vector_norm(inverse)/torch.linalg.vector_norm(exact))
            self.assertLessEqual(actual, result['mean_free_pressure_relative_error_bound']*(1+1e-10))

    def test_zero_field_and_invalid_or_nonfinite_profiles(self):
        zeros = torch.zeros((1, 1, 2, 3, 5), dtype=torch.float64)
        result = equation_quality(SealedGrid, zeros, self.sample(), zeros, self.bounded)
        self.assertTrue(result['equation_passed'])
        for bad in (True, 0, 1e-5, float('nan')):
            with self.assertRaisesRegex(ValueError, 'pressure error target'):
                stopping_policy({**self.bounded, 'mean_free_pressure_relative_error': bad})
        with self.assertRaisesRegex(ValueError, 'closed learned'):
            stopping_policy({**self.original, 'mean_free_pressure_relative_error': 1e-6})


if __name__ == '__main__':
    unittest.main()
