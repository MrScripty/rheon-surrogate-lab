"""Check diagnostic metrics against an independent sealed-grid stencil."""
import unittest
import numpy as np
import torch
from context import model
from cpu import relative_residual


class OperatorDiagnostics(unittest.TestCase):
    def test_rectangular_cubic_operator_and_residual(self):
        model.configure('cpu')
        rng = np.random.default_rng(917)
        for shape in ((3,5,7), (1,4,2), (5,1,1)):
            pressure = rng.normal(size=shape)
            expected = np.zeros_like(pressure)
            for axis in range(3):
                for index in np.ndindex(shape):
                    for offset in (-1,1):
                        other = list(index); other[axis] += offset
                        if 0 <= other[axis] < shape[axis]:
                            expected[index] += (.125/1000)*(pressure[index]-pressure[tuple(other)])
            field = torch.from_numpy(pressure).reshape(1,1,*shape)
            sample = {'cell_size_m':torch.tensor([.125],dtype=torch.float64),
                      'rho':torch.tensor([1000.],dtype=torch.float64),
                      'rhs':torch.from_numpy(expected).reshape_as(field)}
            actual = model.full_operator(field,sample['cell_size_m'],sample['rho'])
            np.testing.assert_allclose(actual.numpy().reshape(shape),expected,atol=1e-18,rtol=1e-14)
            self.assertLess(relative_residual(field,sample),1e-14)
            self.assertAlmostEqual(relative_residual(field*2,sample),1.)


if __name__ == '__main__': unittest.main()
