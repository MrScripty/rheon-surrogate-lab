"""Native-memory adapter fixtures; no neural model or simulation is executed."""
import gc
import unittest

import numpy as np
import torch

from full_state_gpu_timing import FIELDS, field_sizes, cpu_views, numpy_outputs, check_owned


class NativeMemoryContract(unittest.TestCase):
    def setUp(self):
        self.sizes = dict(pressure=24, vx=36, vy=32, vz=30, tracer=24)
        self.native = {name: np.arange(size, dtype='<f8' if name == 'pressure' else '<f4')
                       for name, size in self.sizes.items()}

    def outputs(self, k):
        return [{name: torch.arange(size, dtype=torch.float64 if name == 'pressure' else torch.float32)
                 for name, size in self.sizes.items()} for _ in range(k)]

    def test_rectangular_staggered_grid_extents(self):
        self.assertEqual(field_sizes([2, 3, 4]), self.sizes)
        for counts in ([0, 3, 4], [True, 3, 4], [2, 3], [2., 3, 4], '234'):
            with self.assertRaises(ValueError):
                field_sizes(counts)

    def test_native_views_preserve_all_input_bytes_without_conversion(self):
        self.native['pressure'][0] = -0.0
        before = {name: value.tobytes() for name, value in self.native.items()}
        values = cpu_views(np, torch, self.native, self.sizes)
        self.assertEqual(set(values), set(FIELDS))
        for name, value in values.items():
            self.assertEqual(value.numpy().tobytes(), before[name])
            self.assertTrue(np.shares_memory(value.numpy(), self.native[name]))
            self.assertEqual(value.device.type, 'cpu')

    def test_labels_missing_fields_and_extra_fields_are_not_inputs(self):
        for values in ({k: v for k, v in self.native.items() if k != 'pressure'},
                       dict(self.native, rhs=np.zeros(24))):
            with self.assertRaises(ValueError):
                cpu_views(np, torch, values, self.sizes)

    def test_wrong_dtype_extent_and_noncontiguous_input_are_rejected(self):
        for value in (np.zeros(24, dtype='<f4'), np.zeros(25, dtype='<f8'),
                      np.zeros(48, dtype='<f8')[::2]):
            with self.assertRaises(ValueError):
                cpu_views(np, torch, dict(self.native, pressure=value), self.sizes)

    def test_output_views_retain_independent_provider_lifetime(self):
        provider = self.outputs(4)
        values = numpy_outputs(torch, provider, 4, self.sizes)
        check_owned(np, values, self.native)
        expected = [[row[name].tobytes() for name in FIELDS] for row in values]
        del provider
        gc.collect()
        self.assertEqual([[row[name].tobytes() for name in FIELDS] for row in values], expected)
        values[0]['pressure'][0] = -17
        self.assertEqual(values[1]['pressure'][0], 0)
        self.assertEqual(self.native['pressure'][0], 0)

    def test_incomplete_wrong_dtype_and_foreign_device_output_rejected(self):
        with self.assertRaises(ValueError):
            numpy_outputs(torch, self.outputs(1), 2, self.sizes)
        for value in (torch.zeros(24, dtype=torch.float32),
                      torch.empty(24, dtype=torch.float64, device='meta')):
            rows = self.outputs(1); rows[0]['pressure'] = value
            with self.assertRaises(ValueError):
                numpy_outputs(torch, rows, 1, self.sizes)

    def test_aliasing_input_another_output_and_prior_call_is_rejected(self):
        with self.assertRaises(ValueError):
            check_owned(np, [dict(self.native)], self.native)
        independent = numpy_outputs(torch, self.outputs(2), 2, self.sizes)
        independent[1]['pressure'] = independent[0]['pressure']
        with self.assertRaises(ValueError):
            check_owned(np, independent, self.native)
        independent = numpy_outputs(torch, self.outputs(1), 1, self.sizes)
        with self.assertRaises(ValueError):
            check_owned(np, independent, self.native, independent)


if __name__ == '__main__':
    unittest.main()
