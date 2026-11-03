"""Independent known-distribution checks for the retained timing summaries."""
import unittest

from summarize_full_state_gpu_timing import samples_summary


class TimingDistribution(unittest.TestCase):
    def test_twenty_sample_linear_percentile_and_median(self):
        result=samples_summary(list(range(20,0,-1)))
        self.assertEqual(result['samples'],20)
        self.assertEqual(result['minimum'],1)
        self.assertEqual(result['median'],10.5)
        self.assertAlmostEqual(result['p95'],19.05)
        self.assertEqual(result['maximum'],20)
        self.assertEqual(samples_summary([2.5]*20)['p95'],2.5)

    def test_incomplete_or_nonphysical_wall_samples_cannot_be_success(self):
        for values in ([],[False],[0.],[-1.],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):
                samples_summary(values)


if __name__=='__main__':
    unittest.main()
