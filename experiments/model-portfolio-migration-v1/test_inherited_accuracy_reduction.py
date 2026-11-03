"""Ensure late physical leads and undefined relative errors remain visible."""
import unittest

from reduce_inherited_generator_qualification import summarize


class AccuracyReductionTests(unittest.TestCase):
    def test_late_lead_can_define_the_worst_pressure(self):
        rows = [dict(identifier='scene-a',start=0,K=2,metrics=[
            dict(lead=1,values=dict(pressure_relative_l2=.1)),
            dict(lead=2,values=dict(pressure_relative_l2=.9))])]
        result = summarize(rows)
        self.assertEqual(result['worst_metrics']['pressure_relative_l2'],.9)
        self.assertEqual(result['worst_metric_cases']['pressure_relative_l2'],dict(identifier='scene-a',start=0,lead=2))
        self.assertEqual(result['physical_outputs'],2)

    def test_undefined_error_is_counted_without_becoming_zero(self):
        rows = [dict(identifier='scene-b',start=24,K=2,metrics=[
            dict(lead=1,values=dict(pressure_relative_l2=None)),
            dict(lead=2,values=dict(pressure_relative_l2=.4))])]
        result = summarize(rows)
        self.assertEqual(result['undefined_metrics']['pressure_relative_l2'],1)
        self.assertEqual(result['worst_metrics']['pressure_relative_l2'],.4)

    def test_all_undefined_has_no_fabricated_maximum_or_witness(self):
        rows = [dict(identifier='scene-c',start=64,K=1,metrics=[dict(lead=1,values=dict(velocity_relative_l2=None))])]
        result = summarize(rows)
        self.assertEqual(result['undefined_metrics']['velocity_relative_l2'],1)
        self.assertIsNone(result['worst_metrics']['velocity_relative_l2'])
        self.assertIsNone(result['worst_metric_cases']['velocity_relative_l2'])


if __name__ == '__main__':
    unittest.main()
