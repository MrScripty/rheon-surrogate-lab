"""Single-condition fixtures from two genuine four-output raw captures."""
import copy
import json
from pathlib import Path
import statistics
import unittest

from qualify_common_generator_measurements import ROOT, raw_scalars


class RawBlockOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT/'runs/model-portfolio-migration-v1'
        original = json.loads((ROOT/'docs/plans/model-portfolio-migration-v1/reports/joint-generator-case-preflight-01.json').read_bytes())
        cls.case = original['cases'][0]
        cls.phases = []
        for phase in (0,1):
            report = json.loads((folder/f'joint-generator-GPU-benchmark-01/b4-phase{phase}.json').read_bytes())
            cls.phases.append(next(r for r in report['rows']
                if (r['identifier'],r['start']) == (cls.case['identifier'],cls.case['start'])))

    def test_original_qualified_case_matches_literal_raw_samples(self):
        a,b = self.phases
        result = raw_scalars(a,b,self.case,4)
        self.assertEqual(result['pressure_relative_l2'],max(r['values']['pressure_relative_l2'] for r in a['metrics']))
        self.assertEqual(result['block_median_ms'],statistics.median(a['block_ms']+b['block_ms']))

    def test_missing_intermediate_lead_identifies_lead_boundary(self):
        a,b = copy.deepcopy(self.phases)
        for row in (a,b):
            del row['metrics'][1]
        with self.assertRaisesRegex(ValueError,'^Raw block omits an ordered physical lead$'):
            raw_scalars(a,b,self.case,4)

    def test_last_lead_controls_worst_error(self):
        a,b = copy.deepcopy(self.phases)
        for row in (a,b):
            for lead,value in zip(row['metrics'],(0.1,0.2,0.3,0.9),strict=True):
                lead['values']['pressure_relative_l2'] = value
        self.assertEqual(raw_scalars(a,b,self.case,4)['pressure_relative_l2'],0.9)

    def test_incorrect_amortization_identifies_raw_timing_boundary(self):
        a,b = copy.deepcopy(self.phases)
        for row in (a,b):
            row['per_output_ms'] = [value/8 for value in row['block_ms']]
        with self.assertRaisesRegex(ValueError,'^Raw samples, ownership or per-output divisor differs$'):
            raw_scalars(a,b,self.case,4)

    def test_undefined_intermediate_error_cannot_pass_as_zero(self):
        a,b = copy.deepcopy(self.phases)
        for row in (a,b):
            row['metrics'][1]['values']['pressure_relative_l2'] = None
        with self.assertRaisesRegex(ValueError,'^Undefined lead cannot be silently omitted from block error$'):
            raw_scalars(a,b,self.case,4)

    def test_changed_grid_density_is_a_case_mismatch(self):
        a,b = copy.deepcopy(self.phases)
        for row in (a,b):
            row['cells_per_metre'] *= 2
        with self.assertRaisesRegex(ValueError,'^Raw physical case, density, dimensions or horizon differs$'):
            raw_scalars(a,b,self.case,4)


if __name__ == '__main__':
    unittest.main()
