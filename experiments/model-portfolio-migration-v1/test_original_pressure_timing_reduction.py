"""Reject misleading timing evidence; retain signed calibration observations."""
import copy
import unittest

from reduce_original_pressure_timing import validate_case


class ReductionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(samples_per_case=2,warmups_per_case=1,calibration_pairs_per_case=2)
        self.baseline = dict(trajectory_id='a',parent=0,native_frame=4,counts_xyz=[3,2,1],cell_size_m=.1,
            input_sha256={'base':'b','native_input_residual':'r','cell_size_m':'h','rho':'rho','dt':'dt'},
            reference_pressure_sha256='p',reference_rhs_sha256='rhs',output_sha256='output')
        identity = {name:value for name,value in self.baseline.items() if name != 'output_sha256'}
        self.case = dict(id='case-000',original_case=identity,cells_per_metre=10.,input_dt_s=1/60)
        self.row = dict(self.baseline,case_index=0,case_id='case-000',cells_per_metre=10.,input_dt_s=1/60,
            output_independent_of_inputs=True,raw_wall_nanoseconds=[100,200],median_ms=.00015,
            instrumented_first_call_ns=800,instrumented_warmup_ns=[400],calibration=[
                dict(pair_index=0,order=['on','off'],on_ns=100,off_ns=120,on_minus_off_ns=-20),
                dict(pair_index=1,order=['off','on'],on_ns=150,off_ns=110,on_minus_off_ns=40)])

    def validate(self,row=None):
        return validate_case(self.row if row is None else row,self.baseline,self.case,0,0,self.protocol)

    def test_negative_overhead_retained_and_primary_samples_untouched(self):
        samples,differences = self.validate()
        self.assertEqual(samples,[100,200])
        self.assertEqual(differences,[-20,40])

    def test_changed_output_or_reference_rejected(self):
        for field in ('output_sha256','reference_pressure_sha256'):
            row = copy.deepcopy(self.row)
            row[field] = 'corrupt'
            with self.assertRaisesRegex(ValueError,'identities'):
                self.validate(row)

    def test_instrumented_or_divided_primary_summary_rejected(self):
        row = copy.deepcopy(self.row)
        row['median_ms'] = .000075
        with self.assertRaisesRegex(ValueError,'raw samples/statistic'):
            self.validate(row)

    def test_nonpositive_clock_and_wrong_calibration_order_rejected(self):
        row = copy.deepcopy(self.row)
        row['raw_wall_nanoseconds'][0] = 0
        with self.assertRaisesRegex(ValueError,'raw samples/statistic'):
            self.validate(row)
        row = copy.deepcopy(self.row)
        row['calibration'][0]['order'] = ['off','on']
        with self.assertRaisesRegex(ValueError,'Calibration order'):
            self.validate(row)


if __name__ == '__main__':
    unittest.main()
