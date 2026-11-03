"""Check full physical leads, exact Kdivision and calibration-order admission."""
from copy import deepcopy
import unittest

from reduce_complete_generator_gpu_benchmark import METRICS, validate_row


class ReductionAdmission(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(samples=2,warmups=1,calibration_pairs=2)
        self.case = dict(scene=dict(counts=[2,3,4],cells_per_metre=25),
            reference_leads=[dict(time_s=1/60),dict(time_s=2/60)])
        self.row = dict(counts_xyz=[2,3,4],cells_per_metre=25,K=2,output_dt_s=1/60,block_advance_s=2/60,
            repeat_and_ownership_checks_passed=True,block_ms=[4.,6.],per_output_ms=[2.,3.],
            warmup_block_ms=[7.],first_call_block_ms=8.,
            calibration=[dict(order=['on','off'],hook_on_ms=5.,hook_off_ms=4.),
                         dict(order=['off','on'],hook_on_ms=6.,hook_off_ms=5.)],
            metrics=[dict(lead=1,time_s=1/60,values=dict(pressure_relative_l2=None)),
                     dict(lead=2,time_s=2/60,values=dict(pressure_relative_l2=.25))],
            output_hashes=[{n:'a'*64 for n in ('pressure','vx','vy','vz','tracer')} for _ in range(2)])
        for lead in self.row['metrics']:
            lead['values'] = {n:lead['values'].get(n,0.) for n in METRICS}

    def test_complete_case_with_undefined_relative_error_is_retained(self):
        validate_row(self.row,self.case,dict(K=2),self.protocol,0)

    def test_missing_intermediate_lead_is_rejected(self):
        row = deepcopy(self.row)
        row['metrics'].pop(0)
        with self.assertRaisesRegex(ValueError,'Missing primary,warmup,calibration or physical lead'):
            validate_row(row,self.case,dict(K=2),self.protocol,0)

    def test_division_by_internal_substeps_is_rejected(self):
        row = deepcopy(self.row)
        row['per_output_ms'] = [1.,1.5]
        with self.assertRaisesRegex(ValueError,'Divide block latency only by physical Koutputs'):
            validate_row(row,self.case,dict(K=2),self.protocol,0)

    def test_phase1_must_reverse_calibration_order(self):
        with self.assertRaisesRegex(ValueError,'Calibration order or timing differs'):
            validate_row(self.row,self.case,dict(K=2),self.protocol,1)

    def test_missing_velocity_metric_is_rejected(self):
        row = deepcopy(self.row)
        del row['metrics'][1]['values']['velocity_relative_l2']
        with self.assertRaisesRegex(ValueError,'Native physical timestamp,field hash or scalar differs'):
            validate_row(row,self.case,dict(K=2),self.protocol,0)


if __name__ == '__main__':
    unittest.main()
