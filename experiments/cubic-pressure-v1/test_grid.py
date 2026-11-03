import unittest
from grid import counts,config,plan,LAB,decode,read
from contract import ContractError, load_module


class GridTests(unittest.TestCase):
    def test_counts_change_with_domain_while_cells_remain_cubic(self):
        for extent,expected in [([1,1,1],[16,16,16]),([1.5,1,.75],[24,16,12]),([2,1,.5],[32,16,8]),([.5,.5,.5],[8,8,8])]:
            self.assertEqual(counts(extent,16),expected)
            self.assertEqual([length/count for length,count in zip(extent,expected)],[.0625]*3)

    def test_fractional_and_over_budget_never_stretch_cells(self):
        for extent,density in [([1.1,1,1],16),([1,1,1],True),([1,1,0],16),([1,1,float('nan')],16),([4,4,4],16)]:
            with self.subTest(extent=extent,density=density),self.assertRaises(ContractError):counts(extent,density)

    def test_new_profile_reports_model_shape_limit_before_training(self):
        value=config(decode(read(LAB/'experiments/cubic-pressure-v1/configs/development.json')))
        observed=plan(value)
        self.assertEqual(observed['grids'][1]['counts_xyz'],[24,16,12])
        self.assertFalse(observed['model_shape_compatible'])
        self.assertEqual(observed['cell_size_m'],.0625)

    def test_training_preflight_and_unchanged_model_reject_rectangles(self):
        import torch
        from unittest.mock import patch
        pipeline = load_module('cubic_pipeline_test', LAB/'experiments/cubic-pressure-v1/pipeline.py')
        model = load_module('unchanged_pressure_model_test', LAB/'experiments/global-pressure-v1/training/model.py')
        value = config(decode(read(LAB/'experiments/cubic-pressure-v1/configs/development.json')))
        with patch.object(torch.optim, 'Adam', side_effect=AssertionError('Optimizer must not start')):
            with self.assertRaisesRegex(ContractError, 'model_shape_unsupported'):
                pipeline.model_preflight(value)
            with self.assertRaisesRegex(model.ModelContractError, 'only exact 16/32 cubes'):
                model.PressureCorrectionNet()(torch.zeros((1,10,12,16,24), dtype=torch.float32))


if __name__=='__main__':unittest.main()
