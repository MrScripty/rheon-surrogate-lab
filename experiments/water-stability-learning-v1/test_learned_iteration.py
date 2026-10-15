"""Residual feedback and budget on a two-cell sealed pressure problem."""
import unittest
import torch
from learned_iteration import correct


class TwoCells:
    @staticmethod
    def full_operator(pressure,h,rho):
        difference=pressure[...,0]-pressure[...,1]
        return torch.stack((difference,-difference),dim=-1)*(h/rho).reshape(-1,1,1,1,1)

    @staticmethod
    def predict_pressure(network,sample,normalization):
        # 90% of the exact two-cell correction, using no reference labels.
        delta=sample['native_input_residual']*(sample['rho']/sample['cell_size_m']).reshape(-1,1,1,1,1)*.45
        delta=delta-delta[...,:1]
        return sample['base']+delta


class Iteration(unittest.TestCase):
    def sample(self):
        base=torch.tensor([0.,1.],dtype=torch.float64).reshape(1,1,1,1,2)
        return {'base':base,'native_input_residual':torch.tensor([-.001,.001],dtype=torch.float64).reshape_as(base),
            'cell_size_m':torch.tensor([1.],dtype=torch.float64),'rho':torch.tensor([1000.],dtype=torch.float64),
            'dt':torch.tensor([1/60],dtype=torch.float64)}

    def test_previous_pressure_and_updated_residual_are_used_without_labels(self):
        sample=self.sample()
        for budget in (1,2,4,8):
            actual=correct(TwoCells,None,None,sample,budget)
            self.assertAlmostEqual(float(actual.flatten()[1]),2.-.1**budget,places=14)
            self.assertEqual(float(actual.flatten()[0]),0.)
        self.assertEqual(float(sample['base'].flatten()[1]),1.)

    def test_single_proposal_preserves_original_path_and_invalid_budget_fails(self):
        sample=self.sample()
        self.assertTrue(torch.equal(correct(TwoCells,None,None,sample,1),TwoCells.predict_pressure(None,sample,None)))
        for budget in (0,9,True):
            with self.assertRaisesRegex(ValueError,'one to eight'):correct(TwoCells,None,None,sample,budget)


if __name__=='__main__':unittest.main()
