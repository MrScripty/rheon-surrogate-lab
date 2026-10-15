"""Residual feedback and budget on a two-cell sealed pressure problem."""
import unittest
import torch
from learned_iteration import correct,equation_quality,stopping_policy


class TwoCells:
    CORRECTION={'steps':4}
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
    policy={'schema':'rheon-learned-stopping-policy-v1','relative_residual':1e-6,
        'absolute_residual':1e-12,'predicted_divergence':1e-7}
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

    def test_adaptive_stopping_uses_original_rhs_and_predicted_divergence(self):
        sample=self.sample();trace=[]
        pressure=correct(TwoCells,None,None,sample,8,self.policy,trace)
        self.assertEqual(trace[-1]['proposals'],6)
        self.assertTrue(trace[-1]['equation_passed'])
        # Original RHS [-.002,.002], not the correction RHS [-.001,.001].
        self.assertAlmostEqual(trace[-1]['rhs_l2'],.002*2**.5)
        self.assertAlmostEqual(trace[-1]['relative_residual'],.5e-6,places=13)
        # A tighter divergence requirement must force another correction.
        strict={**self.policy,'predicted_divergence':1e-12};other=[]
        correct(TwoCells,None,None,sample,8,strict,other)
        self.assertEqual(other[-1]['proposals'],8)
        self.assertTrue(other[-1]['equation_passed'])
        self.assertEqual(float(pressure.flatten()[0]),0.)

    def test_zero_work_and_exhaustion_are_explicit(self):
        sample=self.sample();sample['native_input_residual'].zero_();trace=[]
        self.assertTrue(torch.equal(correct(TwoCells,None,None,sample,8,self.policy,trace),sample['base']))
        self.assertEqual(trace[-1]['proposals'],0)
        sample=self.sample();trace=[]
        correct(TwoCells,None,None,sample,1,self.policy,trace)
        self.assertFalse(trace[-1]['equation_passed'])
        self.assertEqual(trace[-1]['proposals'],1)

    def test_invalid_policy_and_nonfinite_quality_fail(self):
        for value in ({**self.policy,'relative_residual':1e-5},{**self.policy,'extra':0},
                      {**self.policy,'absolute_residual':True},{**self.policy,'predicted_divergence':float('nan')}):
            with self.assertRaises(ValueError):stopping_policy(value)
        sample=self.sample();rhs=torch.full_like(sample['base'],float('nan'))
        self.assertFalse(equation_quality(TwoCells,sample['base'],sample,rhs,self.policy)['equation_passed'])

    def test_budget_counts_network_evaluations_for_single_pass_models(self):
        class SinglePass(TwoCells):CORRECTION={'steps':1}
        sample=self.sample();trace=[]
        correct(SinglePass,None,None,sample,32,trace=trace)
        self.assertEqual(len(trace),32)
        for model,budget in ((TwoCells,9),(SinglePass,33)):
            with self.assertRaisesRegex(ValueError,'at most 32'):correct(model,None,None,sample,budget)


class ActualRecurrence(unittest.TestCase):
    def test_four_uninterrupted_evaluations_preserve_pressure_and_count(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'water-conditioning-v1/training'))
        import model
        import test_residual
        fixture=test_residual.Residual();fixture.setUp()
        config=fixture.config;config['correction']['steps']=4;model.select(config)
        network=model.PressureCorrectionNet().eval();calls=[]
        hook=network.register_forward_hook(lambda *_:calls.append(1))
        try:
            with torch.no_grad():
                original=correct(model,network,fixture.norm,fixture.sample,1)
                self.assertEqual(len(calls),4);calls.clear()
                split=correct(model,network,fixture.norm,fixture.sample,4,single_evaluation=True)
                self.assertEqual(len(calls),4)
            self.assertTrue(torch.equal(original,split))
            self.assertEqual(model.CORRECTION['steps'],4)
            with self.assertRaisesRegex(ValueError,'at most 32'):
                correct(model,network,fixture.norm,fixture.sample,33,single_evaluation=True)
        finally:hook.remove()


if __name__=='__main__':unittest.main()
