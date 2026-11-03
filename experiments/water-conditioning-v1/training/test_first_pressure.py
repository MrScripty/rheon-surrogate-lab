"""First-correction pressure supervision without inference or legacy changes."""
import copy
import unittest
from unittest.mock import patch
import torch
import model
import physical_residual
import policy
import test_residual


class FirstPressure(unittest.TestCase):
    def setUp(self):
        fixture=test_residual.Residual();fixture.setUp()
        self.sample,self.norm=fixture.sample,fixture.norm
        self.old=policy.decode(policy.read(policy.LAB/'experiments/water-late-residual-v1/configs/extra-late.json'))

    def config(self, weight):
        return {**self.old,'schema':'rheon-water-residual-experiment-v9','first_pressure_weight':weight}

    def test_closed_weight_and_legacy_parity(self):
        for value in (-1,2,True,float('nan'),None):
            with self.assertRaisesRegex(ValueError,'first_pressure_weight'):
                policy.experiment(self.config(value))
        with self.assertRaisesRegex(ValueError,'residual_configuration'):
            policy.experiment({**self.config(1),'unknown':1})
        model.select(self.old);torch.manual_seed(19)
        network=model.PressureCorrectionNet()
        before=model.training_loss(network,self.sample,self.norm)
        before.sum().backward()
        gradients=[p.grad.clone() for p in network.parameters()]
        prediction=model.predict_pressure(network,self.sample,self.norm)
        network.zero_grad(set_to_none=True)
        policy.experiment(self.config(0));model.select(self.config(0))
        after=model.training_loss(network,self.sample,self.norm)
        after.sum().backward()
        self.assertTrue(torch.equal(before,after))
        for expected,p in zip(gradients,network.parameters()):self.assertTrue(torch.equal(expected,p.grad))
        model.select(self.config(1))
        self.assertTrue(torch.equal(prediction,model.predict_pressure(network,self.sample,self.norm)))

    def test_full_first_loss_and_gradient_use_existing_two_predictions(self):
        model.select(self.config(1))
        a=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        b=torch.tensor(3.,dtype=torch.float64,requires_grad=True)
        def predict(_network,_sample,_norm,_owner,_correction,intermediates):
            intermediates.extend([a,b]);return b
        with patch.object(physical_residual,'predict_pressure',side_effect=predict) as forward, \
             patch.object(physical_residual,'intermediate_contraction',return_value=torch.tensor(5.)):
            value=model.training_loss(None,self.sample,self.norm,criterion=lambda p:p.square())
        self.assertEqual(float(value.detach()),9+4+.1*5)
        value.backward()
        self.assertEqual(float(a.grad),4.);self.assertEqual(float(b.grad),6.)
        self.assertEqual(forward.call_count,1)


if __name__=='__main__':unittest.main()
