"""Analytical labels, auxiliary loss and separate output-scale regression checks."""
import copy
import unittest
import torch
import analytical
import model
import policy
import test_residual


class Analytical(unittest.TestCase):
    def setUp(self):
        fixture=test_residual.Residual();fixture.setUp()
        self.sample=fixture.sample;self.norm=fixture.norm;self.config=fixture.config

    def versioned_config(self):
        result=copy.deepcopy(self.config)
        result.update(schema='rheon-water-residual-experiment-v3',objective='canonical',
            output_scale='previous-pressure-rms',analytical_feedback={'seed':42,'weight':.1,'patterns':8})
        result['execution']['prepared_fields']=False
        return result

    def test_eigenvalue_labels_match_sealed_operator_on_rectangular_grid(self):
        components=[((1,2,3),.7),((4,6,8),1.2)]
        sample=analytical.make_sample(self.sample,components,-2.)
        actual=model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])
        torch.testing.assert_close(actual,sample['rhs'],rtol=2e-13,atol=2e-17)
        self.assertEqual(float(sample['reference'][0,0,0,0,0]),0.)
        self.assertAlmostEqual(float(sample['reference'].square().mean().sqrt()),2.)
        self.assertLess(abs(float(sample['rhs'].sum())),1e-16)

    def test_loss_rejects_zero_and_amplifying_corrections(self):
        sample=analytical.make_sample(self.sample,[((4,6,8),1.)],2.)
        reference=sample['reference']
        self.assertAlmostEqual(float(analytical.loss(reference,sample,model.full_operator)),0.,places=20)
        self.assertAlmostEqual(float(analytical.loss(torch.zeros_like(reference),sample,model.full_operator)),1.)
        self.assertAlmostEqual(float(analytical.loss(-reference,sample,model.full_operator)),13.)
        prediction=(reference*.4).requires_grad_(True)
        loss=analytical.loss(prediction,sample,model.full_operator)
        gradient,=torch.autograd.grad(loss,prediction)
        # Along the exact pressure direction, loss=(1-a)^2 for 0<a<1.
        self.assertAlmostEqual(float((gradient*reference).sum()),-1.2,places=10)

    def test_generation_is_reproducible_signed_and_does_not_consume_torch_rng(self):
        spec={'seed':42,'weight':.1,'patterns':10}
        before=torch.random.get_rng_state().clone()
        first,order,evidence=analytical.bank(self.sample,spec,30)
        second,again,receipt=analytical.bank(self.sample,spec,30)
        self.assertTrue(torch.equal(torch.random.get_rng_state(),before))
        self.assertEqual(evidence,receipt);self.assertTrue((order==again).all())
        self.assertEqual(len(first),40)
        for a,b in zip(first,second):self.assertTrue(torch.equal(a['rhs'],b['rhs']))
        torch.testing.assert_close(first[0]['reference'],-first[3]['reference'],rtol=0,atol=0)

    def test_versioned_policy_rejects_unsupported_feedback(self):
        config=self.versioned_config();policy.experiment(config)
        for key,value,diagnostic in [('weight',float('nan'),'analytical_feedback'),('patterns',0,'analytical_feedback'),('seed',-1,'analytical_feedback')]:
            changed=copy.deepcopy(config);changed['analytical_feedback'][key]=value
            with self.assertRaisesRegex(ValueError,diagnostic):policy.experiment(changed)
        config['execution']['prepared_fields']=True
        with self.assertRaisesRegex(ValueError,'residual_scale'):policy.experiment(config)

    def test_output_scale_is_train_only_and_preserves_input_and_loss_units(self):
        parents=[{'entry':{'split':'train'},'density_kg_m3':1000.,'cell_size_m':.125,
            'arrays':{key:self.sample[key].numpy() for key in ('native_input_residual','base','reference')}}]
        parents.append({'entry':{'split':'validation'}})  # No validation fields are available.
        config=self.versioned_config();model.select(config)
        first=model.fit_normalization(parents,lambda _p:copy.deepcopy(self.norm))
        config['output_scale']='correction-rms';model.select(config)
        second=model.fit_normalization(parents,lambda _p:copy.deepcopy(self.norm))
        self.assertEqual(first['scales']['pressure'],second['scales']['pressure'])
        expected=float((self.sample['reference']-self.sample['base']).square().mean().sqrt())
        self.assertAlmostEqual(model.scales(second)['correction_pressure'],expected)
        network=model.PressureCorrectionNet()
        with torch.no_grad():
            a=model.predict_pressure(network,self.sample,first)-self.sample['base']
            b=model.predict_pressure(network,self.sample,second)-self.sample['base']
        torch.testing.assert_close(b,a*(expected/3.),rtol=1e-10,atol=1e-14)
        self.assertTrue(torch.equal(model.pressure_loss(self.sample['base'],self.sample,first),
                                    model.pressure_loss(self.sample['base'],self.sample,second)))


if __name__=='__main__':unittest.main()
