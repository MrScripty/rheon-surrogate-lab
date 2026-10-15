"""Exercise the actual conditioned network and physical output reconstruction."""
import unittest
import torch
import model
import train


class Conditioning(unittest.TestCase):
    def setUp(self):
        model.configure('cpu'); torch.manual_seed(31)
        self.net=model.PressureCorrectionNet().eval()
        self.norm={'schema':'rheon-cubic-pressure-normalization-v2','channels':model.CHANNELS,
            'scales':{name:{'value':value} for name,value in [('residual',.2),('pressure',3.),('rhs',.3),('rho',1.),('dt',1.)]}}
        shape=(1,1,5,7,9)
        base=torch.randn(shape,dtype=torch.float64);base-=base[:,:,0:1,0:1,0:1].clone()
        self.sample={'base':base,'native_input_residual':torch.randn(shape,dtype=torch.float64),
            'cell_size_m':torch.tensor([.125],dtype=torch.float64),
            'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}

    def select(self,variant):
        model.select({'schema':'rheon-water-conditioning-experiment-v1','conditioning':variant})

    def test_scalar_units_are_explicit_and_fields_unchanged(self):
        self.select('scalar-units')
        s=self.sample
        before=model.owner.build_inputs(s['native_input_residual'],s['base'],s['cell_size_m'],s['rho'],s['dt'],self.norm)
        actual=model.build_inputs(s['native_input_residual'],s['base'],s['cell_size_m'],s['rho'],s['dt'],self.norm)
        self.assertTrue(torch.equal(actual[:,:5],before[:,:5]))
        self.assertTrue(torch.equal(actual[:,5],torch.ones_like(actual[:,5])))
        torch.testing.assert_close(actual[:,6],torch.ones_like(actual[:,6]))

    def test_amplitude_reconstruction_and_zero_bypass(self):
        self.select('amplitude-and-units')
        with torch.no_grad():
            original=model.predict_pressure(self.net,self.sample,self.norm)-self.sample['base']
            for gain in (.25,.5,2.,4.):
                scaled={**self.sample,'native_input_residual':gain*self.sample['native_input_residual']}
                predicted=model.predict_pressure(self.net,scaled,self.norm)-self.sample['base']
                torch.testing.assert_close(predicted,gain*original,rtol=2e-6,atol=1e-10)
                self.assertEqual(float(predicted[0,0,0,0,0]),0.)
            zero={**self.sample,'native_input_residual':torch.zeros_like(self.sample['base'])}
            self.assertTrue(torch.equal(model.predict_pressure(self.net,zero,self.norm),self.sample['base']))
        self.assertFalse(torch.cuda.is_initialized())

    def test_parameter_inventory_stays_compatible_but_profile_changes(self):
        self.select('scalar-units')
        self.assertEqual(model.parameter_count(self.net),371729)
        self.assertNotEqual(model.ARCHITECTURE['schema'],model.owner.ARCHITECTURE['schema'])

    def test_combined_finite_checks_detect_a_bad_tensor(self):
        self.assertTrue(train.finite_tensors([torch.ones(3),torch.zeros(2)]))
        for bad in (float('nan'),float('inf'),-float('inf')):
            self.assertFalse(train.finite_tensors([torch.ones(3),torch.tensor([0.,bad])]))

    def test_stacked_examples_preserve_independent_loss_gradient(self):
        self.select('scalar-units')
        samples=[]
        for gain in (.5,2.):
            sample={key:value.clone() for key,value in self.sample.items()}
            sample['reference']=sample['base']+gain*sample['native_input_residual']
            sample['reference']-=sample['reference'][:,:,0:1,0:1,0:1].clone()
            sample['rhs']=model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho'])
            sample['native_input_residual']=sample['rhs']-model.full_operator(sample['base'],sample['cell_size_m'],sample['rho'])
            samples.append(sample)
        self.net.zero_grad(set_to_none=True)
        for sample in samples:
            loss=model.pressure_loss(model.predict_pressure(self.net,sample,self.norm),sample,self.norm).mean()/len(samples)
            loss.backward()
        independent=[p.grad.clone() for p in self.net.parameters()]
        self.net.zero_grad(set_to_none=True)
        stacked={key:torch.cat([sample[key] for sample in samples],dim=0) for key in samples[0]}
        model.pressure_loss(model.predict_pressure(self.net,stacked,self.norm),stacked,self.norm).mean().backward()
        for expected,parameter in zip(independent,self.net.parameters()):
            # Float32 batch kernels change reduction order; near-zero bias
            # gradients require an absolute floor as well as a norm-relative bound.
            difference=torch.linalg.vector_norm(parameter.grad-expected)
            limit=1e-4*torch.linalg.vector_norm(expected)+1e-6
            self.assertLessEqual(float(difference),float(limit))


if __name__=='__main__':unittest.main()
