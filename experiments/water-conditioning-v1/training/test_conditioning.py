"""Exercise the actual conditioned network and physical output reconstruction."""
import unittest
import torch
import model


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


if __name__=='__main__':unittest.main()
