"""Physical residual contract and linear correction properties on rectangular grids."""
import copy
import unittest
import torch
import model
import policy
from prepared import PreparedSample


class Residual(unittest.TestCase):
    def setUp(self):
        model.configure('cpu');torch.manual_seed(47)
        self.config=policy.decode(policy.read(policy.LAB/'experiments/water-conditioning-v1/configs/residual-linear.json'))
        model.select(self.config);self.net=model.PressureCorrectionNet().eval()
        shape=(1,1,5,7,9)
        base=torch.randn(shape,dtype=torch.float64);base-=base[:,:,:1,:1,:1].clone()
        ref=torch.randn(shape,dtype=torch.float64);ref-=ref[:,:,:1,:1,:1].clone()
        self.sample={'base':base,'reference':ref,'cell_size_m':torch.tensor([.125],dtype=torch.float64),
            'rho':torch.tensor([1000.],dtype=torch.float64),'dt':torch.tensor([1/60],dtype=torch.float64)}
        rhs=model.full_operator(ref,self.sample['cell_size_m'],self.sample['rho'])
        self.sample.update(rhs=rhs,native_input_residual=rhs-model.full_operator(base,self.sample['cell_size_m'],self.sample['rho']))
        self.norm={'schema':'rheon-physical-residual-normalization-v1','channels':['dimensionless_laplacian_rhs'],
            'scales':{name:{'value':value} for name,value in [('residual',.2),('pressure',3.),('rhs',.3),('rho',1.),('dt',1.),('laplacian_rhs',2.)]}}

    def test_integrated_operator_scaling_matches_independent_neighbor_stencil(self):
        sample=self.sample;field=sample['reference'];expected=torch.zeros_like(field)
        # Independently enumerate graph neighbors, including sealed wall rows.
        for z in range(5):
            for y in range(7):
                for x in range(9):
                    for dz,dy,dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                        a,b,c=z+dz,y+dy,x+dx
                        if 0<=a<5 and 0<=b<7 and 0<=c<9:
                            expected[0,0,z,y,x]+=field[0,0,z,y,x]-field[0,0,a,b,c]
        actual=model.full_operator(field,sample['cell_size_m'],sample['rho'])*8000.
        torch.testing.assert_close(actual,expected,rtol=2e-15,atol=1e-14)

    def test_only_residual_channel_and_physical_scaling(self):
        s=self.sample
        actual=model.build_inputs(s['native_input_residual'],s['base'],s['cell_size_m'],s['rho'],s['dt'],self.norm)
        expected=(s['native_input_residual']*8000./2.).float()
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(actual.shape,(1,1,5,7,9))
        changed={**s,'dt':s['dt']*2,'base':s['base']*3}
        second=model.build_inputs(changed['native_input_residual'],changed['base'],changed['cell_size_m'],changed['rho'],changed['dt'],self.norm)
        self.assertTrue(torch.equal(actual,second))

    def test_linear_correction_scaling_sign_superposition_and_zero(self):
        s=self.sample;zero=torch.zeros_like(s['base'])
        def correction(residual):
            sample={**s,'base':zero,'native_input_residual':residual}
            return model.predict_pressure(self.net,sample,self.norm)
        a=s['native_input_residual'];b=torch.flip(a,[2,4])*.7
        with torch.no_grad():
            first=correction(a)
            for gain in (-2.,-1.,.5,2.):
                torch.testing.assert_close(correction(a*gain),first*gain,rtol=3e-5,atol=1e-7)
            # Near-zero cells cancel; compare the full correction norm, not a
            # relative per-cell error. This tests f32 linearity, not qualification.
            combined=correction(a+b);expected=first+correction(b)
            error=torch.linalg.vector_norm(combined-expected)
            self.assertLessEqual(float(error),3e-5*float(torch.linalg.vector_norm(expected)))
            self.assertTrue(torch.equal(correction(zero),zero))
        self.assertFalse(any(isinstance(layer,(torch.nn.GroupNorm,torch.nn.SiLU)) for layer in self.net.modules()))
        self.assertTrue(all(layer.bias is None for layer in self.net.modules() if isinstance(layer,torch.nn.Conv3d)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_preparation_keeps_exact_prediction_loss_and_gradients(self):
        for activation in ('silu','linear'):
            config=copy.deepcopy(self.config);config['correction']['activation']=activation
            model.select(config);torch.manual_seed(47);network=model.PressureCorrectionNet()
            prepared=PreparedSample(self.sample,self.norm)
            predicted=model.predict_pressure(network,self.sample,self.norm)
            self.assertTrue(torch.equal(predicted,prepared.prediction(network)))
            expected=model.pressure_loss(predicted,self.sample,self.norm).mean()
            expected.backward();gradients=[p.grad.clone() for p in network.parameters()]
            network.zero_grad(set_to_none=True);actual=prepared.loss(network).mean()
            self.assertTrue(torch.equal(actual,expected));actual.backward()
            for parameter,gradient in zip(network.parameters(),gradients):self.assertTrue(torch.equal(parameter.grad,gradient))

    def test_repeated_updates_recompute_residual_without_labels(self):
        config=copy.deepcopy(self.config);config['correction']['steps']=4
        config['execution']['prepared_fields']=False;policy.experiment(config);model.select(config)
        network=model.PressureCorrectionNet().eval();s={k:v for k,v in self.sample.items() if k not in ('rhs','reference')}
        with torch.no_grad():
            actual=model.predict_pressure(network,s,self.norm)
            config['correction']['steps']=1;model.select(config)
            expected=s['base'].clone()
            for _ in range(4):
                residual=s['native_input_residual']-model.full_operator(expected-s['base'],s['cell_size_m'],s['rho'])
                expected=model.predict_pressure(network,{**s,'base':expected,'native_input_residual':residual},self.norm)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
