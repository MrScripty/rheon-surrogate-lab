"""Supplemental examples preserve the primary stream and shared memory budget."""
import copy
import unittest
import numpy as np
import torch
import analytical
import policy
import test_multigrid


class Supplemental(unittest.TestCase):
    def setUp(self):
        fixture=test_multigrid.MultiGrid();fixture.setUp()
        self.sample=fixture.sample
        self.spec={**fixture.spec,'amplitude_reference':'train-correction-cell-rms','grid_capacity':16}
        self.primary=analytical.bank(self.sample,self.spec,31,amplitude_pa=2.)
        self.config=policy.decode(policy.read(policy.LAB/'experiments/water-coverage-feedback-v1/configs/feedback-only.json'))

    def test_private_balanced_supplement_preserves_primary_and_rng(self):
        before=self.primary[1].copy();torch_state=torch.random.get_rng_state().clone()
        np.random.seed(932);expected=np.random.random(3);np.random.seed(932)
        spec={'source':'primary','weight':.1,'seed':739}
        values,order,evidence=analytical.supplemental_bank(self.sample,spec,31,self.primary,self.spec,2.)
        self.assertIs(values,self.primary[0]);self.assertTrue(np.array_equal(before,self.primary[1]))
        self.assertEqual(sorted(evidence['grid_exposures']),[15,16])
        self.assertEqual(evidence['resident_tensor_bytes'],0)
        self.assertEqual(evidence['combined_resident_tensor_bytes'],self.primary[2]['resident_tensor_bytes'])
        self.assertTrue(torch.equal(torch_state,torch.random.get_rng_state()))
        self.assertTrue(np.array_equal(expected,np.random.random(3)))
        self.assertFalse(np.array_equal(order,before))
        again=analytical.bank(self.sample,self.spec,31,amplitude_pa=2.)
        self.assertEqual(again[2],self.primary[2]);self.assertTrue(np.array_equal(again[1],before))
        for a,b in zip(again[0],self.primary[0]):
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))

    def test_target_labels_and_combined_budget(self):
        spec={**self.spec,'source':'grids','seed':17,'grids':[{'counts_xyz':[7,9,11],'cells_per_metre':8}]}
        values,_,evidence=analytical.supplemental_bank(self.sample,spec,31,self.primary,self.spec,2.)
        from model import full_operator
        for sample in values:
            torch.testing.assert_close(full_operator(sample['reference'],sample['cell_size_m'],sample['rho']),sample['rhs'],rtol=3e-13,atol=3e-17)
        limit=self.primary[2]['resident_tensor_bytes']+evidence['resident_tensor_bytes']-1
        constrained={**self.spec,'cache_max_bytes':limit}
        with self.assertRaisesRegex(ValueError,'analytical_cache'):
            analytical.supplemental_bank(self.sample,{**spec,'cache_max_bytes':limit},31,self.primary,constrained,2.)

    def test_weighted_supplemental_gradient_matches_independent_losses(self):
        import train
        import model
        import test_residual
        fixture=test_residual.Residual();fixture.setUp()
        model.select({**self.config,'schema':'rheon-water-residual-experiment-v7',
            'supplemental_feedback':{'source':'primary','weight':.1,'seed':42}})
        first=copy.deepcopy(fixture.net);second=copy.deepcopy(fixture.net)
        primary=self.primary[0][0];supplement=self.primary[0][5]
        for sample,weight,manufactured in ((fixture.sample,1.,False),(primary,.1,True),(supplement,.1,True)):
            criterion=(lambda prediction,s=sample:analytical.loss(prediction,s,model.full_operator)) if manufactured else None
            (weight*model.training_loss(first,sample,fixture.norm,criterion).mean()).backward()
        expected=torch.nn.utils.clip_grad_norm_(first.parameters(),1.,error_if_nonfinite=True)
        optimizer=torch.optim.SGD(second.parameters(),lr=0.)
        _,actual=train.one_update(second,optimizer,[],[(0,0)],fixture.norm,cache={(0,0):fixture.sample},
            analytical_sample=primary,analytical_weight=.1,supplemental_sample=supplement,supplemental_weight=.1)
        self.assertEqual(float(expected),actual)
        for a,b in zip(first.parameters(),second.parameters()):
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_versioned_closed_contract(self):
        extra={'source':'primary','weight':.1,'seed':42}
        old={**self.config,'supplemental_feedback':extra}
        with self.assertRaisesRegex(ValueError,'residual_configuration'):policy.experiment(old)
        config={**old,'schema':'rheon-water-residual-experiment-v7'}
        policy.experiment(config)
        for change in ({'source':'invalid'},{'seed':True},{'weight':float('nan')},{'unknown':1}):
            changed=copy.deepcopy(config);changed['supplemental_feedback'].update(change)
            with self.assertRaises(ValueError):policy.experiment(changed)
        with self.assertRaises(ValueError):policy.experiment({k:v for k,v in config.items() if k!='supplemental_feedback'})


if __name__=='__main__':unittest.main()
