"""Multi-grid units, independent eigenvalue labels, bounded sampling and legacy parity."""
import copy
from pathlib import Path
import unittest
import torch
import analytical
import model
import policy
import test_residual


class MultiGrid(unittest.TestCase):
    def setUp(self):
        fixture=test_residual.Residual();fixture.setUp();self.sample=fixture.sample
        self.config=policy.decode(policy.read(policy.LAB/'experiments/water-conditioning-v1/configs/multigrid-smoke.json'))
        self.spec=copy.deepcopy(self.config['analytical_feedback']);self.spec['patterns']=8
        self.spec['grids']=[{'counts_xyz':[9,7,5],'cells_per_metre':8},
            {'counts_xyz':[8,6,4],'cells_per_metre':16}]

    def test_grid_labels_units_gauge_and_balanced_exposure(self):
        before=torch.random.get_rng_state().clone()
        samples,order,evidence=analytical.bank(self.sample,self.spec,31)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(sorted(evidence['grid_exposures']),[15,16])
        self.assertEqual(len(samples),64)
        actual=sum(v.numel()*v.element_size() for s in samples for v in s.values())
        self.assertEqual(actual,evidence['resident_tensor_bytes'])
        expected_amplitude=float((self.sample['reference']-self.sample['base']).square().mean().sqrt())
        for index,sample in enumerate(samples):
            grid=self.spec['grids'][index//32]
            self.assertEqual(list(sample['base'].shape[2:]),grid['counts_xyz'][::-1])
            self.assertEqual(float(sample['cell_size_m'][0]),1/grid['cells_per_metre'])
            self.assertEqual(float(sample['reference'][0,0,0,0,0]),0.)
            self.assertAlmostEqual(float(sample['reference'].square().mean().sqrt()),expected_amplitude*abs(analytical.AMPLITUDES[index%4]))
            torch.testing.assert_close(model.full_operator(sample['reference'],sample['cell_size_m'],sample['rho']),sample['rhs'],rtol=3e-13,atol=3e-17)
        _,again,receipt=analytical.bank(self.sample,self.spec,31)
        self.assertTrue((order==again).all());self.assertEqual(evidence,receipt)

    def test_invalid_grids_spacing_and_cache_reject_at_owner(self):
        for key,value,diagnostic in [('grids',[], 'analytical_grids'),
            ('grids',[{'counts_xyz':[41]*3,'cells_per_metre':16}],'analytical_grid'),
            ('grids',[{'counts_xyz':[8]*3,'cells_per_metre':0}],'analytical_spacing'),
            ('cache_max_bytes',1,'analytical_cache'),('sampling','random','analytical_sampling')]:
            spec={**self.spec,key:value}
            with self.assertRaisesRegex(ValueError,diagnostic):analytical.validate_grids(spec)
        duplicate={**self.spec,'grids':[self.spec['grids'][0]]*2}
        with self.assertRaisesRegex(ValueError,'analytical_grid'):analytical.validate_grids(duplicate)

    def test_schema_is_closed_and_legacy_dispatch_preserves_bank(self):
        policy.experiment(self.config)
        changed=copy.deepcopy(self.config);changed['analytical_feedback']['extra']=0
        with self.assertRaisesRegex(ValueError,'analytical_feedback'):policy.experiment(changed)
        legacy={'seed':42,'weight':.1,'patterns':8}
        original=analytical.single_grid_bank(self.sample,legacy,17)
        routed=analytical.bank(self.sample,legacy,17)
        self.assertEqual(original[2],routed[2]);self.assertTrue((original[1]==routed[1]).all())
        for a,b in zip(original[0],routed[0]):
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))


if __name__=='__main__':unittest.main()
