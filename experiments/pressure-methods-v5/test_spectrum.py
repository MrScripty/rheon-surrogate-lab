"""Independent modal eigenvalues, SI wavelengths and split identity checks."""
import math,sys,unittest
from pathlib import Path
import torch
from spectrum import pattern
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pressure-methods-v1'))
import numerics
class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def row(self):return dict(seed=1703,counts_xyz=[13,9,7],cells_per_metre=8,mode_band=[1,5],mode_count=6,pressure_rms_pa=1.)
    def test_generated_rhs_matches_closed_form_neumann_eigenvalues(self):
        row=self.row();pressure,details=pattern(row,0);shape=row['counts_xyz'][::-1];value=torch.zeros_like(pressure);expected=torch.zeros_like(pressure)
        for d in details:
            term=torch.ones(shape,dtype=torch.float64);eigenvalue=0.
            for axis,(n,k)in enumerate(zip(shape,d['mode_zyx'])):
                view=[1]*3;view[axis]=n;term*=torch.cos(math.pi/n*k*(torch.arange(n,dtype=torch.float64)+.5)).reshape(view)
                eigenvalue+=4*math.sin(math.pi*k/(2*n))**2
            value+=d['coefficient']*term;expected+=d['coefficient']*eigenvalue*term
        scale=row['pressure_rms_pa']/float(value.square().mean().sqrt());h=torch.tensor([1/row['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
        actual=numerics.fine_action(pressure,h,rho);expected*=scale*float(h[0]/rho[0])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-17,rtol=1e-11));self.assertAlmostEqual(float(actual.sum()),0.,places=15)
    def test_wavelengths_track_extents_and_counts_separately(self):
        row=self.row();_,details=pattern(row,0)
        for d in details:
            for n,k,w in zip(row['counts_xyz'][::-1],d['mode_zyx'],d['wavelength_metres_zyx']):
                if k:self.assertEqual(w,2*n/row['cells_per_metre']/k)
                else:self.assertIsNone(w)
    def test_patterns_are_deterministic_and_distinct(self):
        row=self.row();a,da=pattern(row,0);b,db=pattern(row,0);c,_=pattern(row,1)
        self.assertTrue(torch.equal(a,b));self.assertEqual(da,db);self.assertFalse(torch.equal(a,c));self.assertEqual(float(a[0,0,0,0,0]),0.)
if __name__=='__main__':unittest.main()
