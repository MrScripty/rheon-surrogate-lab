"""Independent variational, compatibility and stencil checks for the new hierarchy."""
import unittest
import torch
import multilevel as mg

class GalerkinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_transpose_and_constant_preservation(self):
        generator=torch.Generator().manual_seed(773)
        for n in [1,2,5,8,13,32]:
            p=mg.interpolate(n,max(1,(n+1)//2));t=mg.Transfer(p)
            x=torch.randn((1,1,n,3,2),generator=generator,dtype=torch.float64)
            y=torch.randn((1,1,p.shape[1],3,2),generator=generator,dtype=torch.float64)
            self.assertTrue(torch.allclose(t.prolong(torch.ones_like(y),2),torch.ones_like(x),atol=1e-15,rtol=0))
            self.assertAlmostEqual(float((x*t.prolong(y,2)).sum()),float((t.restrict(x,2)*y).sum()),places=12)
    def test_fine_flux_stencil(self):
        for shape in [(1,1,1),(5,7,9),(8,8,8),(13,9,7)]:
            x=torch.randn((1,1,*shape),dtype=torch.float64);expected=torch.zeros_like(x)
            for axis in [2,3,4]:
                left=[slice(None)]*5;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
                d=x[tuple(left)]-x[tuple(right)];expected[tuple(left)]+=d;expected[tuple(right)]-=d
            self.assertTrue(torch.allclose(mg.geometry(shape).action(x),expected,atol=1e-13,rtol=1e-13))
    def test_recursive_galerkin(self):
        level=mg.geometry((17,13,9))
        while level.child is not None:
            x=torch.randn((1,1,*level.child.shape),dtype=torch.float64)
            expected=level.restrict(level.action(level.prolong(x)))
            actual=level.child.action(x)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-10,rtol=1e-11))
            self.assertLess(float(level.action(torch.ones((1,1,*level.shape),dtype=torch.float64)).abs().max()),1e-10)
            level=level.child
    def test_positive_symmetric_correction(self):
        for shape in [(5,7,9),(16,16,16),(17,13,9)]:
            level=mg.geometry(shape)
            x=torch.randn((1,1,*shape),dtype=torch.float64);x-=x.mean()
            y=torch.randn_like(x);y-=y.mean()
            mx=level.cycle(x);my=level.cycle(y)
            self.assertGreater(float((x*mx).sum()),0.)
            self.assertAlmostEqual(float((y*mx).sum()),float((x*my).sum()),places=9)
            self.assertLess(float(torch.linalg.vector_norm(x-level.action(mx))),float(torch.linalg.vector_norm(x)))
    def test_incompatible_equation_rejected(self):
        with self.assertRaises(ValueError):mg.validate(torch.ones((1,1,5,5,5),dtype=torch.float64))

if __name__=='__main__':unittest.main()
