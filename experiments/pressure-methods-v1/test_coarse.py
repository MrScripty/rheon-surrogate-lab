"""Independent fine-stencil checks for coarse transfers and exact anchors."""
import unittest
import torch
import coarse
import numerics


class CoarseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(2026101808)
        self.h = torch.tensor([.125],dtype=torch.float64)
        self.rho = torch.tensor([1000.],dtype=torch.float64)

    def request(self,shape):
        pressure = torch.randn((1,1,*shape),dtype=torch.float64,generator=self.generator)
        pressure = pressure-pressure[:,:,:1,:1,:1].clone()
        return pressure,numerics.fine_action(pressure,self.h,self.rho)

    def test_constant_preservation_on_odd_thin_axes(self):
        for shape in [(1,7,9),(5,7,9),(1,1,1),(16,32,64)]:
            axes,_ = coarse.coarse_geometry(shape,'cpu')
            ones = torch.ones((1,*[a.shape[0] for a in axes]),dtype=torch.float64)
            self.assertTrue(torch.allclose(coarse.prolong(ones,axes),torch.ones((1,1,*shape),dtype=torch.float64),atol=1e-14,rtol=0))

    def test_galerkin_equals_independent_fine_stencil(self):
        for shape in [(1,7,9),(5,7,9),(8,8,8)]:
            axes,factor = coarse.coarse_geometry(shape,'cpu')
            size = factor.shape[0]+1
            fields = coarse.prolong(torch.eye(size,dtype=torch.float64).reshape(size,*[a.shape[0] for a in axes]),axes)
            observed = coarse.restrict(numerics.fine_action(fields,torch.ones(size,dtype=torch.float64),torch.ones(size,dtype=torch.float64)),axes).flatten(1).T
            self.assertTrue(torch.allclose(observed[1:,1:],factor@factor.T,rtol=2e-13,atol=1e-13))

    def test_coarse_correction_orthogonality(self):
        for shape in [(1,7,9),(5,7,9),(32,32,32)]:
            _,rhs = self.request(shape)
            correction = coarse.numerical(rhs,self.h,self.rho)
            axes,_ = coarse.coarse_geometry(shape,'cpu')
            restricted = coarse.restrict(rhs-numerics.fine_action(correction,self.h,self.rho),axes)
            self.assertLess(float(restricted.abs().max()),2e-14)
            self.assertEqual(float(correction.flatten()[0]),0.)

    def test_anchors_equal_full_numerical_solution(self):
        for shape in [(1,7,9),(5,7,9),(16,16,16),(1,1,1)]:
            _,rhs = self.request(shape)
            approximate = coarse.sparse(rhs,self.h,self.rho)
            exact = numerics.correction(rhs,self.h,self.rho,None)
            *_,indices = coarse.anchor_geometry(shape,'cpu')
            z,y,x = indices
            pick = lambda p:p[:,:,z[:,None,None],y[None,:,None],x[None,None,:]]
            self.assertTrue(torch.allclose(pick(approximate),pick(exact),rtol=2e-13,atol=2e-13))

    def test_pinned_infill_and_autograd(self):
        delta = torch.randn((1,1,5,7,9),dtype=torch.float64,generator=self.generator,requires_grad=True)
        result = coarse.pin_neural_correction(delta)
        *_,indices = coarse.anchor_geometry((5,7,9),'cpu')
        z,y,x = indices
        self.assertEqual(float(result[:,:,z[:,None,None],y[None,:,None],x[None,None,:]].detach().abs().max()),0.)
        result.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(delta.grad).all()))
        self.assertGreater(float(delta.grad.abs().sum()),0.)

    def test_incompatible_rhs_rejected(self):
        rhs = torch.ones((1,1,5,7,9),dtype=torch.float64)
        for method in [coarse.numerical,coarse.sparse]:
            with self.assertRaises(ValueError):
                method(rhs,self.h,self.rho)


if __name__ == '__main__':
    unittest.main()
