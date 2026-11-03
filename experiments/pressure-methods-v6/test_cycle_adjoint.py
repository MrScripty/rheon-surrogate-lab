"""Independent dense Jacobian, transpose, finite-difference and full-grid checks."""
import unittest,math
from v6_support import HERE
import torch,multilevel
class AdjointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2026101917)
        import adjoint_multilevel
        self.impl=adjoint_multilevel;self.h=torch.tensor([.0625],dtype=torch.float64);self.rho=torch.tensor([1000.],dtype=torch.float64)
    def test_dense_unrestricted_transpose_includes_cell_zero_gauge(self):
        for shape in [(3,3,3),(5,3,3),(7,5,3)]:
            n=math.prod(shape);eye=torch.eye(n,dtype=torch.float64);level=multilevel.geometry(shape)
            matrix=torch.stack([level.cycle(eye[:,j].reshape(1,1,*shape)).flatten()for j in range(n)],dim=1)
            self.assertLess(float(torch.linalg.vector_norm(matrix-matrix.T))/float(torch.linalg.vector_norm(matrix)),1e-12)
            gauge=eye.clone();gauge[:,0]-=1;operator=(self.rho/self.h)[0]*(gauge@matrix)
            r=torch.randn((1,1,*shape),dtype=torch.float64);r-=r.mean();r.requires_grad_(True);g=torch.randn_like(r)
            self.impl.correction(r,self.h,self.rho).backward(g)
            expected=operator.T@g.flatten()
            self.assertLess(float(torch.linalg.vector_norm(r.grad.flatten()-expected))/float(torch.linalg.vector_norm(expected)),1e-11)
    def test_gradients_agree_with_original_autograd_on_real_3d_shapes(self):
        for shape in [(16,16,16),(33,25,17),(64,64,64),(128,128,128)]:
            r=torch.randn((1,1,*shape),dtype=torch.float64);r-=r.mean();a=r.clone().requires_grad_(True);b=r.clone().requires_grad_(True);g=torch.randn_like(r)
            first=multilevel.correction(a,self.h,self.rho);second=self.impl.correction(b,self.h,self.rho);first.backward(g);second.backward(g)
            self.assertLess(float(torch.linalg.vector_norm(first-second))/float(torch.linalg.vector_norm(first)),1e-11)
            self.assertLess(float(torch.linalg.vector_norm(a.grad-b.grad))/float(torch.linalg.vector_norm(a.grad)),1e-11)
    def test_directional_finite_difference(self):
        r=torch.randn((1,1,13,9,7),dtype=torch.float64);r-=r.mean();r.requires_grad_(True);g=torch.randn_like(r);d=torch.randn_like(r);d-=d.mean()
        self.impl.correction(r,self.h,self.rho).backward(g);analytic=float((r.grad*d).sum());eps=1e-5
        with torch.no_grad():finite=float(((self.impl.correction(r+eps*d,self.h,self.rho)-self.impl.correction(r-eps*d,self.h,self.rho))*g).sum())/(2*eps)
        self.assertLess(abs(analytic-finite)/max(abs(analytic),abs(finite),1e-100),1e-8)
    def test_zero_is_finite_and_physical_parameters_are_fixed(self):
        r=torch.zeros((1,1,9,7,5),dtype=torch.float64,requires_grad=True);p=self.impl.correction(r,self.h,self.rho);p.sum().backward()
        self.assertTrue(bool(torch.isfinite(r.grad).all()));self.assertFalse(bool(p.any()))
        with self.assertRaises(ValueError):self.impl.correction(r,self.h.clone().requires_grad_(True),self.rho)
if __name__=='__main__':
    import math
    unittest.main()
