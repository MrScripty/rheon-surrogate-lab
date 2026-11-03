"""Exercise real GCR algebra, non-SPD proposals, zero residual and breakdown."""
import unittest
import torch
from preconditioned import gcr

class GCRTests(unittest.TestCase):
    def test_nonsymmetric_preconditioner(self):
        a=torch.tensor([[3.,1.],[1.,2.]],dtype=torch.float64)
        b=torch.tensor([1.,-2.],dtype=torch.float64)
        m=torch.tensor([[-1.,2.],[0.,1.]],dtype=torch.float64)
        x,work=gcr(b,lambda p:a@p,lambda r:m@r,lambda r:float(torch.linalg.vector_norm(r))<1e-12)
        self.assertTrue(torch.allclose(a@x,b,atol=1e-12,rtol=0));self.assertEqual(work['iterations'],2)
    def test_zero_input_never_calls_preconditioner(self):
        def fail(_):raise AssertionError('Unexpected proposal')
        x,work=gcr(torch.zeros(3,dtype=torch.float64),lambda p:p,fail,lambda r:not bool(r.any()))
        self.assertEqual(work['iterations'],0);self.assertFalse(bool(x.any()))
    def test_breakdown_has_no_hidden_fallback(self):
        x,work=gcr(torch.ones(3,dtype=torch.float64),lambda p:p,lambda r:torch.zeros_like(r),lambda r:False)
        self.assertIsNotNone(work['breakdown']);self.assertEqual(work['iterations'],0);self.assertFalse(bool(x.any()))
    def test_true_residual_and_restart(self):
        a=torch.diag(torch.arange(1,7,dtype=torch.float64));b=torch.ones(6,dtype=torch.float64);seen=[]
        def accepted(r):seen.append(r.clone());return float(torch.linalg.vector_norm(r))<1e-9
        x,work=gcr(b,lambda p:a@p,lambda r:r,accepted,max_iterations=64,restart=3)
        self.assertLess(float(torch.linalg.vector_norm(b-a@x)),1e-9)
        self.assertEqual(work['operator_applications'],2*work['iterations'])
        self.assertTrue(torch.allclose(seen[-1],b-a@x,atol=1e-12,rtol=0))

if __name__=='__main__':unittest.main()
