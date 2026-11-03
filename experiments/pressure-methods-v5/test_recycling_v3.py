"""Sequence-level algebra and host-commit tests for actual recycled GCR."""
import unittest
import torch
from recycling_v3 import RecycledGCR
class RecyclingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.a=torch.diag(torch.tensor([1.,2.,3.,5.],dtype=torch.float64));self.action=lambda x:self.a@x
        self.accept=lambda r:float(torch.linalg.vector_norm(r))<1e-10
    def solve(self,solver,b,context='scene',precondition=lambda r:r):return solver.solve(b,self.action,precondition,self.accept,context)
    def test_repeated_equation_reuses_real_directions(self):
        s=RecycledGCR(4);b=torch.tensor([1.,2.,-3.,4.],dtype=torch.float64)
        x,w=self.solve(s,b);self.assertTrue(torch.allclose(self.action(x),b,atol=1e-10,rtol=0));self.assertEqual(w['iterations'],4);s.commit(True)
        x,w=self.solve(s,2*b);self.assertEqual(w['iterations'],0);self.assertEqual(w['cache_in'],4);self.assertTrue(torch.allclose(self.action(x),2*b,atol=1e-10,rtol=0));s.commit(True)
    def test_abrupt_changed_and_unrelated_rhs_use_true_equation(self):
        s=RecycledGCR(2)
        for b in [torch.ones(4,dtype=torch.float64),torch.tensor([7.,-3.,1.,-9.],dtype=torch.float64),-torch.arange(1,5,dtype=torch.float64)]:
            x,w=self.solve(s,b);self.assertTrue(torch.allclose(self.action(x),b,atol=1e-10,rtol=0));s.commit(True)
    def test_geometry_context_resets_reuse(self):
        s=RecycledGCR(4);b=torch.ones(4,dtype=torch.float64);self.solve(s,b,'scene-A');s.commit(True)
        _,w=self.solve(s,b,'scene-B');self.assertEqual(w['cache_in'],0);self.assertTrue(w['context_reset']);s.commit(True)
    def test_unaccepted_host_step_never_commits_new_directions(self):
        s=RecycledGCR(4);b=torch.ones(4,dtype=torch.float64);self.solve(s,b);s.commit(False);self.assertFalse(s.basis)
        _,w=self.solve(s,b);self.assertEqual(w['cache_in'],0);s.commit(False)
    def test_missing_host_decision_is_an_error(self):
        s=RecycledGCR(4);b=torch.ones(4,dtype=torch.float64);self.solve(s,b)
        with self.assertRaises(ValueError):self.solve(s,b)
    def test_zero_equation_bypasses_proposals(self):
        s=RecycledGCR(4)
        def fail(_):raise AssertionError('Zero equation invoked a proposal')
        x,w=self.solve(s,torch.zeros(4,dtype=torch.float64),precondition=fail);self.assertEqual(w['iterations'],0);self.assertEqual(w['operator_applications'],0);s.commit(True)
    def test_breakdown_never_falls_back(self):
        s=RecycledGCR(4);x,w=self.solve(s,torch.ones(4,dtype=torch.float64),precondition=torch.zeros_like)
        self.assertIsNotNone(w['breakdown']);self.assertFalse(bool(x.any()));s.commit(False);self.assertFalse(s.basis)
    def test_zero_capacity_matches_independent_original_gcr(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pressure-methods-v4'))
        from preconditioned import gcr
        b=torch.arange(1,5,dtype=torch.float64);s=RecycledGCR(0);x,w=self.solve(s,b)
        expected,prior=gcr(b,self.action,lambda r:r,self.accept)
        self.assertTrue(torch.equal(x,expected));self.assertEqual(w['iterations'],prior['iterations']);s.commit(True)
    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):self.solve(RecycledGCR(4),torch.full((4,),float('nan'),dtype=torch.float64))
    def test_useful_old_directions_survive_small_new_complement(self):
        old=torch.tensor([100.,1.,0.,0.],dtype=torch.float64)
        changed=torch.tensor([100.,1.,.1,0.],dtype=torch.float64)
        iterations={}
        for strategy in ['tail','contribution']:
            solver=RecycledGCR(2,strategy=strategy)
            for b in [old,changed,old]:
                x,work=self.solve(solver,b);self.assertTrue(torch.allclose(self.action(x),b,atol=1e-10,rtol=0));solver.commit(True)
            iterations[strategy]=work['iterations']
        self.assertEqual(iterations['contribution'],0)
        self.assertGreater(iterations['tail'],iterations['contribution'])
    def test_nonsymmetric_indefinite_neural_proposal_and_reuse(self):
        a=torch.tensor([[3.,1.],[1.,2.]],dtype=torch.float64)
        proposal=torch.tensor([[-1.,2.],[0.,1.]],dtype=torch.float64)
        solver=RecycledGCR(2)
        for b in [torch.tensor([1.,-2.],dtype=torch.float64),torch.tensor([-3.,1.],dtype=torch.float64)]:
            x,w=solver.solve(b,lambda p:a@p,lambda r:proposal@r,self.accept,'non-SPD-proposal')
            self.assertTrue(torch.allclose(a@x,b,atol=1e-10,rtol=0));solver.commit(True)
if __name__=='__main__':unittest.main()
