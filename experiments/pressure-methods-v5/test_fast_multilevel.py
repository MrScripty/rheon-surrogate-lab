"""The optimization retains the actual fine operator and recursive hierarchy."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pressure-methods-v4'))
import torch
import multilevel as prior
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fast_multilevel as fast
class FastHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_edge_stencil_matches_qualified_full_operator(self):
        torch.manual_seed(1701)
        for shape in [(1,1,1),(5,7,9),(16,16,16),(33,25,17),(64,64,64)]:
            x=torch.randn((1,1,*shape),dtype=torch.float64)
            self.assertTrue(torch.allclose(fast.fine_action(x),prior.geometry(shape).action(x),atol=2e-13,rtol=2e-13))
            self.assertFalse(bool(fast.fine_action(torch.ones_like(x)).any()))
    def test_complete_cycle_matches_original_on_rectangles_and_cubes(self):
        torch.manual_seed(1702);h=torch.tensor([.125],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
        for shape in [(5,7,9),(16,16,16),(33,25,17),(64,64,64)]:
            x=torch.randn((1,1,*shape),dtype=torch.float64);x-=x.mean()
            before=prior.correction(x,h,rho);after=fast.correction(x,h,rho)
            self.assertLess(float(torch.linalg.vector_norm(before-after))/float(torch.linalg.vector_norm(before)),1e-12)
            self.assertEqual(float(after[0,0,0,0,0]),0.)
    def test_incompatible_equation_is_not_projected_away(self):
        with self.assertRaises(ValueError):fast.correction(torch.ones((1,1,5,5,5),dtype=torch.float64),torch.tensor([.125],dtype=torch.float64),torch.tensor([1000.],dtype=torch.float64))
if __name__=='__main__':unittest.main()
