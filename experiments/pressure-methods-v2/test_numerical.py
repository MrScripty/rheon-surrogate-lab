import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/pressure-methods-v1'))
import torch
import coarse
import numerical
from numerics import fine_action

class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_coarse_matrix_matches_fine_action_on_odd_rectangles_and_thin_grids(self):
        one=torch.ones(1,dtype=torch.float64)
        for shape in [(11,7,5),(1,9,5)]:
            for nodes in (2,3,4,6):
                axes,_=numerical.geometry(shape,nodes,'cpu')
                generator=torch.Generator().manual_seed(2026101821)
                coefficients=torch.randn((1,*(a.shape[0] for a in axes)),generator=generator,dtype=torch.float64)
                p=coarse.prolong(coefficients,axes);rhs=fine_action(p,one,one)
                solved=numerical.correction(rhs,one,one,nodes)
                self.assertLess(float(torch.linalg.vector_norm(fine_action(solved,one,one)-rhs))/float(torch.linalg.vector_norm(rhs)),1e-11)
    def test_two_level_cg_recovers_independent_random_pressure(self):
        one=torch.ones(1,dtype=torch.float64)
        generator=torch.Generator().manual_seed(2026101822)
        for shape in [(13,9,7),(32,21,15)]:
            pressure=torch.randn((1,1,*shape),generator=generator,dtype=torch.float64)
            rhs=fine_action(pressure,one,one)
            sample=dict(base=torch.zeros_like(rhs),rhs=rhs,native_input_residual=rhs,cell_size_m=one,rho=one)
            for nodes in (2,3,4,6):
                solved,_=numerical.pcg(sample,nodes=nodes,max_iterations=256,tolerance=1e-8)
                self.assertLess(float(torch.linalg.vector_norm(fine_action(solved,one,one)-rhs))/float(torch.linalg.vector_norm(rhs)),1e-8)
    def test_incompatible_rhs_rejected(self):
        with self.assertRaises(ValueError):numerical.validate(torch.ones((1,1,8,7,5),dtype=torch.float64))

if __name__=='__main__':unittest.main()
