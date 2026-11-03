"""Separable transform parity and a real allocation-bounded 40-cubed solve."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'water-residual-learning-v1'))
import audit


class SpectralMemory(unittest.TestCase):
    def test_small_transform_matches_independent_dense_contraction(self):
        audit.TORCH=torch
        torch.set_num_threads(1)
        bases=[]
        for n in (5,7,9):
            k=torch.arange(n,dtype=torch.float64)
            b=torch.cos(torch.pi*k[:,None]*(k[None,:]+.5)/n)*(2/n)**.5
            b[0]/=2**.5
            bases.append(b)
        field=torch.arange(5*7*9,dtype=torch.float64).reshape(5,7,9)/315
        dense=torch.einsum('az,by,cx,zyx->abc',*bases,field)
        actual=audit.cosine_transform(field,bases)
        torch.testing.assert_close(actual,dense,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(audit.cosine_transform(actual,bases,inverse=True),field,rtol=1e-13,atol=1e-13)

    def test_large_real_solve_under_hard_additional_memory_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'result.json'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('probe_memory.py')),
                '--output',str(output)],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            record=json.loads(output.read_text())
            self.assertEqual(record['status'],'passed')
            self.assertFalse(record['cuda_initialized'])
            self.assertLess(record['relative_pressure_error'],1e-11)
            self.assertLess(record['peak_rss_bytes']-record['before_peak_rss_bytes'],128*1024**2)


if __name__=='__main__':unittest.main()
