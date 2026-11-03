"""Regressions for sampled diagnostics retaining full-resolution restore evidence."""
import argparse
import io
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def run(project,output):
    if output.exists():raise ValueError('Fresh export regression evidence required')
    witness=read(project/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(project/name)!=expected:raise ValueError('Captured publication source changed')
    sys.path.insert(0,str(project))
    import train,data,numpy as np
    class ExportTests(unittest.TestCase):
        def test_unsampled_voxel_change_is_still_committed_by_full_field_inventory(self):
            array=np.arange(128**3,dtype='<f8').reshape(128,128,128)/17
            original={'case_000':array};changed={'case_000':array.copy()}
            changed['case_000'][1,1,1]=np.nextafter(changed['case_000'][1,1,1],np.inf)
            self.assertTrue(np.array_equal(train.publication_arrays(original)['case_000'],train.publication_arrays(changed)['case_000']))
            self.assertNotEqual(data.inventory(original)['case_000']['sha256'],data.inventory(changed)['case_000']['sha256'])
            self.assertEqual(data.inventory(original)['case_000']['shape'],[128]*3)
        def test_full_largest_publication_population_fits_evidence_cap(self):
            fields={f'case_{i:03d}':np.zeros((128,128,128),dtype='<f8')for i in range(12)}
            packet=io.BytesIO();np.savez(packet,**train.publication_arrays(fields))
            self.assertLessEqual(len(packet.getvalue()),8*1024**2)
            for value in fields.values():self.assertEqual(value.shape,(128,128,128))
        def test_small_and_odd_grid_export_preserves_values_and_gauge(self):
            fields={str(i):np.arange(np.prod(shape),dtype='<f8').reshape(shape)for i,shape in enumerate([(5,7,9),(127,73,51),(64,64,64)])}
            result=train.publication_arrays(fields)
            for key,value in result.items():
                self.assertLessEqual(max(value.shape),32)
                self.assertEqual(value[0,0,0],fields[key][0,0,0])
                self.assertEqual(value.dtype,fields[key].dtype)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExportTests))
    write(output,dict(status='passed'if result.wasSuccessful()else'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),project_source_witness_sha256=sha(project/'source-witness.json'),source_sha256=sha(__file__)))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.project.resolve(),args.output)
