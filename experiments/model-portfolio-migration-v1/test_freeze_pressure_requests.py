"""Physical/case binding tests require no tensor runtime or inference."""
import copy
import hashlib
import struct
import unittest

from freeze_pressure_requests import bind_cases


class PressureCaseBindingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(population='validation', parents=[])
        self.cases = []
        for index, (cpm, dt) in enumerate(((25., 1./30), (100., 1./120))):
            spacing, rho, counts = 1./cpm, 1000., [32,16,64]
            hashes = {name:hashlib.sha256(name.encode()).hexdigest()
                      for name in ('base','native_input_residual','reference','rhs')}
            scalar = {name:hashlib.sha256(struct.pack('<d', value)).hexdigest()
                      for name,value in (('cell_size_m',spacing),('rho',rho),('dt',dt))}
            parent = dict(counts_xyz=counts, cell_size_m=spacing, density_kg_m3=rho,
                          scene=dict(cells_per_metre=cpm,counts_xyz=counts), source='original',
                          capture_manifest_sha256=str(index+1)*64,
                          entry=dict(split='validation',trajectory_id='scene-'+str(index),dt_s=dt),
                          frames=[dict(index=15,sha256=hashes)])
            original = dict(parent=index,native_frame=15,trajectory_id=parent['entry']['trajectory_id'],
                            counts_xyz=counts,cell_size_m=spacing,
                            input_sha256={**{name:hashes[name] for name in ('base','native_input_residual')},**scalar},
                            reference_pressure_sha256=hashes['reference'],reference_rhs_sha256=hashes['rhs'])
            dims = dict(grid_x=32,grid_y=16,grid_z=64,cells_per_metre=cpm,input_dt_s=dt,fluid_density_kg_m3=rho)
            self.manifest['parents'].append(parent)
            self.cases.append(dict(id=f'case-{index:03d}',dimensions=dims,original_case=original))

    def test_same_grid_preserves_distinct_grid_density_and_timestep_without_inferred_timestamp(self):
        rows = bind_cases(self.manifest,self.cases)
        self.assertEqual([r['counts_xyz'] for r in rows],[[32,16,64],[32,16,64]])
        self.assertEqual([r['cells_per_metre'] for r in rows],[25.,100.])
        self.assertEqual([r['input_dt_s'] for r in rows],[1./30,1./120])
        self.assertEqual([r['voxels'] for r in rows],[32768,32768])
        self.assertTrue(all(r['physical_timestamp_s'] is None for r in rows))

    def test_changed_case_timestep_is_rejected(self):
        self.cases[0]['dimensions']['input_dt_s'] = 1./60
        with self.assertRaisesRegex(ValueError,'Native case spacing, density or timestep differs'):
            bind_cases(self.manifest,self.cases)

    def test_captured_scalar_tensor_hash_is_checked_independently_of_numeric_metadata(self):
        self.cases[0]['original_case']['input_sha256']['dt'] = '0'*64
        with self.assertRaisesRegex(ValueError,'Physical scalar differs from its captured input tensor: dt'):
            bind_cases(self.manifest,self.cases)

    def test_changed_reference_field_is_rejected(self):
        self.cases[0]['original_case']['reference_pressure_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'Original field identity differs from its native case: reference'):
            bind_cases(self.manifest,self.cases)

    def test_omitted_original_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Native cases omit original captured frames'):
            bind_cases(self.manifest,self.cases[:1])

    def test_reserved_exposure_cannot_be_relabelled_as_consumed_validation(self):
        self.manifest['parents'][0]['entry']['split'] = 'reserved'
        with self.assertRaisesRegex(ValueError,'Original trajectory/exposure differs'):
            bind_cases(self.manifest,self.cases)


if __name__ == '__main__':
    unittest.main()
