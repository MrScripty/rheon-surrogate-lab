"""Inert assembly against actual retained pressure declarations and current owners.

The environment explicitly supplies the already-consumed validation probe. These
checks publish only disposable static Model metadata; no Trial/Run/optimizer or
numerical-module import is performed.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import typed_pressure_project as producer

ROOT=Path(__file__).resolve().parents[5]
PREPARATION=ROOT/'runs/dependency-migration-v1/receipts/typed-pressure-preparation-01'
TULDOK=ROOT/'runs/dependency-migration-v1/Tuldok'


class InertImportTests(unittest.TestCase):
    def test_cold_import_loads_no_numerical_or_application_modules(self):
        source='import sys;sys.path.insert(0,sys.argv[1]);import typed_pressure_project;assert not any(name in sys.modules for name in ("torch","numpy","app","recipe_contract","projects","psycopg"))'
        subprocess.run([sys.executable,'-I','-S','-c',source,str(Path(__file__).parent)],check=True,capture_output=True,timeout=10)


@unittest.skipUnless((PREPARATION/'declarations.json').is_file(), 'Requires explicit retained pressure preparation fixture')
class PressureAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.bundle=json.loads((PREPARATION/'declarations.json').read_bytes())
        self.probe=json.loads((PREPARATION/'pressure-probe.json').read_bytes())
        self.tmp=tempfile.TemporaryDirectory(prefix='.typed-pressure-contract-',dir=ROOT/'runs/dependency-migration-v1')
        self.addCleanup(self.tmp.cleanup)
        self.destination=Path(self.tmp.name)/'project'

    def test_current_decoders_accept_actual_pressure_material_with_exact_source_mapping(self):
        with patch('subprocess.Popen',side_effect=AssertionError('Assembly must not execute')):
            receipt=producer.assemble(ROOT,self.bundle,self.probe,self.destination,TULDOK)
        project=json.loads((self.destination/'project.json').read_bytes())
        recipe=json.loads((self.destination/'train-recipe.json').read_bytes())
        self.assertEqual(project['schema_version'],2)
        self.assertEqual(recipe['runtime_device'],'cuda')
        self.assertEqual(recipe['schema_version'],3)
        self.assertEqual(recipe['model_evidence']['probe']['candidates'][0]['frozen_reference']['split'],'validation')
        self.assertEqual(recipe['model_evidence']['fields'][0]['contract']['output']['units'],'Pa')
        self.assertEqual(recipe['model_evidence']['fields'][0]['geometry']['shape_xyz'],[16,16,16])
        self.assertTrue(all(Path(item['path']).name!='runtime-manifest.json' for item in recipe['sources']))
        self.assertEqual(next(item for item in project['data'] if item['id']=='runtime_manifest')['sha256'],self.bundle['runtime']['manifest_sha256'])
        self.assertLess(sum((ROOT/item['path']).stat().st_size for item in recipe['sources']),8*1024**2)
        self.assertFalse(receipt['execution_submitted'])
        self.assertEqual(receipt['optimizer_updates_executed'],0)
        for item in receipt['source_mapping']:
            current=item['current']
            self.assertEqual(item['source']['sha256'],producer.digest(ROOT/current['path']))

    def test_stale_source_and_existing_destination_refuse_without_project_publication(self):
        stale=copy.deepcopy(self.bundle)
        stale['recipes']['train']['sources'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):producer.assemble(ROOT,stale,self.probe,self.destination,TULDOK)
        self.assertFalse((self.destination/'project.json').exists())
        marker=self.destination/'preserve'
        marker.write_bytes(b'preserve existing material')
        with self.assertRaises(ValueError):producer.assemble(ROOT,self.bundle,self.probe,self.destination,TULDOK)
        self.assertEqual(marker.read_bytes(),b'preserve existing material')

    @unittest.skipUnless(os.environ.get('TULDOK_DATABASE_URL'),'Requires isolated PostgreSQL for actual static publication')
    def test_actual_pressure_model_publishes_through_current_project_owners_without_execution(self):
        receipt=producer.assemble(ROOT,self.bundle,self.probe,self.destination,TULDOK)
        sys.path.insert(0,str(TULDOK))
        from app import Dataset
        from auth_helpers import ExplicitOwnerFixture
        dataset=Dataset(Path(self.tmp.name)/'catalog',project_roots={'pressure':ROOT})
        self.addCleanup(dataset.close)
        actor=ExplicitOwnerFixture.bootstrap(dataset)
        with actor.bound(dataset),patch('subprocess.Popen',side_effect=AssertionError('Static pressure publication must not execute')):
            project=dataset.projects.register_snapshot({'root':'pressure','manifest_path':receipt['current_project']['path']})
            selected=next(row for row in project['recipes'] if row['id']=='train')
            inspected=dataset.pipelines.inspect_external_recipe(project['id'],'train',selected['sha256'])
            draft=dataset.models.create({'name':'Actual prospective pressure static publication',
                'kind':'captured_source','task':inspected['model_task'],
                'source_selector':{'project_id':project['id'],'recipe_id':'train','expected_sha256':selected['sha256']},
                'specification':receipt['model_specification'],'preprocessing_policy':receipt['preprocessing_policy'],
                'output_policy':receipt['output_policy']})
            published=dataset.models.publish(draft['id'],{'revision':1})
            self.assertEqual(dataset.models.material(published['ref'],verify=True),published)
            self.assertEqual(published['material']['task'],inspected['model_task'])
            self.assertEqual(dataset.runs.list(),[])
            self.assertEqual(dataset.db.execute('SELECT count(*) FROM model_trials').fetchone()[0],0)


if __name__=='__main__':unittest.main()
