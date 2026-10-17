"""Pressure declaration/custody preparation only; no optimizer or inference."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import typed_pressure_evidence as evidence
import typed_pressure_workflow as workflow

ROOT = Path(os.environ['RHEON_PRESSURE_MATERIAL_ROOT']).resolve() if os.environ.get('RHEON_PRESSURE_MATERIAL_ROOT') else None
MANIFEST = os.environ.get('RHEON_PRESSURE_TRAINING_MANIFEST')
TULDOK = Path(__file__).resolve().parents[3] / 'Tuldok'


class PressureGlue(unittest.TestCase):
    def test_runtime_input_projection_binds_bytes_witness_and_same_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / 'runtime.json'
            runtime.write_bytes(b'{"actual":"measured current manifest fixture"}')
            identity = hashlib.sha256(runtime.read_bytes()).hexdigest()
            witness = root / 'witness.json'
            witness.write_text(json.dumps({'runtime_manifest_sha256': identity}))
            receipt = {'project_id': 'a'*32, 'manifest_sha256': 'b'*64,
                       'reference_id': 'runtime_manifest', 'path': 'runtime.json', 'measured_sha256': identity}
            receipts = {'runtime_manifest': receipt, 'dataset': dict(receipt, reference_id='dataset')}
            original = copy.deepcopy(receipts)
            self.assertEqual(evidence.trainer_input_receipts(receipts, runtime, witness), {'dataset': receipts['dataset']})
            self.assertEqual(receipts, original)
            for changed in (dict(receipt, measured_sha256='0'*64), dict(receipt, project_id='c'*32),
                            dict(receipt, kind='run_artifact')):
                with self.assertRaises(ValueError):
                    evidence.trainer_input_receipts(dict(receipts, runtime_manifest=changed), runtime, witness)

    def test_restore_alias_keeps_roles_limits_and_original_declarations(self):
        original = {'trained_profile_json': ('profile.json', 'training_diagnostics', 1000),
                    'trained_weights_npz': ('weights.npz', 'checkpoint', 2000)}
        before = copy.deepcopy(original)
        mapped = evidence.restore_artifact_names(original)
        self.assertEqual(original, before)
        self.assertEqual(mapped['trained_profile_json'], ('training-profile.json', 'training_diagnostics', 1000))
        self.assertEqual(mapped['trained_weights_npz'], original['trained_weights_npz'])

    def test_pressure_metric_separates_gauge_and_rejects_undefined_nonfinite(self):
        if importlib.util.find_spec('numpy') is None:
            self.skipTest('Requires existing NumPy for mathematical metric check')
        import numpy as np
        reference = np.asarray([0., 2., 4.], dtype='<f8')
        self.assertEqual(evidence.mean_free_relative_l2(reference + 10, reference), 0.)
        self.assertEqual(evidence.mean_free_relative_l2(reference * 2, reference), 1.)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            evidence.mean_free_relative_l2(reference, np.zeros(3, dtype='<f8'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            evidence.mean_free_relative_l2(np.asarray([float('nan')]*3), reference)


@unittest.skipUnless(ROOT and MANIFEST, 'Requires explicit actual prepared pressure material')
class ActualPressureDeclarations(unittest.TestCase):
    def setUp(self):
        self.bundle = workflow.declarations(ROOT, MANIFEST)

    def test_actual_sources_units_context_and_finite_prospective_intent(self):
        sys.path.insert(0, str(TULDOK))
        import extension_contract
        for value in self.bundle['definitions'].values():
            extension_contract.declaration(workflow.canonical(value), 'contribution')
        ports = {p['id']: p for p in self.bundle['definitions']['task']['ports']}
        self.assertEqual(ports['initial_residual']['units'], 'm3/s2')
        self.assertEqual(ports['full_pressure']['units'], 'Pa')
        self.assertEqual(set(ports), {'initial_residual', 'previous_pressure', 'native_pressure', 'full_pressure'})
        self.assertEqual(self.bundle['preprocessing_policy']['scalar_context']['density']['value'], 1000)
        self.assertEqual(self.bundle['output_policy']['gauge'], 'cell_zero')
        self.assertEqual(self.bundle['planned_optimizer_updates'], 1)
        self.assertEqual(self.bundle['restore_optimizer_updates'], 0)
        self.assertEqual(self.bundle['trial_intent']['admission'], 'closed')
        self.assertIsNone(self.bundle['physical_context']['historical_parent_run'])

    def test_changed_captured_source_and_heldout_material_are_rejected(self):
        original = workflow.digest
        def changed(path):
            return '0'*64 if Path(path).name == 'model.py' else original(path)
        with patch.object(workflow, 'digest', side_effect=changed), self.assertRaisesRegex(ValueError, 'source changed'):
            workflow.declarations(ROOT, MANIFEST)
        original_decode = workflow.decode
        def heldout(raw):
            value = original_decode(raw)
            if isinstance(value, dict) and value.get('schema') == 'rheon-migration-training-material-v1':
                value['heldout_parent_count'] = 1
            return value
        with patch.object(workflow, 'decode', side_effect=heldout), self.assertRaisesRegex(ValueError, 'no heldout'):
            workflow.declarations(ROOT, MANIFEST)

    def test_consumed_probe_has_exact_original_payloads_without_numerical_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'probe.npz'
            program = '''import json,sys
from pathlib import Path
import typed_pressure_workflow as producer
result=producer.consumed_validation_probe(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(result))
'''
            result = json.loads(subprocess.check_output([sys.executable, '-B', '-c', program, str(ROOT),
                str(ROOT/'runs/water-early-pressure-v1/control/project'), str(output)],
                cwd=Path(__file__).parent, text=True))
            self.assertEqual(result['population'], 'validation')
            self.assertEqual(result['original_training_run_id'], 'e20126d732a24013aafa6cc09ebbc0a5')
            self.assertEqual(result['array_sha256']['native_pressure'],
                             'cefe605b34f0d87f27d3cd0a20c28d7aaea64d521b09861a2dd4394e5aff396d')
            self.assertEqual(result['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(result['optimizer_updates_executed'], 0)
            with self.assertRaisesRegex(ValueError, 'fresh'):
                workflow.consumed_validation_probe(ROOT, ROOT/'runs/water-early-pressure-v1/control/project', output)


@unittest.skipUnless(ROOT and MANIFEST and os.environ.get('TULDOK_DATABASE_URL'),
                     'Requires explicit actual pressure material and isolated PostgreSQL')
class InertCurrentOperations(unittest.TestCase):
    setUp = ActualPressureDeclarations.setUp

    def test_current_model_study_drafts_keep_trial_admission_and_execution_unavailable(self):
        sys.path[:0] = [str(TULDOK), str(TULDOK/'tests')]
        import psycopg
        from psycopg import sql
        from app import Dataset
        from auth_helpers import ExplicitOwnerFixture
        from access import AccessError
        from metadata import connection_info
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Dataset(Path(temporary)/'catalog', project_roots={'pressure': ROOT})
            catalog_id = dataset.db.dataset_id
            try:
                owner = ExplicitOwnerFixture.bootstrap(dataset)
                with self.assertRaises(AccessError):
                    workflow.retain_prospective_intent(dataset, self.bundle)
                with (owner.bound(dataset),
                      patch.object(dataset.studies, 'prepare_trial', side_effect=AssertionError('No qualified pressure Trial')),
                      patch.object(dataset.studies, 'admit', side_effect=AssertionError('No execution'))):
                    result = workflow.retain_prospective_intent(dataset, self.bundle)
                    self.assertEqual(result['status'], 'unsupported')
                    self.assertEqual(dataset.models.get(result['model_draft_id'])['task']['id'], workflow.TASK_ID)
                    study = dataset.studies.get(result['study_id'])['study']
                    self.assertFalse(study['open'])
                    self.assertEqual(study['trial_ids'], [])
                    self.assertEqual(dataset.runs.list(), [])
                    for table in ('model_trials', 'access_execution_grants', 'access_execution_consumptions'):
                        self.assertEqual(dataset.db.execute('SELECT count(*) FROM '+table).fetchone()[0], 0)
            finally:
                dataset.close()
                with psycopg.connect(connection_info(), autocommit=True) as connection:
                    connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier('catalog_'+catalog_id)))


@unittest.skipUnless(ROOT and os.environ.get('RHEON_PRESSURE_ASSEMBLY') and os.environ.get('TULDOK_DATABASE_URL'),
                     'Requires exact assembled current pressure project and isolated PostgreSQL')
class ActualBoundTrialPreparation(unittest.TestCase):
    def test_actual_current_model_material_and_single_trial_prepare_without_execution(self):
        sys.path[:0] = [str(TULDOK), str(TULDOK/'tests')]
        import psycopg
        from psycopg import sql
        from app import Dataset
        from auth_helpers import ExplicitOwnerFixture
        from metadata import connection_info
        from torch_runtimes import TorchRuntimes
        import recipe_process
        import torch_runtime
        assembly_path = Path(os.environ['RHEON_PRESSURE_ASSEMBLY']).resolve()
        assembly = workflow.decode(assembly_path.read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Dataset(Path(temporary)/'catalog', project_roots={'pressure': ROOT})
            catalog_id = dataset.db.dataset_id
            try:
                owner = ExplicitOwnerFixture.bootstrap(dataset)
                with owner.bound(dataset):
                    runtime = torch_runtime.load(assembly['runtime']['manifest'])
                    TorchRuntimes(dataset.db, dataset.access, dataset.path).create(runtime, assembly['runtime']['manifest_sha256'])
                    with (patch.object(recipe_process, 'launch', side_effect=AssertionError('Preparation launched execution')),
                          patch.object(dataset.studies, 'admit', side_effect=AssertionError('Preparation admitted execution'))):
                        result = workflow.prepare_current_trial(dataset, ROOT, assembly_path, 'pressure-inert-prepare-001')
                    self.assertEqual(result['status'], 'prepared_without_execution')
                    self.assertIsNone(result['run'])
                    self.assertFalse(result['admission_open'])
                    trial = dataset.studies.trial(result['trial']['id'])
                    self.assertEqual(trial, result['trial'])
                    material = dataset.pipelines.recipe_material(trial['material'], verify=True)
                    self.assertEqual(material['material']['model'], result['model'])
                    self.assertEqual(dataset.studies.prepare_trial(result['study_id'], result['prepare_body']), trial)
                    self.assertEqual(dataset.runs.list(), [])
                    self.assertEqual(dataset.db.execute('SELECT count(*) FROM model_trials').fetchone()[0], 1)
                    for table in ('access_execution_grants', 'access_execution_consumptions', 'model_attempt_reservations'):
                        self.assertEqual(dataset.db.execute('SELECT count(*) FROM '+table).fetchone()[0], 0)
                    receipt = os.environ.get('RHEON_PRESSURE_TRIAL_RECEIPT')
                    if receipt:
                        Path(receipt).write_text(json.dumps(result, indent=2)+'\n')
            finally:
                dataset.close()
                with psycopg.connect(connection_info(), autocommit=True) as connection:
                    connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier('catalog_'+catalog_id)))


if __name__ == '__main__':
    unittest.main()
