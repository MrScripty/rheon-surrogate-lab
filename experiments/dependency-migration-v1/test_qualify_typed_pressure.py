"""Qualification-client gates, uncertain commits, and actual custody mapping.

No optimizer, numerical import, worker or live-catalog mutation is used here.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qualify_typed_pressure as client


ROOT = Path(__file__).resolve().parents[5]
ASSEMBLY = ROOT / 'runs/dependency-migration-v1/typed-pressure-project-02/assembly.json'
TULDOK = ROOT / 'runs/dependency-migration-v1/Tuldok'


class ClientSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_import_is_inert_and_requires_explicit_execution_mode(self):
        script = ('import sys;sys.path.insert(0,sys.argv[1]);import qualify_typed_pressure;'
                  'assert not any(n in sys.modules for n in ("torch","numpy","app","psycopg","train"))')
        subprocess.run([sys.executable, '-I', '-S', '-B', '-c', script, str(Path(client.__file__).parent)],
                       check=True, capture_output=True, timeout=10)

    def test_uncertain_commit_is_not_replayed_after_restart(self):
        class Unknown(Exception):
            code, status = 'commit_unknown', 503
        called = []
        def unknown():
            called.append('dispatch')
            raise Unknown('commit may have succeeded')
        with self.assertRaises(Unknown):
            client.Journal(self.path).mutation('admit', 'studies.admit', {'request_key': 'stable-key'}, unknown)
        record = json.loads(next(self.path.glob('*.json')).read_bytes())
        self.assertEqual(record['status'], 'unresolved')
        self.assertEqual(record['code'], 'commit_unknown')
        with self.assertRaises(ValueError):
            client.Journal(self.path).mutation('admit', 'studies.admit', {'request_key': 'stable-key'}, unknown)
        self.assertEqual(called, ['dispatch'])

    def test_credential_token_never_enters_mutation_receipt(self):
        client.Journal(self.path).mutation('credential', 'access.issue', {'scopes': ['inspect']},
            lambda: {'token': 'secret-marker', 'identity': {'actor_id': 'public'}}, redact_result=True)
        self.assertNotIn('secret-marker', next(self.path.glob('*.json')).read_text())

    def test_new_output_directory_cannot_reuse_single_smoke_gate(self):
        gate = self.path / 'gate.json'
        client.claim_smoke(gate, 'a'*64, self.path / 'first')
        with self.assertRaises(FileExistsError):
            client.claim_smoke(gate, 'a'*64, self.path / 'second')

    def gate(self):
        return {'schema': 'rheon-tuldok-migration-cutover-v1', 'm3_service_switched': True,
            'm3_complete': False, 'ordinary_admissions_open': False,
            'smoke_authorized': {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1},
            'catalog_id': 'a'*32, 'endpoint': 'http://127.0.0.1:8765'}

    def test_gate_requires_specific_closed_smoke_authority_and_exact_endpoint(self):
        path = self.path / 'gate.json'
        gate = self.gate()
        client.write(path, gate)
        client.check_gate(path, client.digest(path), gate['catalog_id'], gate['endpoint'])
        for key, changed in [('m3_service_switched', False), ('ordinary_admissions_open', True),
                             ('m3_complete', True), ('catalog_id', 'b'*32),
                             ('endpoint', 'http://127.0.0.1:8766'),
                             ('smoke_authorized', {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 2})]:
            altered = {**gate, key: changed}
            client.write(path, altered)
            with self.subTest(key=key), self.assertRaises(ValueError):
                client.check_gate(path, client.digest(path), gate['catalog_id'], gate['endpoint'])
        client.write(path, gate)
        with self.assertRaises(ValueError):
            client.check_gate(path, '0'*64, gate['catalog_id'], gate['endpoint'])

    def test_ordinary_restore_binds_actual_run_artifacts_with_only_profile_alias(self):
        run = {'id': 'a'*32, 'artifacts': [
            {'name': 'weights.npz', 'sha256': '1'*64},
            {'name': 'checkpoint-001.npz', 'sha256': '2'*64},
            {'name': 'training-profile.json', 'sha256': '3'*64},
            {'name': 'profile.json', 'sha256': '4'*64}]}
        definition = {'inputs': dict.fromkeys(['dataset', 'trained_weights_npz',
            'trained_checkpoint_001_npz', 'trained_profile_json'])}
        project = {'data': [{'id': 'dataset', 'observed_sha256': '5'*64}]}
        result = client.restore_bindings(definition, project, run)
        self.assertEqual(result['trained_profile_json'], {'kind': 'run_artifact', 'run_id': run['id'],
            'artifact_name': 'training-profile.json', 'expected_sha256': '3'*64})
        self.assertEqual(result['trained_weights_npz']['artifact_name'], 'weights.npz')
        self.assertEqual(result['dataset'], {'reference_id': 'dataset', 'expected_sha256': '5'*64})
        with self.assertRaises(ValueError):
            client.restore_bindings(definition, project, {**run, 'artifacts': run['artifacts'][1:]})

    def test_restore_claim_requires_actual_cuda_train_same_origin_and_no_cpu_update(self):
        trained = {'status': 'complete', 'mode': 'train', 'optimizer_updates': 1,
                   'execution': {'device': 'cuda:0'}, 'closure': {'source': 'same'}, 'prediction_inventory': {'same': 1}}
        restored = {**trained, 'mode': 'restore', 'optimizer_updates': 0,
                    'execution': {'device': 'cpu', 'cuda_initialized': False}, 'originating_training_run': {'run_id': 'a'*32}}
        client.validate_receipts(trained, restored, 'a'*32)
        for changes in ({'optimizer_updates': 1}, {'execution': {'device': 'cpu', 'cuda_initialized': True}},
                        {'originating_training_run': {'run_id': 'b'*32}}, {'prediction_inventory': {'changed': 1}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                client.validate_receipts(trained, {**restored, **changes}, 'a'*32)


@unittest.skipUnless(ASSEMBLY.exists(), 'Requires explicitly assembled consumed pressure material')
class ActualSourceTests(unittest.TestCase):
    def test_actual_frozen_closure_is_measured_without_loading_model_or_application(self):
        before = set(sys.modules)
        receipt, assembly, project, recipes, _runtime = client.verify_sources(ROOT, ASSEMBLY, TULDOK, verify_runtime=False)
        self.assertEqual(receipt['project']['id'], project['id'])
        self.assertEqual(assembly['limits']['attempt_slots'], 2)
        self.assertEqual(recipes['train']['runtime_device'], 'cuda')
        self.assertGreater(len(receipt['inventory']), 30)
        self.assertFalse({'torch', 'numpy', 'train', 'app', 'psycopg'} & (set(sys.modules)-before))
        self.assertFalse(receipt['runtime_fully_verified'])


@unittest.skipUnless(ASSEMBLY.exists() and os.environ.get('RHEON_PRESSURE_DRIVER_TEST_DATABASE_URL'),
                     'Requires an explicitly selected disposable PostgreSQL test connection')
class ActualOwnerHTTPPreparation(unittest.TestCase):
    def test_current_http_driver_prepares_exact_closed_trial_and_stops_before_grant(self):
        """Exercise actual transport/registry/owners; no admission or worker allowed."""
        sys.path[:0] = [str(TULDOK), str(TULDOK / 'tests')]
        import psycopg
        from psycopg import sql
        from http.server import ThreadingHTTPServer
        import threading
        from app import Dataset, make_handler
        from auth_helpers import ExplicitOwnerFixture
        from metadata import connection_info
        from torch_runtimes import TorchRuntimes
        import torch_runtime
        import recipe_process
        selected_dsn = os.environ['RHEON_PRESSURE_DRIVER_TEST_DATABASE_URL']
        with patch.dict(os.environ, {'TULDOK_DATABASE_URL': selected_dsn}), tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset = Dataset(directory / 'catalog', project_roots={'pressure': ROOT})
            catalog_id = dataset.db.dataset_id
            server = None
            try:
                owner = ExplicitOwnerFixture.bootstrap(dataset)
                verification, assembly, project, recipes, runtime = client.verify_sources(ROOT, ASSEMBLY, TULDOK)
                with owner.bound(dataset):
                    TorchRuntimes(dataset.db, dataset.access, dataset.path).create(runtime, assembly['runtime']['manifest_sha256'])
                server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(dataset))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                endpoint = 'http://127.0.0.1:' + str(server.server_port)
                session = directory / 'owner.session'
                session.write_text(owner.token)
                session.chmod(0o600)
                prepared = directory / 'prepared.json'
                client.write(prepared, {'source_verification': verification,
                    'target': {'endpoint': endpoint, 'catalog_id': catalog_id, 'project_alias': 'pressure'}})
                gate = directory / 'test-only-gate.json'
                client.write(gate, {'schema': 'rheon-tuldok-migration-cutover-v1', 'm3_service_switched': True,
                    'm3_complete': False, 'ordinary_admissions_open': False,
                    'smoke_authorized': {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1},
                    'endpoint': endpoint, 'catalog_id': catalog_id, 'project_roots': {'pressure': str(ROOT)},
                    'app_source_sha256': verification['tuldok_source']['content_sha256'],
                    'app_sha256': client.digest(TULDOK / 'app.py'), 'rheon_binary_sha256': '0'*64})
                output = directory / 'output'
                output.mkdir()
                args = SimpleNamespace(project_root=ROOT, assembly=ASSEMBLY, tuldok_root=TULDOK,
                    project_alias='pressure', endpoint=endpoint, catalog_id=catalog_id,
                    catalog_root=dataset.path, session_file=session, agent_config=None, output=output,
                    prepared_receipt=prepared, prepared_sha256=client.digest(prepared),
                    m3_receipt=gate, m3_sha256=client.digest(gate), app_launch_receipt=gate,
                    app_launch_sha256=client.digest(gate))
                class StoppedBeforeGrant(Exception):
                    pass
                original = client.Journal.mutation
                def boundary(journal, phase, operation, arguments, invoke, **kwargs):
                    if phase == 'approve-trial':
                        raise StoppedBeforeGrant()
                    return original(journal, phase, operation, arguments, invoke, **kwargs)
                with patch.object(client.Journal, 'mutation', new=boundary), \
                     patch.object(recipe_process, 'launch', side_effect=AssertionError('inert preparation launched execution')), \
                     patch.object(dataset.studies, 'admit', side_effect=AssertionError('inert preparation admitted a Run')):
                    with self.assertRaises(StoppedBeforeGrant):
                        client.execute(args, verification, assembly, project, recipes, runtime)
                closed = json.loads((output / 'closed-trial.json').read_bytes())
                self.assertFalse(closed['study']['study']['open'])
                self.assertEqual(closed['study']['study']['trial_ids'], [closed['trial']['id']])
                self.assertEqual(closed['composition']['material']['model'], closed['model']['ref'])
                with owner.bound(dataset):
                    self.assertEqual(dataset.runs.list(), [])
                    self.assertEqual(dataset.db.execute('SELECT count(*) FROM model_trials').fetchone()[0], 1)
                    for table in ('access_execution_grants', 'access_execution_consumptions', 'model_attempt_reservations'):
                        self.assertEqual(dataset.db.execute('SELECT count(*) FROM ' + table).fetchone()[0], 0)
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)
                dataset.close()
                with psycopg.connect(connection_info(), autocommit=True) as connection:
                    connection.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier('catalog_' + catalog_id)))


if __name__ == '__main__':
    unittest.main()
