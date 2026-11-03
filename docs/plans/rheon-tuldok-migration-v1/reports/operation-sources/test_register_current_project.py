"""Offline Project04 registration gates; no authority, socket or catalog access."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('register_current_project', HERE / 'register_current_project.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def documents():
    cutover = {'schema': 'rheon-tuldok-migration-cutover-v1', 'm3_service_switched': True,
        'm3_complete': False, 'ordinary_admissions_open': False, 'endpoint': owner.ENDPOINT, 'catalog_id': owner.CATALOG}
    deployment, digest = owner.checked(owner.APPLICATION / 'deployment-manifest.json')
    app_sha = next(row['sha256'] for row in deployment['files'] if row['path'] == 'app.py')
    launch = {'status': 'current_service_launched_smoke_pending', 'endpoint': owner.ENDPOINT,
        'catalog_id': owner.CATALOG, 'endpoint_owned_by_selected_pid': True, 'structural_service_switch_only': True,
        'deployment': str(owner.APPLICATION), 'deployment_manifest_sha256': digest, 'app_sha256': app_sha,
        'project_roots': {'current-lab': str(owner.ROOT), 'pressure': str(owner.ROOT)}}
    project, _ = owner.checked(owner.ROOT / owner.MANIFEST, owner.MANIFEST_SHA)
    return cutover, launch, deployment, digest, project


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_exact_public_documents_and_both_root_aliases(self):
        owner.validate(*documents())
        cases = [(0, 'ordinary_admissions_open', True), (0, 'm3_service_switched', False),
                 (0, 'catalog_id', '0'*32), (1, 'endpoint_owned_by_selected_pid', False),
                 (1, 'deployment_manifest_sha256', '0'*64), (1, 'deployment', str(owner.MIGRATION/'Tuldok')),
                 (1, 'project_roots', {'current-lab': str(owner.ROOT), 'pressure': '/tmp'}),
                 (4, 'id', '0'*32)]
        for index, key, value in cases:
            values = list(copy.deepcopy(documents())); values[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): owner.validate(*values)

    def test_local_preparation_never_reads_private_authority_or_opens_network(self):
        cutover, launch, *_ = documents()
        first, second = self.path/'cutover.json', self.path/'launch.json'
        owner.persist(first, cutover, exclusive=True); owner.persist(second, launch, exclusive=True)
        args = SimpleNamespace(cutover_receipt=first, cutover_sha256=owner.checked(first)[1],
            launch_receipt=second, launch_sha256=owner.checked(second)[1], application=owner.APPLICATION,
            owner_session_file=self.path/'private-authority-must-not-be-read', output=self.path/'must-not-be-created')
        with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
            record, _launch, _api = owner.prepare(args)
        self.assertEqual(record['request']['operation'], 'project_snapshots.open')
        self.assertEqual(record['request']['body'], {'root': 'current-lab', 'manifest_path': owner.MANIFEST})
        self.assertEqual(record['manifest_sha256'], owner.MANIFEST_SHA)
        self.assertFalse(args.owner_session_file.exists())
        self.assertFalse(args.output.exists())

    def test_changed_gate_and_symlink_are_refused(self):
        path = self.path/'gate.json'; owner.persist(path, {'value': 1}, exclusive=True)
        _, digest = owner.checked(path)
        path.write_text('{"value":2}')
        with self.assertRaises(ValueError): owner.checked(path, digest)
        link = self.path/'link.json'; link.symlink_to(path)
        with self.assertRaises(OSError): owner.checked(link)

    def test_unknown_registration_commit_is_never_replayed(self):
        path = self.path/'journal.json'; calls = []
        class Unknown(Exception): code, status = 'commit_unknown', 503
        def invoke(): calls.append('dispatch'); raise Unknown()
        with self.assertRaises(Unknown): owner.dispatch(path, {'request': {}}, invoke)
        record = json.loads(path.read_bytes())
        self.assertEqual(record['status'], 'unresolved_inspect_durable_state')
        self.assertEqual(record['code'], 'commit_unknown')
        with self.assertRaises(FileExistsError): owner.dispatch(path, {'request': {}}, invoke)
        self.assertEqual(calls, ['dispatch'])

    def test_normal_snapshot_proof_does_not_claim_execution(self):
        value = {'id': owner.PROJECT, 'schema_version': 2, 'root': 'current-lab',
            'manifest_path': owner.MANIFEST, 'manifest_sha256': owner.MANIFEST_SHA,
            'manifest_status': 'unchanged', 'mode': 'inspection', 'execution': 'unsupported'}
        self.assertEqual(owner.prove_project(value)['id'], owner.PROJECT)
        path = self.path/'journal.json'
        with self.assertRaises(ValueError):
            owner.dispatch(path, {'request': {}}, lambda: {**value, 'id': '0'*32})
        self.assertEqual(json.loads(path.read_bytes())['status'], 'unresolved_inspect_durable_state')


if __name__ == '__main__': unittest.main()
