"""Offline boundary checks; never access the original process, DB or service."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('m3_reviewed', Path(__file__).with_name('m3_orchestrator.py'))
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)


class OfflineBoundaries(unittest.TestCase):
    def test_help_and_import_never_load_application_or_numerical_modules(self):
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(Path(m3.__file__)), '--help'],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(any(name in sys.modules for name in ('metadata', 'app', 'torch', 'psycopg')))

    def test_real_invalid_action_exits_nonzero_before_catalog_endpoint_or_private_access(self):
        source = '''import runpy,sys
def guard(event,args):
 if event=="socket.connect" or (event=="import" and args[0] in ("metadata","psycopg","app")) or (event=="open" and isinstance(args[0],str) and (".private." in args[0] or args[0].startswith("/proc/"))):
  print("FORBIDDEN_ACCESS",file=sys.stderr);raise AssertionError("No original access")
sys.addaudithook(guard);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', source, str(Path(m3.__file__)),
            'preflight', '--receipt', '/tmp/m3-forbidden-action-never-written.json', '--publish-original'],
            capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'M3 step stopped (ValueError)', result.stderr)
        self.assertNotIn(b'FORBIDDEN_ACCESS', result.stderr)
        self.assertEqual(result.stdout, b'')

    def fixture(self, root):
        receipt = root / 'passed.json'
        receipt.write_text(json.dumps({'status': 'passed', 'count': 74}))
        artifact = root / 'artifact'
        artifact.write_bytes(b'exact artifact')
        row = {'path': str(receipt), 'sha256': m3.sha(receipt),
               'checks': [{'path': ['status'], 'equals': 'passed'}]}
        doc = {'format': 'rheon_m3_root_gates_v1',
               'tuldok_source': {'revision': 'fixture', 'tree': 'fixture', 'content_sha256': 'fixture'},
               'receipts': {name: copy.deepcopy(row) for name in
                            ('q2_independent', 'q3_pipeline', 'q8_lifecycle', 'q6_browser', 'deployment_review')},
               'artifacts': {name: {'path': str(artifact), 'sha256': m3.sha(artifact)}
                             for name in ('deployment_manifest', 'rheon_binary', 'schema_rehearsal')}}
        return doc, receipt

    def test_tuldok_gate_does_not_require_independent_rheon74(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            document, _receipt = self.fixture(path)
            gate = path / 'gate.json'; gate.write_text(json.dumps(document))
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']):
                actual, _proof = m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))
            self.assertNotIn('q7_full74', actual['receipts'])

    def test_missing_real_browser_or_independent_deployment_review_blocks(self):
        for name in ('q6_browser', 'deployment_review'):
            with self.subTest(missing=name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary); document, _receipt = self.fixture(path)
                del document['receipts'][name]
                gate = path / 'gate.json'; gate.write_text(json.dumps(document))
                with patch.object(m3, 'source_witness', return_value=document['tuldok_source']), \
                     self.assertRaisesRegex(ValueError, 'gates missing'):
                    m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))

    def test_schema_rehearsal_hash_binding_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); document, _receipt = self.fixture(path)
            del document['artifacts']['schema_rehearsal']
            gate = path / 'gate.json'; gate.write_text(json.dumps(document))
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']), \
                 self.assertRaisesRegex(ValueError, 'schema-rehearsal bindings missing'):
                m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))

    def test_changed_receipt_or_failed_predicate_blocks_before_any_owner_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); document, receipt = self.fixture(path)
            gate = path / 'gate.json'; gate.write_text(json.dumps(document))
            receipt.write_text('{"status":"failed"}')
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']), self.assertRaisesRegex(ValueError, 'hash'):
                m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))
            for row in document['receipts'].values(): row['sha256'] = m3.sha(receipt)
            gate.write_text(json.dumps(document))
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']), self.assertRaisesRegex(ValueError, 'failed'):
                m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))

    def test_optional74_gate_must_prove_all74(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); document, _receipt = self.fixture(path)
            document['receipts']['q7_full74'] = copy.deepcopy(document['receipts']['q2_independent'])
            gate = path / 'gate.json'; gate.write_text(json.dumps(document))
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']), self.assertRaisesRegex(ValueError, '74'):
                m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))
            document['receipts']['q7_full74']['checks'].append({'path': ['count'], 'equals': 74})
            gate.write_text(json.dumps(document))
            with patch.object(m3, 'source_witness', return_value=document['tuldok_source']):
                m3.gates(SimpleNamespace(gates=gate, gates_sha256=m3.sha(gate)))

    def test_action_flag_must_match_mode_before_private_selection_read(self):
        with patch.object(m3, 'selected', side_effect=AssertionError('must not read secrets')):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                m3.main(['preflight', '--receipt', '/tmp/never-m3.json', '--publish-original'])

    def test_durable_receipt_replace_has_no_pending_and_private_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            m3.persist(path, {'status': 'exact'})
            self.assertEqual(json.loads(path.read_bytes()), {'status': 'exact'})
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            self.assertFalse(path.with_suffix('.json.pending').exists())

    def test_prepared_identity_uses_converter_canonical_digest_not_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            specification = {'target': 'exact-original-fixture'}
            prepared = {'spec': specification, 'other': [1, 2, 3]}
            (stage / 'prepared.json').write_text(json.dumps(prepared, indent=2) + '\n')
            spec_path = stage / 'spec.json'; spec_path.write_text(json.dumps(specification))
            owner = SimpleNamespace(digest=m3.digest)
            args = SimpleNamespace(stage=stage, spec=spec_path, prepared_sha256=m3.digest(prepared))
            self.assertNotEqual(args.prepared_sha256, m3.sha(stage / 'prepared.json'))
            with patch.object(m3, 'stage_spec', return_value=(owner, specification)):
                self.assertEqual(m3.require_prepared(args, object()), (owner, prepared))
                args.prepared_sha256 = m3.sha(stage / 'prepared.json')
                with self.assertRaisesRegex(ValueError, 'document changed'):
                    m3.require_prepared(args, object())

    def test_reused_process_identity_never_receives_signal_and_pidfd_is_closed(self):
        with patch.object(m3.os, 'pidfd_open', return_value=123), \
             patch.object(m3, 'process', return_value={'start_ticks': 999}), \
             patch.object(m3.signal, 'pidfd_send_signal') as send, patch.object(m3.os, 'close') as close:
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                m3.interrupt_exact(99999, {'start_ticks': 1})
            send.assert_not_called()
            close.assert_called_once_with(123)

    def test_bookend_must_bind_exact_root_schema_rehearsal_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); backup = root / 'backup'; backup.mkdir()
            (backup / 'backup-receipt.json').write_text('{}')
            schema = root / 'schema.json'; schema.write_text('{"status":"passed"}')
            expected = {'path': str(schema), 'sha256': m3.sha(schema)}
            data = {'schema': 'rheon-fresh-stopped-original-bookend-v1', 'status': 'exact_backup_revalidated',
                    'schema_version': 8, 'dataset_id': 'fixture', 'stopped_original_pid': m3.PID,
                    'complete_backup': str(backup), 'backup_receipt_sha256': m3.sha(backup / 'backup-receipt.json')}
            data.update({key: True for key in ('all_domain_tables_equal', 'all_sequences_equal',
                'all_managed_hashes_equal', 'all_file_and_directory_modes_equal', 'metadata_pre_post_stable',
                'original_port_closed', 'original_source_command_absent')})
            path = root / 'bookend.json'
            for binding in (None, dict(expected, sha256='0'*64), expected):
                with self.subTest(binding=binding):
                    data['schema_rehearsal_receipt'] = binding
                    path.write_text(json.dumps(data))
                    args = SimpleNamespace(bookend=path, bookend_sha256=m3.sha(path), bookend_max_age=600)
                    with patch.object(m3, 'BACKUP', backup):
                        if binding == expected:
                            m3.bookend(args, {'dataset_id': 'fixture'}, expected, fresh=True)
                        else:
                            with self.assertRaisesRegex(ValueError, 'schema rehearsal receipt binding'):
                                m3.bookend(args, {'dataset_id': 'fixture'}, expected, fresh=True)

    def test_database_readonly_is_enforced_before_verify_only_owner_can_bind_credential(self):
        events = []
        connection = SimpleNamespace(execute=lambda statement: events.append(statement))
        db = SimpleNamespace(connection=connection)
        args = SimpleNamespace(stage=Path('/fixture/stage'), prepared_sha256='fixture')
        def publication(*_args, **kwargs):
            self.assertEqual(events, ['SET default_transaction_read_only = on'])
            self.assertTrue(kwargs['verify_only'])
            events.append('credential-binding-can-now-start')
            raise PermissionError('Expired credential fixture; denial must remain read-only')
        with self.assertRaises(PermissionError):
            m3.verify_conversion(SimpleNamespace(publish=publication), db, args, 'fixture-not-a-real-token')
        self.assertEqual(events, ['SET default_transaction_read_only = on', 'credential-binding-can-now-start'])


if __name__ == '__main__':
    unittest.main()
