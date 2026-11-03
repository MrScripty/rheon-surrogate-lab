"""Availability evidence refuses stale or merely similar source/protocol bytes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_full_state_evidence import audit, observe_claim, protocol_index, summarize


class EvidenceAvailabilityTests(unittest.TestCase):
    def test_changed_source_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'model.py'
            original = b'original historical source\n'
            source.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            self.assertEqual(observe_claim(root, 'model.py', expected)['status'], 'exact')
            source.write_bytes(b'current changed source\n')
            self.assertEqual(observe_claim(root, 'model.py', expected)['status'], 'changed')
            source.unlink()
            self.assertEqual(observe_claim(root, 'model.py', expected)['status'], 'missing')

    def test_protocol_identity_is_bytes_not_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / 'registered-protocols'
            nested.mkdir()
            raw = b'{"horizons": [1, 2, 3, 4]}\n'
            selected = nested / 'renamed-protocol.json'
            selected.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(protocol_index(root)[expected], [selected])
            selected.write_text('{"horizons": [1]}\n')
            self.assertNotIn(expected, protocol_index(root))

    def test_recorded_source_conflict_and_missing_protocol_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'docs/plans').mkdir(parents=True)
            protocols = root / 'protocols'
            protocols.mkdir()
            source = root / 'model.py'
            source.write_bytes(b'current source\n')
            current = hashlib.sha256(source.read_bytes()).hexdigest()
            historical = hashlib.sha256(b'older source\n').hexdigest()
            evidence = root / 'result.json'
            evidence.write_text(json.dumps(dict(status='completed',
                source_sha256={'model.py': historical}, protocol_sha256='0' * 64)))
            protocol = protocols / 'unrelated.json'
            protocol.write_text(json.dumps(dict(source_sha256={'model.py': current})))
            worklist = root / 'worklist.json'
            worklist.write_text(json.dumps(dict(named_entries=[dict(
                item_id='expanded-variants/0', record=dict(id='candidate', device='gpu'),
                historical_evidence_routes=dict(sources=['../../result.json']))])))
            result = audit(root, worklist, protocols)
            summary = summarize(result)
            self.assertEqual(summary['source_status'], {'changed': 1})
            self.assertEqual(summary['protocol_status'], {'not_located_in_declared_corpus': 1})
            self.assertFalse(summary['complete_execution_bindings'])
            self.assertEqual(result['variants'][0]['execution_adapter_review'], 'pending')


if __name__ == '__main__':
    unittest.main()
