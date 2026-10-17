"""Local-evidence checks only; no authoritative gate/catalog/endpoint mutation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import build_m3_gates as builder
import m3_orchestrator as m3


class GateBuilderBoundaries(unittest.TestCase):
    def test_current03_actual_payload_bytes_modes_sources_and_no_extra_files(self):
        self.assertEqual(builder.verify_payload(), 192)

    def test_existing_real_q3_a8_deployment_full74_pass_exact_actual_predicates(self):
        selection = m3.read(m3.OPS / 'm3-gate-inputs-existing.json')
        checks = builder.required_checks()
        self.assertEqual(len(selection['receipts']), 6)
        for name, row in selection['receipts'].items():
            path = builder.bound(row)
            builder.evaluate(m3.read(path), checks[name])
            if name == 'q3_cpu_restore':
                changed = m3.read(path)
                changed['execution']['cuda_initialized'] = True
                with self.assertRaisesRegex(ValueError, 'PASS predicate failed'):
                    builder.evaluate(changed, checks[name])

    def test_missing_q2_browser_proofs_never_become_a_gate(self):
        selection = m3.read(m3.OPS / 'm3-gate-inputs-existing.json')
        with patch.object(m3, 'source_witness', return_value=selection['tuldok_source']):
            with self.assertRaisesRegex(ValueError, 'actual completed receipt selections missing'):
                builder.build(selection)

    def test_pending_independent_review_cannot_satisfy_passed_final_acceptance(self):
        pending = {'schema': 'rheon-independent-q2-final-review-v1', 'status': 'pending', 'attempt': '04',
                   'scope': 'isolated-rehearsal', 'accepted': False,
                   'limits': {'production_or_service_cutover_qualified': False}}
        with self.assertRaisesRegex(ValueError, 'PASS predicate failed'):
            builder.evaluate(pending, builder.required_checks()['q2_independent'])

    def test_boolean_true_is_not_interchangeable_with_numeric_one(self):
        with self.assertRaises(ValueError):
            builder.evaluate({'accepted': 1}, [builder.eq('accepted', True)])

    def test_cli_check_only_rejects_missing_proof_and_never_imports_database(self):
        path = m3.OPS / 'm3-gate-inputs-existing.json'
        script = '''import runpy,sys
def guard(event,args):
 if event=="socket.connect" or (event=="import" and args[0] in ("metadata","psycopg","app")):
  print("FORBIDDEN_ACCESS", file=sys.stderr);raise AssertionError("No catalog/endpoint")
sys.addaudithook(guard)
sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', script, str(m3.OPS),
            str(Path(builder.__file__)), '--selection', str(path), '--selection-sha256', m3.sha(path), '--check-only'],
            capture_output=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(b'FORBIDDEN_ACCESS', result.stderr)
        self.assertIn(b'no PASS evidence is inferred', result.stderr)
        self.assertEqual(result.stdout, b'')

    def test_explicit_write_is_exclusive_and_uses_only_disposable_ops_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / 'selected.json'; selection.write_text('{}')
            args = ['--selection', str(selection), '--selection-sha256', m3.sha(selection), '--write-gates']
            with patch.object(builder, 'build', return_value={'fixture': 'all completed', 'receipts': {}}), \
                 patch.object(m3, 'OPS', root), patch('builtins.print'):
                builder.main(args)
                first = (root / 'm3-gates.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    builder.main(args)
            self.assertEqual((root / 'm3-gates.json').read_bytes(), first)
            self.assertEqual((root / 'm3-gates.json').stat().st_mode & 0o077, 0)


if __name__ == '__main__':
    unittest.main()
