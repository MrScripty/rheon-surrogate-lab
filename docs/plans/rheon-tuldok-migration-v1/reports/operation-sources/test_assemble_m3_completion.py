"""Four local completion boundaries; no live endpoint/catalog/authority access."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import assemble_m3_completion as completion


class CompletionBoundaries(unittest.TestCase):
    def test_import_help_and_fixed_evidence_are_local_and_machine_bytes_stay_pending(self):
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(Path(completion.__file__)), '--help'],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        for name, (path, digest) in completion.FIXED.items():
            value = completion.bound({'path': str(path), 'sha256': digest})
            if name == 'machine': self.assertEqual(value['status'], 'machine_qualified_browser_pending')
            if name == 'cutover': self.assertFalse(value['m3_complete'])
        self.assertFalse(any(name in sys.modules for name in ('metadata', 'psycopg', 'app', 'numpy', 'torch')))

    def test_pending_independent_review_cannot_authorize_completion(self):
        pending = {'schema': 'rheon-live-machine-browser-independent-review-v1', 'status': 'pending', 'accepted': False}
        with self.assertRaisesRegex(ValueError, 'independent PASS'):
            completion.validate_review(pending, {}, {})

    def test_exact_existing_field_reuse_required_even_when_browser_says_passed(self):
        machine = completion.bound({'path': str(completion.FIXED['machine'][0]), 'sha256': completion.FIXED['machine'][1]})
        project = completion.bound({'path': str(completion.FIXED['project_manifest'][0]), 'sha256': completion.FIXED['project_manifest'][1]})
        bindings = {name: {'path': str(path), 'sha256': digest} for name, (path, digest) in completion.FIXED.items()}
        browser = {'schema': 'rheon-root-gated-live-browser-readback-v1', 'status': 'passed',
            'catalog_id': completion.CATALOG, 'endpoint': completion.ENDPOINT, 'project_id': project['id'],
            'project_manifest_sha256': bindings['project_manifest']['sha256'],
            'collection': {'record_id': completion.EXISTING_RECORD, 'content_hash': completion.FIELD_SOURCE,
                'read_only_reuse': True, 'created_new_item': False, 'existing_record_lookup': {'collection_writes': 0}},
            'mutations': [], 'extra_runs': 0, 'approval_or_admission_or_promotion': False,
            'trial': machine['trial'], 'field_view': {'checkpoint_sha256': machine['checkpoint_document_sha256'],
                'cpu_pressure_sha256': machine['probe']['pressure_sha256'], 'units': 'Pa', 'dtype': '<f8'},
            'gates': [bindings[name] for name in ('cutover', 'launch', 'machine', 'project_provenance')],
            'activity': {'authority': 'external_observation', 'first_sequence': 1, 'last_sequence': 2,
                'healthy': True, 'state': 'running', 'phase': 'Dependency migration observation',
                'progress': {'completed': 74, 'total': 74}, 'machine_status': machine['status'],
                'message': '74/74 current regression case receipts passed; 0 failed, 0 unreadable; typed-pressure-live-01: status=' + machine['status']},
            'scientific_acceptance': 'not_asserted', 'screenshots': ['fixture-only-no-browser-claim']}
        browser.update({key: machine[key] for key in ('model', 'composition_sha256', 'training_run_id', 'restore_run_id', 'checkpoint_sha256')})
        browser['activity']['message_sha256'] = hashlib.sha256(browser['activity']['message'].encode()).hexdigest()
        launcher = {'status': 'passed', 'returncode': 0, 'owned_child_reaped': True,
                    'cleanup_error_kind': None, 'import_dispatch_may_have_committed': False}
        completion.validate_browser(browser, launcher, machine, project, bindings)
        for changes in ({'created_new_item': True}, {'read_only_reuse': False}, {'record_id': '0'*32}):
            changed = copy.deepcopy(browser); changed['collection'].update(changes)
            with self.assertRaisesRegex(ValueError, 'reuse exact existing field'):
                completion.validate_browser(changed, launcher, machine, project, bindings)
        changed = copy.deepcopy(browser)
        changed['activity']['message'] = 'Observation unavailable: OSError'
        changed['activity']['message_sha256'] = hashlib.sha256(changed['activity']['message'].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'healthy 74/74'):
            completion.validate_browser(changed, launcher, machine, project, bindings)

    def test_writer_requires_explicit_flag_and_never_overwrites_even_in_disposable_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); selection = root / 'selection.json'; selection.write_text('{}')
            common = ['--selection', str(selection), '--selection-sha256', completion.sha(selection)]
            with patch.object(completion, 'assemble', return_value={'fixture': True}), patch.object(completion, 'OPS', root), patch('builtins.print'):
                completion.main(common + ['--check-only'])
                self.assertFalse((root / 'm3-complete.json').exists())
                completion.main(common + ['--write-completion'])
                first = (root / 'm3-complete.json').read_bytes()
                with self.assertRaises(FileExistsError): completion.main(common + ['--write-completion'])
            self.assertEqual((root / 'm3-complete.json').read_bytes(), first)


if __name__ == '__main__': unittest.main()
