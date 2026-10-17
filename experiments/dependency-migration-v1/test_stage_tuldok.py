import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import stage_tuldok as deployment


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.destination = self.base / 'deployed/current-01'
        for name, raw in {'app.py': b'"""Inert copying contract fixture."""\n',
                          'static/app.js': b'const version = 1;\n',
                          'docs/contracts/current.md': b'Current contract\n',
                          'tools/migrations/rheon-lab-once/convert.py': b'converter must not deploy\n',
                          'tests/test_rheon_lab_migration.py': b'migration tests must not deploy\n',
                          '.github/workflows/tests.yml': b'CI must not deploy\n',
                          'data/weights.npz': b'weights must not deploy\n',
                          'owner.private.json': b'credentials must not deploy\n'}.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True, capture_output=True)
        self.git('add', '.')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source fixture')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.source), *args], check=True, capture_output=True)

    def build(self, smoke=None):
        return deployment.stage(self.source, self.destination, sys.executable,
            smoke_runner=smoke or (lambda *_: {'status': 'passed', 'scope': 'copying fixture only'}))

    def test_exact_dirty_bytes_modes_origin_and_excluded_payloads(self):
        (self.source / 'app.py').write_bytes(b'"""Current dirty bytes."""\n')
        (self.source / 'app.py').chmod(0o755)
        value = self.build()
        manifest, digest = deployment.verify(self.destination)
        self.assertEqual(value['manifest_sha256'], digest)
        self.assertEqual(value['app_sha256'], hashlib.sha256((self.source / 'app.py').read_bytes()).hexdigest())
        self.assertEqual(manifest['origin']['git_head'], self.git('rev-parse', 'HEAD').stdout.decode().strip())
        self.assertEqual((self.destination / 'app.py').stat().st_mode & 0o777, 0o755)
        self.assertEqual({r['path'] for r in manifest['files']}, {'app.py', 'static/app.js', 'docs/contracts/current.md'})
        for name in ('tests', 'tools', '.github', '.git', 'data', 'owner.private.json'):
            self.assertFalse((self.destination / name).exists())

    def test_changed_source_during_smoke_fails_without_partial_artifact(self):
        def changed(*_):
            (self.source / 'app.py').write_bytes(b'changed after copy\n')
            return {'status': 'passed'}
        with self.assertRaisesRegex(ValueError, 'Source'):
            self.build(changed)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.destination.parent.glob('.*.stage-*')))

    def test_smoke_failure_never_publishes(self):
        def failed(*_):
            raise ValueError('actual smoke failed')
        with self.assertRaisesRegex(ValueError, 'smoke failed'):
            self.build(failed)
        self.assertFalse(self.destination.exists())

    def test_selected_symlink_and_unknown_resource_refused(self):
        (self.source / 'static/app.js').unlink()
        (self.source / 'static/app.js').symlink_to(self.source / 'owner.private.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.build()
        self.assertFalse(self.destination.exists())
        (self.source / 'static/app.js').unlink()
        (self.source / 'static/app.js').write_bytes(b'current\n')
        (self.source / 'static/unknown.json').write_text('{"private":"must not copy"}')
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            self.build()

    def test_concurrent_destination_is_not_overwritten_and_existing_attempt_refused(self):
        def raced(*_):
            self.destination.mkdir()
            return {'status': 'passed'}
        with self.assertRaises(FileExistsError):
            self.build(raced)
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        with self.assertRaisesRegex(ValueError, 'absent destination'):
            self.build()

    def test_verify_rejects_mutated_extra_and_missing_payloads(self):
        self.build()
        path = self.destination / 'static/app.js'
        original = path.read_bytes()
        path.write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'differs'):
            deployment.verify(self.destination)
        path.write_bytes(original)
        extra = self.destination / 'tests/test_retired.py'
        extra.parent.mkdir()
        extra.write_bytes(b'must not exist')
        with self.assertRaisesRegex(ValueError, 'extra files'):
            deployment.verify(self.destination)
        extra.unlink()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            deployment.verify(self.destination)

    def test_receipt_failure_preserves_known_published_artifact(self):
        # Fail actual CLI receipt publication after the real no-replace artifact
        # publication. Reconciliation must retain the published owner bytes.
        real_stage = deployment.stage
        def checked_stage(source, destination, python):
            return real_stage(source, destination, python,
                smoke_runner=lambda *_: {'status': 'passed', 'scope': 'copying fixture only'})
        receipt = self.base / 'missing-parent/receipt.json'
        arguments = ['stage_tuldok.py', '--source', str(self.source), '--destination', str(self.destination),
                     '--python', sys.executable, '--receipt', str(receipt)]
        with patch.object(sys, 'argv', arguments), patch.object(deployment, 'stage', side_effect=checked_stage):
            with self.assertRaises(FileNotFoundError):
                deployment.main()
        manifest, digest = deployment.verify(self.destination)
        self.assertEqual(manifest['source_content_sha256'], deployment.sha(deployment.encoded(manifest['files'])))
        self.assertEqual((self.destination / 'app.py').read_bytes(), (self.source / 'app.py').read_bytes())
        self.assertEqual(len(digest), 64)
        self.assertFalse(receipt.exists())


if __name__ == '__main__':
    unittest.main()
