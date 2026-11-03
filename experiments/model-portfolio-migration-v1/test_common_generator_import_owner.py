"""Focused custody and replay barriers; no live API or project mutation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import import_common_generator_measurements as owner


class ImportOwnerTests(unittest.TestCase):
    def test_existing_status_blocks_replay_before_qualification_or_authentication(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            before,status = job/'before.json',job/'status.json'
            before.write_text('{}')
            status.write_text('{}')
            with patch.object(owner,'JOB',job),patch.object(owner,'BEFORE',before),patch.object(owner,'STATUS',status), \
                    patch.object(owner,'qualified',side_effect=AssertionError('Must reject before reading more inputs')):
                with self.assertRaisesRegex(ValueError,'^Capture before custody and preserve original request ownership$'):
                    owner.adopt()
            self.assertEqual(list(job.glob('*.request.json')),[])

    def test_valid_header_then_same_length_changed_body_is_rejected(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            folder = project/'.tuldok'
            folder.mkdir()
            value = dict(name='test-project',project_id='a'*32,inventory=[],roots=[],receipts=[])
            value['checkpoint'] = owner.digest(owner.encode(value))
            path = folder/'project.json'
            path.write_bytes(owner.encode(value))
            with patch.object(owner,'PROJECT',project):
                self.assertEqual(owner.header()[1],value)
                changed = path.read_bytes().replace(b'test-project',b'best-project')
                self.assertEqual(len(changed),path.stat().st_size)
                path.write_bytes(changed)
                with self.assertRaisesRegex(ValueError,'^Canonical header encoding or checkpoint differs$'):
                    owner.header()

    def test_fingerprint_ignores_only_explicit_new_metadata(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            blobs = project/'artifacts/sha256'
            blobs.mkdir(parents=True)
            old = blobs/('a'*64)
            old.write_bytes(b'original model fixture')
            with patch.object(owner,'PROJECT',project):
                original = owner.payloads()
                added = blobs/('b'*64)
                added.write_bytes(b'new measurement fixture')
                self.assertEqual(owner.payloads(exclude=[added.name]),original)
                self.assertNotEqual(owner.payloads(),original)
                replacement = blobs/('c'*64)
                replacement.write_bytes(old.read_bytes())
                replacement.replace(old)
                self.assertNotEqual(owner.payloads(exclude=[added.name]),original)


if __name__ == '__main__':
    unittest.main()
