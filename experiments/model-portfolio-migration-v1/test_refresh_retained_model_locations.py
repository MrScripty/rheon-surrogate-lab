"""Compacted evidence resolves to exact retained bytes without new copies."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from refresh_retained_model_locations import resolve


class RetainedLocatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/'old'
        self.keep = self.root/'keeper'
        (self.old/'nested').mkdir(parents=True)
        (self.keep/'catalog').mkdir(parents=True)
        self.original = self.old/'nested/weights.npz'
        self.retained = self.keep/'catalog/source.bin'
        self.retained.write_bytes(b'exact original checkpoint bytes')
        self.retained.chmod(0o400)
        self.expected = hashlib.sha256(self.retained.read_bytes()).hexdigest()
        self.files = {'nested/weights.npz':dict(keeper_path='source.bin',sha256=self.expected,
                       bytes=self.retained.stat().st_size,mode=0o400)}

    def test_resolves_exact_keeper_without_recreating_original(self):
        actual = resolve(self.original,self.expected,old=self.old,keeper=self.keep,files=self.files)
        self.assertEqual(actual,self.retained)
        self.assertFalse(self.original.exists())
        self.assertEqual(actual.stat().st_nlink,1)

    def test_changed_expected_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Compaction mapping differs from original evidence identity'):
            resolve(self.original,'f'*64,old=self.old,keeper=self.keep,files=self.files)

    def test_changed_keeper_is_rejected(self):
        self.retained.chmod(0o600)
        self.retained.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Surviving model evidence differs from its recorded SHA256'):
            resolve(self.original,self.expected,old=self.old,keeper=self.keep,files=self.files)

    def test_keeper_mapping_cannot_escape_its_catalog(self):
        self.files['nested/weights.npz']['keeper_path'] = '../outside'
        with self.assertRaisesRegex(ValueError,'Unsafe manifest path'):
            resolve(self.original,self.expected,old=self.old,keeper=self.keep,files=self.files)


if __name__ == '__main__':
    unittest.main()
