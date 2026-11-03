"""Exercise destructive compaction and recovery on disposable backup fixtures."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('compact', Path(__file__).with_name('compact_backups.py'))
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


class CompactionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original_roots = compact.ROOT, compact.BASE, compact.KEEPER
        self.addCleanup(self.restore_roots)
        compact.ROOT = Path(self.temp.name)
        compact.BASE = compact.ROOT / 'backups'
        compact.KEEPER = compact.BASE / 'completed-current-backup-04'
        for index, name in enumerate((*compact.RETIRED, compact.KEEPER.name)):
            root = compact.BASE / name
            (root / 'catalog').mkdir(parents=True)
            # Same length, different bytes: retain unique payloads and metadata.
            (root / 'metadata.json').write_text(str(index))
            (root / 'catalog/shared').write_bytes(b'identical payload')
            (root / 'catalog/unique').write_bytes(bytes([index]))
            files = {n: dict(kind='file', bytes=(root / 'catalog' / n).stat().st_size,
                             sha256=compact.digest(root / 'catalog' / n))
                     for n in ('shared', 'unique')}
            (root / 'backup-receipt.json').write_text(json.dumps(dict(
                metadata_sha256=compact.digest(root / 'metadata.json'), files=files)))
        self.manifest = compact.ROOT / 'manifest.json'
        self.report = compact.ROOT / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            compact.scan(self.manifest)

    def restore_roots(self):
        compact.ROOT, compact.BASE, compact.KEEPER = self.original_roots

    def test_compaction_and_verified_reconstruction(self):
        with contextlib.redirect_stdout(io.StringIO()):
            compact.apply(self.manifest, self.report)
        for name in compact.RETIRED:
            root = compact.BASE / name
            self.assertFalse((root / 'catalog/shared').exists())
            self.assertTrue((root / 'catalog/unique').is_file())
            destination = compact.ROOT / ('restored-' + name)
            with contextlib.redirect_stdout(io.StringIO()):
                compact.materialize(self.manifest, name, destination)
            self.assertEqual((destination / 'metadata.json').read_bytes(),
                             (root / 'metadata.json').read_bytes())
            self.assertEqual((destination / 'catalog/shared').read_bytes(), b'identical payload')
            self.assertEqual((destination / 'catalog/shared').stat().st_nlink, 1)

    def test_changed_keeper_aborts_before_removal(self):
        (compact.KEEPER / 'catalog/shared').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            compact.apply(self.manifest, self.report)
        self.assertTrue((compact.BASE / compact.RETIRED[0] / 'catalog/shared').is_file())

    def test_changed_original_aborts_before_removal(self):
        (compact.BASE / compact.RETIRED[1] / 'catalog/shared').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            compact.apply(self.manifest, self.report)
        self.assertTrue((compact.BASE / compact.RETIRED[0] / 'catalog/shared').is_file())

    def test_forged_manifest_entry_rejected(self):
        value = json.loads(self.manifest.read_text())
        value['backups'][0]['duplicates'][0]['path'] = '../../metadata.json'
        self.manifest.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            compact.apply(self.manifest, self.report)

    def test_relative_manifest_and_resumable_completion(self):
        relative = Path(os.path.relpath(self.manifest, Path.cwd()))
        with contextlib.redirect_stdout(io.StringIO()):
            compact.apply(relative, self.report)
            compact.apply(relative, self.report.with_name('resumed-report.json'))
        result = json.loads(self.report.read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['files_removed'], 2)


if __name__ == '__main__':
    unittest.main()
