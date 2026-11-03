"""Bounded catalog reconstruction and fail-before-delete compaction fixtures."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import compact_retired_catalog as subject


def sha(raw):return hashlib.sha256(raw).hexdigest()


class RetiredCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/'runs/model-portfolio-migration-v1'
        self.keep=self.root/'runs/dependency-migration-v1/completed-current-backup-04'
        self.old=self.root/'runs/dependency-migration-v1/catalog-rehearsal-02'
        for p in (self.base/'discovery',self.root/'docs/plans/model-portfolio-migration-v1/reports',self.keep/'catalog/source',self.old/'nested'):
            p.mkdir(parents=True,exist_ok=True)
        raw=b'exact source bytes';p=self.keep/'catalog/source/file.bin';p.write_bytes(raw);p.chmod(0o400)
        original=self.old/'nested/renamed.bin';original.write_bytes(raw);original.chmod(0o400)
        (self.old/'.tuldok.json').write_bytes(b'unique root metadata')
        metadata=b'{"original":"SQL"}\n';(self.keep/'metadata.json').write_bytes(metadata)
        receipt=dict(dataset_id='a7da4a8de53541a2a0e2aad08005bb71',metadata_sha256=sha(metadata),files={
            'source':dict(kind='directory'),'source/file.bin':dict(kind='file',bytes=len(raw),sha256=sha(raw))})
        (self.keep/'backup-receipt.json').write_text(json.dumps(receipt))
        for key,value in dict(ROOT=self.root,BASE=self.base,KEEPER=self.keep).items():
            self.addCleanup(patch.stopall);patch.object(subject,key,value).start()
        patch.object(subject,'inactive',lambda:None).start();subject.select_catalog(2)

    def test_compaction_reconstructs_exact_independent_bytes_modes_and_root_metadata(self):
        subject.compact();self.assertFalse((self.old/'nested/renamed.bin').exists())
        self.assertEqual((self.old/'.tuldok.json').read_bytes(),b'unique root metadata')
        destination=self.root/'reconstructed';subject.materialize(destination)
        restored=destination/'nested/renamed.bin';kept=self.keep/'catalog/source/file.bin'
        self.assertEqual(restored.read_bytes(),kept.read_bytes());self.assertEqual(stat.S_IMODE(restored.stat().st_mode),0o400)
        self.assertNotEqual(restored.stat().st_ino,kept.stat().st_ino);self.assertEqual(restored.stat().st_nlink,1)
        self.assertEqual((destination/'.tuldok.json').read_bytes(),b'unique root metadata')

    def test_changed_keeper_refuses_compaction_before_any_source_file_is_removed(self):
        (self.keep/'catalog/source/file.bin').chmod(0o600);(self.keep/'catalog/source/file.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Keeper file'):subject.compact()
        self.assertTrue((self.old/'nested/renamed.bin').exists());self.assertFalse(subject.MARKER.exists())

    def test_unlisted_catalog_and_keeper_destination_are_rejected(self):
        with self.assertRaises(ValueError):subject.select_catalog(4)
        subject.compact()
        with self.assertRaises(ValueError):subject.materialize(self.keep/'new')
        self.assertFalse((self.keep/'new').exists())


if __name__=='__main__':unittest.main()
