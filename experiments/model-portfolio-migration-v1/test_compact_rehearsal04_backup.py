"""Compacted backup materialization, using tiny independent filesystem fixtures."""
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import compact_rehearsal04_backup as owner


def sha(raw):return hashlib.sha256(raw).hexdigest()


class Materialization(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rheon-backup-materialization-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.old=self.root/'old';self.keeper=self.root/'keeper'
        for base in (self.old,self.keeper):(base/'catalog/nested').mkdir(parents=True)
        duplicate=b'Exact captured source; never executed\n';unique=b'Original coordinator bytes\n'
        (self.keeper/'catalog/nested/source.txt').write_bytes(duplicate)
        (self.keeper/'catalog/nested/source.txt').chmod(0o400)
        (self.old/'catalog/nested/journal.json').write_bytes(unique)
        (self.old/'catalog/nested/journal.json').chmod(0o600)
        (self.old/'catalog/nested').chmod(0o750)
        self.metadata=b'{"different_original_snapshot":4}\n'
        receipt=dict(metadata_sha256=sha(self.metadata),files={
            'nested':dict(kind='directory'),
            'nested/source.txt':dict(kind='file',bytes=len(duplicate),sha256=sha(duplicate)),
            'nested/journal.json':dict(kind='file',bytes=len(unique),sha256=sha(unique))})
        (self.old/'metadata.json').write_bytes(self.metadata)
        (self.old/'metadata.json').chmod(0o600)
        (self.old/'backup-receipt.json').write_text(json.dumps(receipt))
        (self.old/'backup-receipt.json').chmod(0o600)
        (self.keeper/'backup-receipt.json').write_text('{"earlier_snapshot":3}')
        self.plan=self.root/'plan.json';self.marker=self.old/'COMPACTED-BACKUP.json'
        self.plan.write_text(json.dumps(dict(original='old',keeper='keeper',
            original_receipt_sha256=owner.digest(self.old/'backup-receipt.json'),
            keeper_receipt_sha256=owner.digest(self.keeper/'backup-receipt.json'),
            duplicates=[dict(path='nested/source.txt')],unique=[dict(path='nested/journal.json')])) )
        self.marker.write_text(json.dumps(dict(status='compacted_requires_materialization',manifest_sha256=owner.digest(self.plan))))
        for name,value in dict(ROOT=self.root,OLD=self.old,KEEPER=self.keeper,PLAN=self.plan,MARKER=self.marker).items():
            patched=patch.object(owner,name,value);patched.start();self.addCleanup(patched.stop)

    def test_materializes_exact_unique_and_duplicate_files_with_independent_custody(self):
        destination=self.root/'restored'
        owner.materialize(destination)
        self.assertEqual((destination/'metadata.json').read_bytes(),self.metadata)
        self.assertEqual(stat.S_IMODE((destination/'metadata.json').stat().st_mode),0o600)
        self.assertEqual(stat.S_IMODE((destination/'catalog/nested').stat().st_mode),0o750)
        restored=destination/'catalog/nested/source.txt';kept=self.keeper/'catalog/nested/source.txt'
        self.assertEqual(restored.read_bytes(),kept.read_bytes())
        self.assertEqual(stat.S_IMODE(restored.stat().st_mode),0o400)
        self.assertNotEqual(restored.stat().st_ino,kept.stat().st_ino)
        self.assertEqual(restored.stat().st_nlink,1)
        self.assertEqual((destination/'catalog/nested/journal.json').read_bytes(),b'Original coordinator bytes\n')

    def test_changed_keeper_receipt_rejects_before_destination_creation(self):
        (self.keeper/'backup-receipt.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):owner.materialize(self.root/'restored')
        self.assertFalse((self.root/'restored').exists())

    def test_materialization_cannot_write_inside_the_retained_keeper(self):
        with self.assertRaises(ValueError):owner.materialize(self.keeper/'new')
        self.assertFalse((self.keeper/'new').exists())


if __name__=='__main__':unittest.main()
