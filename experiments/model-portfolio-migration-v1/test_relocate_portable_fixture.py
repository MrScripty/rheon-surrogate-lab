"""Real Linux no-copy transport; native lifecycle is qualified separately."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import relocate_portable_fixture as subject


class PortableFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);p=patch.object(subject,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.project=self.root/'published';self.destination=self.root/'cold/root/project'
        self.archive=self.root/'cold/original-local';self.generated=self.root/'cold/generated-local'
        self.evidence=self.root/'cold/preparation.json'
        (self.project/'.tuldok').mkdir(parents=True);(self.project/'artifacts/sha256').mkdir(parents=True)
        (self.project/'.tuldok-local').mkdir();(self.project/'.tuldok-local/original').write_bytes(b'original local')
        self.raw=b'{"checkpoint":"expected"}\n';(self.project/'.tuldok/project.json').write_bytes(self.raw)
        (self.project/'artifacts/sha256/model').write_bytes(b'unique model');(self.project/'artifacts/sha256/model').chmod(0o400)
        self.destination.parent.mkdir(parents=True);self.sha=hashlib.sha256(self.raw).hexdigest()

    def prepare(self):return subject.prepare(self.project,self.destination,self.archive,self.evidence,'expected',self.sha)

    def test_same_files_round_trip_with_original_and_generated_local_preserved(self):
        before=(self.project/'artifacts/sha256/model').stat().st_ino
        self.prepare();self.assertFalse(self.project.exists());self.assertTrue(self.archive.exists())
        self.assertFalse((self.destination/'.tuldok-local').exists())
        (self.destination/'.tuldok-local').mkdir();(self.destination/'.tuldok-local/new').write_bytes(b'fresh native cache')
        result=subject.restore(self.evidence,self.generated)
        self.assertEqual(result['payload_copies'],0);self.assertFalse(self.destination.exists())
        self.assertEqual((self.project/'artifacts/sha256/model').stat().st_ino,before)
        self.assertEqual((self.project/'.tuldok-local/original').read_bytes(),b'original local')
        self.assertEqual((self.generated/'new').read_bytes(),b'fresh native cache')

    def test_busy_root_changed_checkpoint_and_occupied_destination_refuse_transport(self):
        with (self.project/'artifacts/sha256/model').open('rb'):
            with self.assertRaisesRegex(ValueError,'open file descriptor'):self.prepare()
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            subject.prepare(self.project,self.destination,self.archive,self.evidence,'different',self.sha)
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh'):self.prepare()
        self.assertTrue((self.project/'.tuldok-local/original').exists());self.assertFalse(self.evidence.exists())

    def test_interruption_after_project_rename_reconciles_original_inode(self):
        original=subject.move
        def interrupted(source,destination):
            original(source,destination)
            if source==self.project:raise RuntimeError('interrupted acknowledgement')
        with patch.object(subject,'move',interrupted):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.prepare()
        self.assertFalse(self.project.exists());self.assertTrue(self.destination.exists())
        subject.restore(self.evidence,self.generated)
        self.assertEqual((self.project/'artifacts/sha256/model').read_bytes(),b'unique model')

    def test_changed_portable_bytes_are_retained_for_diagnosis_without_restoration(self):
        self.prepare();path=self.destination/'artifacts/sha256/model';path.chmod(0o600);path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'portable bytes'):subject.restore(self.evidence,self.generated)
        self.assertTrue(path.exists());self.assertTrue(self.archive.exists());self.assertFalse(self.project.exists())


if __name__=='__main__':unittest.main()
