"""Real filesystem proof for closed-input duplicate retirement and recovery."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compact_composition_inputs as subject


def sha(raw): return hashlib.sha256(raw).hexdigest()


class CompositionInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keeper = self.root/'published'; self.sources = {}
        self.payload = b'original model weights'; self.key = sha(self.payload)
        for name in ('primary', 'current', 'published'):
            root = self.root/name
            (root/'.tuldok/records/project_artifact').mkdir(parents=True)
            (root/'artifacts/sha256').mkdir(parents=True)
            document = json.dumps(dict(bytes=len(self.payload))).encode()
            record = root/'.tuldok/records/project_artifact/record.json'; record.write_bytes(document)
            manifest = dict(project_id=name, checkpoint=name+'-checkpoint', inventory=[dict(
                family='project_artifact', id=self.key, path='records/project_artifact/record.json',
                bytes=len(document), sha256=sha(document))])
            raw = json.dumps(manifest).encode(); (root/'.tuldok/project.json').write_bytes(raw)
            payload = root/'artifacts/sha256'/self.key; payload.write_bytes(self.payload); payload.chmod(0o400)
            if name != 'published': self.sources[name] = root, manifest['checkpoint'], sha(raw)
            else: self.keep_sha = sha(raw)
        self.plan = self.root/'plan.json'; self.report = self.root/'report.json'
        custody = dict(status='all_original_archive_scientific_facts_memberships_and_copied_payloads_exact',
                       composition_manifest_sha256=self.keep_sha, project_id='published',
                       checkpoint='published-checkpoint', project='published')
        publication = dict(status='native_committed_project_published_without_payload_copy',
                           result=dict(project_id='published', checkpoint='published-checkpoint'))
        for name, value in [('custody', custody), ('publication', publication)]:
            (self.root/(name+'.json')).write_text(json.dumps(value))
        for key, value in dict(ROOT=self.root, KEEPER=self.keeper, SOURCES=self.sources, PLAN=self.plan,
                               REPORT=self.report, CUSTODY=self.root/'custody.json',
                               PUBLICATION=self.root/'publication.json', KEEPER_MANIFEST=self.keep_sha).items():
            p = patch.object(subject, key, value); p.start(); self.addCleanup(p.stop)

    def test_retire_and_reconstruct_exact_independent_payloads_with_original_metadata(self):
        before = {name:(s[0]/'.tuldok/project.json').read_bytes() for name,s in self.sources.items()}
        subject.compact()
        report = json.loads(self.report.read_text())
        self.assertEqual(report['files_removed'], 2)
        self.assertEqual(report['payload_copies'], 0)
        for name, (source, _, _) in self.sources.items():
            self.assertFalse((source/'artifacts/sha256'/self.key).exists())
            self.assertEqual((source/'.tuldok/project.json').read_bytes(), before[name])
        subject.reconstruct('current')
        actual = self.sources['current'][0]/'artifacts/sha256'/self.key
        kept = self.keeper/'artifacts/sha256'/self.key
        self.assertEqual(actual.read_bytes(), self.payload)
        self.assertEqual(actual.stat().st_mode & 0o777, 0o400)
        self.assertNotEqual(actual.stat().st_ino, kept.stat().st_ino)
        self.assertEqual(actual.stat().st_nlink, 1)

    def test_changed_keeper_or_unique_source_refuses_before_deletion(self):
        kept = self.keeper/'artifacts/sha256'/self.key
        kept.chmod(0o600); kept.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Content-addressed'): subject.compact()
        self.assertFalse(self.plan.exists())
        self.assertTrue((self.sources['primary'][0]/'artifacts/sha256'/self.key).exists())
        kept.write_bytes(self.payload); kept.chmod(0o400)
        extra = self.sources['current'][0]/'artifacts/sha256'/sha(b'unique')
        extra.write_bytes(b'unique')
        with self.assertRaisesRegex(ValueError, 'Unique'): subject.compact()
        self.assertTrue((self.sources['primary'][0]/'artifacts/sha256'/self.key).exists())

    def test_reconstruction_remains_available_after_keeper_adds_notes_and_artifacts(self):
        subject.compact()
        header = self.keeper/'.tuldok/project.json'
        descriptor = json.loads(header.read_bytes()); descriptor['checkpoint'] = 'later-native-checkpoint'
        header.write_text(json.dumps(descriptor))
        extra = b'later source blob'; (self.keeper/'artifacts/sha256'/sha(extra)).write_bytes(extra)
        subject.reconstruct('primary')
        self.assertEqual((self.sources['primary'][0]/'artifacts/sha256'/self.key).read_bytes(), self.payload)
        # A new compaction pass must still reject a changed keeper checkpoint.
        with self.assertRaisesRegex(ValueError, 'manifest changed'): subject.compact(resume=True)

    def test_busy_native_root_lock_refuses(self):
        fd = os.open(self.keeper, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Isolate native flock exclusion from the separate reader-FD guard.
        with patch.object(subject, 'inactive', lambda _:None):
            with self.assertRaises(BlockingIOError): subject.compact()
        self.assertFalse(self.plan.exists())

    def test_interrupted_removal_resumes_from_original_map(self):
        original = Path.unlink; removed = []
        def interrupt(path, *args, **kwargs):
            if path.name == self.key:
                original(path, *args, **kwargs); removed.append(path)
                if len(removed) == 1: raise RuntimeError('simulated interruption after unlink')
            else: original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interruption'): subject.compact()
        map_sha = sha(self.plan.read_bytes())
        subject.compact(resume=True)
        self.assertEqual(sha(self.plan.read_bytes()), map_sha)
        self.assertEqual(json.loads(self.report.read_text())['files_removed'], 2)
        self.assertEqual((self.keeper/'artifacts/sha256'/self.key).read_bytes(), self.payload)

    def test_changed_mode_and_symlink_are_not_deleted(self):
        source = self.sources['current'][0]/'artifacts/sha256'/self.key
        source.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'mode'): subject.compact()
        self.assertTrue(source.exists()); source.chmod(0o400)
        source.unlink(); source.symlink_to(self.keeper/'artifacts/sha256'/self.key)
        with self.assertRaisesRegex(ValueError, 'Ordinary'): subject.compact()
        self.assertTrue(source.is_symlink()); self.assertFalse(self.plan.exists())


if __name__ == '__main__': unittest.main()
