"""Custody-audit fixtures use independent tiny portable file trees.

These are not native-admission fixtures: this reader's claim is literal custody.
Each negative starts with the passing tree and changes one audited property,
including updating the destination inventory to reach the intended check.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import verify_native_composition as subject


class CompositionCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = [self.root / name for name in ('primary', 'current', 'archive')]
        self.destination = self.root / 'published'
        self.report = self.root / 'audit.json'
        self.ids = ['1'*32, '2'*32, '3'*32]
        self.final_id = '4'*32
        self.run = dict(format='run_history_v1', origin_catalog_id='catalog',
                        run=dict(id='run-a', error=0.25),
                        admission_proof=dict(grant='grant-a', consumption='consumption-a', admission={'id': 'same'}))
        alternate = copy.deepcopy(self.run)
        alternate['admission_proof'].update(grant='grant-b', consumption='consumption-b')
        self.observation_id = subject.sha(subject.canonical(alternate))
        note = dict(format='project_file_v1', project_id=self.ids[0], kind='note',
                    path='notes/model.md', current=dict(revision=1, text='original model note\n'))
        note['history'] = [copy.deepcopy(note['current'])]
        self.shared_raw, self.unique_raw = b'shared original weights', b'archive original bytes'
        self.shared = subject.sha(self.shared_raw)
        self.unique = subject.sha(self.unique_raw)
        missing = subject.sha(b'explicit unavailable regenerable dependency')
        def artifact(identity, count):
            return dict(format='project_artifact_v1', sha256=identity, bytes=count)
        self.originals = [
            {('run_history', 'catalog.run-a'): self.run, ('project_file', 'note-a'): note,
             ('project_artifact', self.shared): artifact(self.shared, len(self.shared_raw))},
            {('run_history', 'catalog.run-b'): dict(format='run_history_v1', run=dict(id='run-b')),
             ('project_artifact', self.shared): artifact(self.shared, len(self.shared_raw))},
            {('run_history', 'catalog.run-a'): alternate,
             ('project_artifact', self.unique): artifact(self.unique, len(self.unique_raw)),
             ('project_artifact', missing): artifact(missing, 45)},
        ]
        self.inputs = []
        for index, (source, docs) in enumerate(zip(self.sources, self.originals)):
            descriptor = self.write_project(source, self.ids[index], docs)
            raw = (source/'.tuldok/project.json').read_bytes()
            self.inputs.append((source, descriptor['checkpoint'], subject.sha(raw)))
        self.write_payload(self.sources[1], self.shared, self.shared_raw)
        self.write_payload(self.sources[2], self.unique, self.unique_raw)
        self.write_note(self.sources[0], note['current']['text'])
        members = []
        for index, docs in enumerate(self.originals):
            records = {}
            for (family, identity), doc in docs.items():
                records.setdefault(family, {})[identity] = subject.sha(subject.canonical(
                    dict(family=family, id=identity, document=doc)))
            members.append(dict(project_id=self.ids[index], checkpoint=str(index+1)*64,
                                name='source-'+self.ids[index], records=records,
                                run_observations={'catalog.run-a': self.observation_id} if index == 2 else {}))
        args = dict(directory='published', name='Combined', request_key='same-logical-key')
        self.request = dict(arguments=args)
        final = {}
        for docs in self.originals:
            for key, value in docs.items():
                final.setdefault(key, copy.deepcopy(value))
        final[('run_observation', self.observation_id)] = dict(format='run_observation_v1',
            origin_catalog_id='catalog', run_id='run-a', sha256=self.observation_id, observation=alternate)
        final[('project_composition', self.final_id)] = dict(destination_project_id=self.final_id, sources=members, **args)
        mapping = dict(family='project_file', source_id='note-a', source_revision=1, destination_id='note-b')
        final[('project_fork', self.final_id)] = dict(destination_project_id=self.final_id,
            artifacts=dict(copied=[self.shared, self.unique], missing_regenerable=[missing]), mapping=[mapping], **args)
        clone = copy.deepcopy(note)
        clone.update(project_id=self.final_id)
        final[('project_file', 'note-b')] = clone
        self.final = final
        self.write_project(self.destination, self.final_id, final)
        self.write_payload(self.destination, self.shared, self.shared_raw)
        self.write_payload(self.destination, self.unique, self.unique_raw)
        self.write_note(self.destination, note['current']['text'])

    def write_project(self, path, identity, docs):
        items = []
        for (family, record_id), doc in sorted(docs.items()):
            relative = 'records/'+family+'/'+record_id+'.json'
            p = path/'.tuldok'/relative
            p.parent.mkdir(parents=True, exist_ok=True)
            raw = subject.canonical(doc)
            p.write_bytes(raw)
            p.chmod(0o600)
            items.append(dict(family=family, id=record_id, path=relative, bytes=len(raw), sha256=subject.sha(raw)))
        digit = str(self.ids.index(identity)+1) if identity in self.ids else '4'
        descriptor = dict(project_id=identity, name='source-'+identity, checkpoint=digit*64, inventory=items)
        (path/'.tuldok/project.json').write_bytes(subject.canonical(descriptor))
        return descriptor

    def write_payload(self, path, identity, raw):
        p = path/'artifacts/sha256'/identity
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        p.chmod(0o600)

    def write_note(self, path, text):
        p = path/'notes/model.md'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        p.chmod(0o600)

    def verify(self):
        return subject.verify(self.inputs, self.destination, self.request, self.report)

    def rewrite_final(self):
        self.write_project(self.destination, self.final_id, self.final)

    def test_all_sources_preserved_including_alternate_proof_and_payload_from_secondary(self):
        result = self.verify()
        self.assertEqual(result['original_memberships'], 8)
        self.assertEqual(result['alternate_Run_proofs'], 1)
        self.assertEqual(result['original_present_payloads_exact'], 2)
        self.assertEqual(result['explicit_missing_dependencies'], 1)
        self.assertEqual(result['editable_notes_exact'], 1)

    def test_changed_fact_with_updated_destination_inventory_is_rejected(self):
        self.final[('run_history', 'catalog.run-a')]['run']['error'] = 0.5
        self.rewrite_final()
        with self.assertRaisesRegex(ValueError, 'Original canonical fact changed or disappeared'):
            self.verify()

    def test_changed_alternate_proof_is_rejected(self):
        self.final[('run_observation', self.observation_id)]['observation']['admission_proof']['grant'] = 'lost original proof'
        self.rewrite_final()
        with self.assertRaisesRegex(ValueError, 'Alternate Run proof does not preserve the complete original document'):
            self.verify()

    def test_omitted_original_source_membership_is_rejected(self):
        del self.final[('project_composition', self.final_id)]['sources'][1]['records']['run_history']
        self.rewrite_final()
        with self.assertRaisesRegex(ValueError, 'Source membership omits or changes an original envelope'):
            self.verify()

    def test_missing_published_payload_is_rejected(self):
        (self.destination/'artifacts/sha256'/self.unique).unlink()
        with self.assertRaisesRegex(ValueError, 'Published payload file membership differs'):
            self.verify()

    def test_corrupt_published_payload_is_rejected(self):
        (self.destination/'artifacts/sha256'/self.unique).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Published payload is missing or changed'):
            self.verify()

    def test_payload_linked_to_any_source_is_rejected_even_when_first_source_is_independent(self):
        self.write_payload(self.sources[0], self.shared, self.shared_raw)
        after = self.destination/'artifacts/sha256'/self.shared
        after.unlink()
        os.link(self.sources[1]/'artifacts/sha256'/self.shared, after)
        with self.assertRaisesRegex(ValueError, 'Publication shares a source file inode'):
            self.verify()

    def test_changed_editable_note_revision_is_rejected(self):
        self.final[('project_file', 'note-b')]['history'][0]['text'] = 'lost original note'
        self.rewrite_final()
        with self.assertRaisesRegex(ValueError, 'Editable authored file lost exact original content/revision history'):
            self.verify()

    def test_source_manifest_not_matching_preservation_receipt_is_rejected(self):
        p = self.sources[2]/'.tuldok/project.json'
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Original source manifest differs from its preservation receipt'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
