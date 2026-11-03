"""Composed inventories retain foreign file documents in the current namespace."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import adopt_matched_generator_outcome_notes_02 as fixed

owner = fixed.original


class ComposedNoteSelection(unittest.TestCase):
    def test_foreign_same_path_does_not_hide_current_owner_note(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.tuldok').mkdir()
            entries = []
            for name,project in [('foreign','old-project'),('owned','current-project')]:
                doc = dict(project_id=project,path='notes/shared.md',current=dict(text=name))
                body = owner.encode(doc)
                (root/'.tuldok'/name).write_bytes(body)
                entries.append(dict(family='project_file',namespace='current-project',id=name,path=name,
                    bytes=len(body),sha256=owner.digest(body)))
            with patch.object(owner,'PROJECT',root):
                result = fixed.target_records(dict(project_id='current-project',inventory=entries))
            self.assertEqual(set(result),{'notes/shared.md'})
            self.assertEqual(result['notes/shared.md'][1]['current']['text'],'owned')

    def test_duplicate_current_owner_path_still_rejects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.tuldok').mkdir()
            body = owner.encode(dict(project_id='current-project',path='notes/shared.md'))
            (root/'.tuldok/one').write_bytes(body)
            entry = dict(family='project_file',namespace='current-project',id='one',path='one',bytes=len(body),sha256=owner.digest(body))
            with patch.object(owner,'PROJECT',root):
                with self.assertRaisesRegex(ValueError,'^Current note ownership/path differs$'):
                    fixed.target_records(dict(project_id='current-project',inventory=[entry,entry]))

    def test_corrupt_foreign_record_still_rejects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.tuldok').mkdir()
            body = owner.encode(dict(project_id='old-project',path='notes/shared.md'))
            (root/'.tuldok/one').write_bytes(body.replace(b'old-project',b'bad-project'))
            entry = dict(family='project_file',namespace='current-project',id='one',path='one',bytes=len(body),sha256=owner.digest(body))
            with patch.object(owner,'PROJECT',root):
                with self.assertRaisesRegex(ValueError,'^Current canonical note identity differs$'):
                    fixed.target_records(dict(project_id='current-project',inventory=[entry]))


if __name__ == '__main__':
    unittest.main()
