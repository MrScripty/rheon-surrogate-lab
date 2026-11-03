"""A retained note request or existing continuation must reject another owner."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import finish_matched_note_publication as owner


class NoteContinuation(unittest.TestCase):
    def test_existing_continuation_rejects_before_any_evidence_or_authentication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(owner,'JOB',root),patch.object(owner,'read',side_effect=AssertionError('Must reject first')):
                with self.assertRaisesRegex(ValueError,'^Fresh continuation from accepted before-custody required'):
                    owner.run()

    def test_existing_native_request_rejects_if_orchestrator_is_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root/'notes'
            notes.mkdir()
            (notes/'adopt.request.json').write_text('{}')
            with patch.object(owner,'JOB',root/'missing'),patch.object(owner,'NOTE_JOB',notes), \
                    patch.object(owner,'read',side_effect=AssertionError('Must reject first')):
                with self.assertRaisesRegex(ValueError,'^Fresh continuation from accepted before-custody required'):
                    owner.run()


if __name__ == '__main__':
    unittest.main()
