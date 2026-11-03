"""Publication barriers must prevent replay and overlapping native owners."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import adopt_matched_generator_outcome_notes as owner


class MatchedOutcomeOwner(unittest.TestCase):
    def test_existing_status_rejects_before_qualification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            before,status = root/'before.json',root/'status.json'
            before.write_text('{}')
            status.write_text('{}')
            with patch.object(owner,'BEFORE',before),patch.object(owner,'STATUS',status),patch.object(owner,'REQUEST',root/'request.json'), \
                    patch.object(owner,'qualified',side_effect=AssertionError('Must reject before qualification/authentication')):
                with self.assertRaisesRegex(ValueError,'^Original before-custody required; never replay a note-adoption owner$'):
                    owner.adopt()

    def test_retained_request_rejects_without_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            before,request = root/'before.json',root/'request.json'
            before.write_text('{}')
            request.write_text('{}')
            with patch.object(owner,'BEFORE',before),patch.object(owner,'STATUS',root/'status.json'),patch.object(owner,'REQUEST',request), \
                    patch.object(owner,'qualified',side_effect=AssertionError('Must reject before qualification/authentication')):
                with self.assertRaisesRegex(ValueError,'^Original before-custody required; never replay a note-adoption owner$'):
                    owner.adopt()

    def test_inflight_measurement_owner_rejects_before_note_access(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(path=str((owner.RUN/'native-matched-generator-GPU-import-01/status.json').relative_to(owner.ROOT)),sha256='bound')
            def read(path):
                if path == owner.IMPORT_CUSTODY:
                    return dict(status='four_readable_matched_GPU_tables_native_custody_exact',original_owner=spec)
                self.assertEqual(path,owner.RUN/'native-matched-generator-GPU-import-01/status.json')
                return dict(status='original_native_matched_request_in_flight',active_request='003-measurement_import',pid=1)
            with patch.object(owner,'JOB',root/'not-started'),patch.object(owner,'REPORT',root/'report.json'), \
                    patch.object(owner,'qualified',return_value=({},{})),patch.object(owner,'read',side_effect=read), \
                    patch.object(owner,'binding',return_value=spec), \
                    patch.object(owner,'header',side_effect=AssertionError('Must reject before accessing notes')):
                with self.assertRaisesRegex(ValueError,'^Original matched-table imports and after-custody must finish first$'):
                    owner.before()
            self.assertFalse((root/'not-started').exists())


if __name__ == '__main__':
    unittest.main()
