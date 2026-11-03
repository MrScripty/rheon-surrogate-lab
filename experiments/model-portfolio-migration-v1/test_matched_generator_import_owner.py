"""Replay barriers must reject before qualification or authentication."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import import_matched_generator_measurements as owner


class MatchedImportOwner(unittest.TestCase):
    def barrier(self,existing):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            before,status = job/'before.json',job/'status.json'
            before.write_text('{}')
            existing(job,status)
            with patch.object(owner,'JOB',job),patch.object(owner,'BEFORE',before),patch.object(owner,'STATUS',status), \
                    patch.object(owner,'qualified',side_effect=AssertionError('Must not reach qualification/authentication')):
                with self.assertRaisesRegex(ValueError,'^Original before-custody required; never replay an import owner$'):
                    owner.adopt()

    def test_existing_status_rejects_replay(self):
        self.barrier(lambda job,status:status.write_text('{}'))

    def test_retained_original_request_rejects_replay_without_status(self):
        self.barrier(lambda job,status:(job/'001-source_register.request.json').write_text('{}'))


if __name__ == '__main__':
    unittest.main()
