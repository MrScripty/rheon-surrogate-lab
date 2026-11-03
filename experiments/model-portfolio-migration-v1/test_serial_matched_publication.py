"""The next publisher must not run while the original import remains live."""
import copy
import os
import unittest

import finish_matched_portfolio_publication as owner


class SerialPublication(unittest.TestCase):
    def setUp(self):
        self.state = dict(status='four_readable_matched_GPU_imports_native_metadata_and_maxima_passed',
            active_request=None,imports=[dict(K=k) for k in range(1,5)],requests=[dict(status=200)],pid=2147483647)

    def test_complete_state_without_live_process_can_advance(self):
        owner.require_terminal_import(self.state)

    def test_live_process_rejects_even_with_completed_status(self):
        self.state['pid'] = os.getpid()
        with self.assertRaisesRegex(ValueError,'^Original reconciled import must be terminal'):
            owner.require_terminal_import(self.state)

    def test_missing_acknowledgement_pending_or_unknown_outcome_rejects(self):
        for changes in [dict(active_request='022-measurement_compare'),dict(imports=[dict(K=1)]),
                        dict(requests=[dict(status=401)]),dict(requests=[{}]),
                        dict(status='reconciled_matched_request_requires_inspection')]:
            with self.subTest(changes=changes):
                state = dict(copy.deepcopy(self.state),**changes)
                with self.assertRaisesRegex(ValueError,'^Original reconciled import must be terminal'):
                    owner.require_terminal_import(state)

    def test_process_identity_is_starttime_and_missing_pid_is_absent(self):
        self.assertTrue(owner.identity(os.getpid()).isdigit())
        self.assertIsNone(owner.identity(2147483647))


if __name__ == '__main__':
    unittest.main()
