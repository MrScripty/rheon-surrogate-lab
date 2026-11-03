"""Only an explicit, terminal, unadmitted401 permits this continuation."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import reconcile_matched_generator_import as owner


class Reconciliation(unittest.TestCase):
    def setUp(self):
        self.state = dict(status='original_matched_request_requires_reconciliation',pid=2147483647,
            imports=[dict(K=1),dict(K=2)],last_acknowledged_checkpoint='accepted',
            requests=[dict(status=200),dict(status=200),dict(status=200),
                dict(status=401,request='original.request.json',response='original.response.json')])
        self.request = dict(operation='project_files.measurement_import',arguments=dict(
            reference_id='K3-complete-GPU',expected_checkpoint='accepted',request_key='keep-exact-key'))
        self.response = dict(status=401,result=dict(code='unauthenticated',error='A valid credential is required.'))

    def invoke(self):
        def read(path):
            if path == owner.PRIOR/'original.request.json':
                return self.request
            self.assertEqual(path,owner.PRIOR/'original.response.json')
            return self.response
        with patch.object(owner,'read',side_effect=read):
            return owner.rejected_request(self.state)

    def test_exact_original_rejected_body_is_reused(self):
        expected = copy.deepcopy(self.request)
        self.assertIs(self.invoke(),self.request)
        self.assertEqual(self.request,expected)

    def test_unknown_live_and_acknowledged_outcomes_reject(self):
        for status,pid,last_status in [('original_native_matched_request_in_flight',2147483647,401),
                                      ('original_matched_request_requires_reconciliation',1,401),
                                      ('original_matched_request_requires_reconciliation',2147483647,200),
                                      ('original_matched_request_requires_reconciliation',2147483647,None)]:
            with self.subTest(status=status,pid=pid,last_status=last_status):
                self.state.update(status=status,pid=pid)
                self.state['requests'][-1]['status'] = last_status
                with self.assertRaisesRegex(ValueError,'^Terminal original owner'):
                    self.invoke()

    def test_ambiguous401_or_changed_request_reject(self):
        self.response['result']['code'] = 'unknown'
        with self.assertRaisesRegex(ValueError,'^Only the explicit unauthenticated'):
            self.invoke()
        self.response['result']['code'] = 'unauthenticated'
        self.request['arguments']['expected_checkpoint'] = 'changed'
        with self.assertRaisesRegex(ValueError,'^Rejected request must be the original K3'):
            self.invoke()

    def test_existing_reconciliation_status_rejects_before_qualification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proof,status = root/'before.json',root/'status.json'
            proof.write_text('{}')
            status.write_text('{}')
            with patch.object(owner,'RECONCILIATION',proof),patch.object(owner,'STATUS',status),patch.object(owner,'JOB',root), \
                    patch.object(owner.original,'qualified',side_effect=AssertionError('Must reject before authentication')):
                with self.assertRaisesRegex(ValueError,'^Fresh qualified reconciliation owner required'):
                    owner.run()


if __name__ == '__main__':
    unittest.main()
