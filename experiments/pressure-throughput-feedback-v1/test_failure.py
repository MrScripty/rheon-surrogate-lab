"""A receipt alone cannot turn infrastructure failure into a model result."""
import json
from pathlib import Path
import tempfile
import unittest
from failure import classify


class Classification(unittest.TestCase):
    def test_only_drained_explicit_numerical_rejection_is_retainable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            receipt=dict(status='completed',error=None,native_process=dict(reaped=True,returncode=0,terminated=False),
                         result=dict(hard_stop=True),rejected_attempts=1)
            for name in ('native','native-control'):
                (root/name).mkdir()
                value={**receipt,'result':dict(hard_stop=False,reference_control=True)} if name=='native-control' else receipt
                (root/name/'receipt.json').write_text(json.dumps(value))
            (root/'native/rejected-attempts.jsonl').write_text(json.dumps(dict(outcome=dict(kind='candidate',qualified=False)))+'\n')
            numerical=ValueError('sparse capture schedule mismatch')
            self.assertEqual(classify(numerical,root)['failure_class'],'scientific_rejection')
            for error in (MemoryError('reserve'),TimeoutError('native deadline'),KeyboardInterrupt(),ValueError('source changed')):
                self.assertEqual(classify(error,root)['failure_class'],'infrastructure')
            receipt['native_process']['reaped']=False
            (root/'native/receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(classify(numerical,root)['failure_class'],'infrastructure')
