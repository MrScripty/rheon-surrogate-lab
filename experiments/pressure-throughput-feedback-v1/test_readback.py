"""A partially downloaded terminal Run is recoverable without new execution."""
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import run_study


class Readback(unittest.TestCase):
    def test_partial_collection_reuses_exact_bytes_and_refuses_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory);payload=b'completed pressure weights'
            row=dict(name='weights.npz',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            run={'id':'a'*32};(target/row['name']).write_bytes(payload)
            owner=types.SimpleNamespace(download=lambda *_:self.fail('existing bytes must not download again'))
            with patch.dict(sys.modules,{'project_cli':owner}):
                result=run_study.collect_artifact(None,run,row,target)
                self.assertEqual(result['verification'],'existing_whole_sha256_verified')
                (target/row['name']).write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError,'existing artifact differs'):
                    run_study.collect_artifact(None,run,row,target)
                self.assertEqual((target/row['name']).read_bytes(),b'wrong')

    def test_interruption_after_first_download_does_not_repeat_it(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory);calls=[];failed=False
            rows=[dict(name=name,bytes=len(name),sha256=hashlib.sha256(name.encode()).hexdigest())
                  for name in ('weights.npz','predictions.npz')]
            def download(_reader,_run,name,sha,destination):
                nonlocal failed
                calls.append(name)
                if name=='predictions.npz' and not failed:
                    failed=True;raise ConnectionError('readback interrupted')
                destination.write_bytes(name.encode());return dict(name=name,sha256=sha)
            with patch.dict(sys.modules,{'project_cli':types.SimpleNamespace(download=download)}):
                with self.assertRaises(ConnectionError):
                    for row in rows:run_study.collect_artifact(None,{'id':'a'*32},row,target)
                for row in rows:run_study.collect_artifact(None,{'id':'a'*32},row,target)
            self.assertEqual(calls,['weights.npz','predictions.npz','predictions.npz'])
            self.assertTrue(all((target/r['name']).read_bytes()==r['name'].encode() for r in rows))
