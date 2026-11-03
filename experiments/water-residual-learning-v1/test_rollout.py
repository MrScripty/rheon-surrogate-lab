"""Real process ownership on failed exit and deadline, isolated temporary roots."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import rollout


class NativeOwnership(unittest.TestCase):
    def helper(self,root,body):
        path=root/'native';path.write_text('#!'+sys.executable+'\n'+body);path.chmod(0o700);return path

    def test_failed_native_exit_retains_failure_and_reaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binary=self.helper(root,'import sys\nsys.exit(3)\n')
            with self.assertRaisesRegex(RuntimeError,'did not complete'):
                rollout.run(binary,{},root/'record',control=True)
            result=json.loads((root/'record/receipt.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertTrue(result['native_process']['reaped'])
            self.assertEqual(result['native_process']['returncode'],3)

    def test_rejected_attempt_retains_raw_evidence_without_advancing_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            attempt={'kind':'diagnostic_attempt','attempt':{'raw_candidate_pressure':[0.,1.],
                'history_transition':'NotAdvanced','native_fallback':False},'stop':'Rejected'}
            complete={'kind':'complete','ai_steps':1,'qualified_ai_steps':0,'hard_stop':True,
                'native_fallback':False,'native_pressure_refinement':False}
            binary=self.helper(root,'import json,sys\njson.loads(sys.stdin.readline())\n'
                +'print('+repr(json.dumps(attempt))+',flush=True)\n'
                +'print('+repr(json.dumps(complete))+',flush=True)\n')
            result=rollout.run(binary,{},root/'record',control=True)
            self.assertEqual(result['rejected_attempts'],1)
            self.assertEqual(result['frames'],0)
            self.assertTrue(result['result']['hard_stop'])
            self.assertFalse(result['native_fallback'])
            self.assertTrue(result['native_process']['reaped'])
            self.assertEqual(json.loads((root/'record/rejected-attempts.jsonl').read_text()),attempt)

    def test_adapter_error_reaches_native_attempt_before_failure_and_reaping(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=root/'adapter.json';manifest.write_text('{}')
            (root/'model_adapter.py').write_text(
                'from types import SimpleNamespace as S\n'
                'def load(path):\n'
                ' return S(torch=S(cuda=S(is_initialized=lambda:False)),model=None,network=S(register_forward_hook=lambda hook:S(remove=lambda:None)),normalization={},'
                'policy={"max_network_calls":32},manifest={"inputs":{"base":"previous_pressure"},"output":{"role":"full_pressure"}})\n'
                'def predict(value,state,trace): raise ValueError("unsupported fixture context")\n'
                'def failure(error,request_id): return {"kind":"error","id":request_id,"code":"unsupported_context","message":str(error)}\n')
            binary=self.helper(root,'import json,sys\njson.loads(sys.stdin.readline())\n'
                +'print(json.dumps({"kind":"predict","id":16}),flush=True)\n'
                +'reply=json.loads(sys.stdin.readline())\n'
                +'assert reply["kind"]=="error" and reply["id"]==16 and reply["code"]=="unsupported_context"\n'
                +'print(json.dumps({"kind":"diagnostic_attempt","attempt":{"raw_reply":reply},"stop":"ProviderFailure"}),flush=True)\n'
                +'print(json.dumps({"kind":"complete","ai_steps":1,"qualified_ai_steps":0,"hard_stop":True}),flush=True)\n')
            with self.assertRaisesRegex(RuntimeError,'current adapter failed'):
                rollout.run(binary,{},root/'record',adapter_manifest=manifest)
            result=json.loads((root/'record/receipt.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['rejected_attempts'],1)
            self.assertTrue(result['native_process']['reaped']);self.assertEqual(result['native_process']['returncode'],0)
            raw=json.loads((root/'record/rejected-attempts.jsonl').read_text())['attempt']['raw_reply']
            self.assertEqual(raw['code'],'unsupported_context')

    def test_deadline_terminates_and_reaps_owned_native(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binary=self.helper(root,'import time\ntime.sleep(30)\n')
            with self.assertRaisesRegex(TimeoutError,'deadline'):
                rollout.run(binary,{},root/'record',control=True,timeout=.05)
            result=json.loads((root/'record/receipt.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertTrue(result['native_process']['terminated'])
            self.assertTrue(result['native_process']['reaped'])
            self.assertIsNotNone(result['native_process']['returncode'])


if __name__=='__main__':unittest.main()
