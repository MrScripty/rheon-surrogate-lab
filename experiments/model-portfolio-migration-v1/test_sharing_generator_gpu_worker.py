"""A mislabeled architecture/horizon must fail before importing the model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from sharing_generator_gpu_worker import run

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'


class CapturedConstructorIdentityTests(unittest.TestCase):
    def test_wrong_selected_architecture_or_horizon_is_rejected(self):
        protocol = json.loads((RUN/'sharing-generator-GPU-benchmark-protocol-01.json').read_text())
        prepared = json.loads((ROOT/protocol['cases']['path']).read_text())
        original = prepared['models'][0]
        for field,value in [('architecture','reuse'),('K',3)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=RUN,prefix='constructor-negative-') as directory:
                root = Path(directory)
                cases = copy.deepcopy(prepared)
                cases['models'][0][field] = value
                case_path = root/'cases.json'
                case_path.write_text(json.dumps(cases))
                selected_protocol = dict(protocol,cases={'path':str(case_path.relative_to(ROOT))})
                protocol_path = root/'protocol.json'
                protocol_path.write_text(json.dumps(selected_protocol))
                output = root/'uncreated-result.json'
                with self.assertRaisesRegex(ValueError,'^Reviewed sharing architecture required$'):
                    run(protocol_path,original['identifier'],0,output)
                self.assertFalse(output.exists())
                self.assertNotIn('torch',sys.modules)
                self.assertNotIn('bundle_model_v5',sys.modules)
                self.assertNotIn('bundle_model_sharing',sys.modules)


if __name__ == '__main__':
    unittest.main()
