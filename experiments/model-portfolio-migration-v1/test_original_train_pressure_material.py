"""Metadata boundaries reject wrong training state before model/Torch imports.

Tiny weight bytes are an opaque identity fixture; this does not qualify NPZ
decoding, CUDA restoration, architecture parity or numerical predictions.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import original_train_pressure_cuda as loader


class OriginalTrainMaterial(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project,self.outputs = self.root/'project',self.root/'outputs'
        self.project.mkdir();self.outputs.mkdir()
        (self.project/'model.py').write_text('# Captured source fixture only.\n')
        (self.project/'config.json').write_text('{}')
        witness = dict(sources={'model.py':self.spec(self.project/'model.py')['sha256']})
        (self.project/'source-witness.json').write_text(json.dumps(witness))
        (self.outputs/'weights.npz').write_bytes(b'opaque identity fixture; never decoded')
        (self.outputs/'normalization.json').write_text('{}')
        history = dict(checkpoints={'checkpoint-001.npz':dict(update=1,inventory={'weight':{'dtype':'<f4'}},
            sha256=self.spec(self.outputs/'weights.npz')['sha256'])})
        (self.outputs/'history.json').write_text(json.dumps(history))
        self.receipt = dict(schema='rheon-water-learning-training-v1',mode='train',status='complete',optimizer_updates=1,
            architecture={'fixture':'metadata-only'},outputs={},closure={
                'experiment_sha256':self.spec(self.project/'config.json')['sha256'],
                'source_witness_sha256':self.spec(self.project/'source-witness.json')['sha256'],
                'normalization_sha256':self.spec(self.outputs/'normalization.json')['sha256']})
        self.entry = dict(fit_item_id='fit-fixture',checkpoint_origin='original_train_output',project='project',artifacts='outputs',
            original_optimizer_updates=1,captured_architecture=self.receipt['architecture'],
            original_configuration=self.spec(self.project/'config.json'),source_witness=self.spec(self.project/'source-witness.json'),
            sources=[self.spec(self.project/'model.py')])
        self.rebind_outputs_and_receipt()
        self.material = dict(root=str(self.root),models=[self.entry])

    def spec(self,path):
        raw = path.read_bytes()
        return dict(path=str(path.relative_to(self.root)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

    def rebind_outputs_and_receipt(self):
        self.entry['outputs'] = {name:self.spec(self.outputs/name) for name in ('weights.npz','history.json','normalization.json')}
        self.receipt['outputs'] = {name:{k:v for k,v in spec.items() if k in ('sha256','bytes')} for name,spec in self.entry['outputs'].items()}
        p = self.outputs/'receipt.json'
        p.write_text(json.dumps(self.receipt))
        self.entry['original_receipt'] = self.spec(p)

    def verify(self):
        return loader.verify_entry(self.material,'fit-fixture',root=self.root)

    def test_exact_training_metadata_selected_without_torch_or_array_import(self):
        before = set(sys.modules)
        verified = self.verify()
        self.assertEqual(verified.receipt['mode'],'train')
        self.assertEqual(verified.inventory,{'weight':{'dtype':'<f4'}})
        self.assertFalse({'torch','numpy'} & (set(sys.modules)-before))

    def test_fabricated_restore_origin_rejects(self):
        self.entry['checkpoint_origin'] = 'cpu_restore'
        with self.assertRaisesRegex(ValueError,'^Exactly one original training-output fit required$'):
            self.verify()

    def test_restore_receipt_with_valid_hash_still_rejects(self):
        self.receipt.update(mode='restore',optimizer_updates=0)
        self.entry['original_optimizer_updates'] = 0
        self.rebind_outputs_and_receipt()
        with self.assertRaisesRegex(ValueError,'^Original completed training receipt required'):
            self.verify()

    def test_changed_captured_source_rejects(self):
        (self.project/'model.py').write_text('# Corrupted captured source fixture.\n')
        with self.assertRaisesRegex(ValueError,'^Original pressure material bytes changed$'):
            self.verify()

    def test_ambiguous_final_inventory_rejects_with_valid_output_hashes(self):
        p = self.outputs/'history.json'
        history = json.loads(p.read_text())
        history['checkpoints']['other.npz'] = copy.deepcopy(history['checkpoints']['checkpoint-001.npz'])
        p.write_text(json.dumps(history))
        self.rebind_outputs_and_receipt()
        with self.assertRaisesRegex(ValueError,'^Selected original final tensor inventory is unavailable or ambiguous$'):
            self.verify()

    def test_cpu_environment_rejects_before_metadata_or_torch(self):
        with patch.dict(loader.os.environ,{'TULDOK_TORCH_DEVICE':'cpu'}), \
                patch.object(loader,'verify_entry',side_effect=AssertionError('Must reject before loading material')):
            with self.assertRaisesRegex(ValueError,'^Explicit CUDA environment required'):
                with loader.restore(self.material,'fit-fixture'):
                    self.fail('CPU restore context must be unreachable')


if __name__ == '__main__':
    unittest.main()
