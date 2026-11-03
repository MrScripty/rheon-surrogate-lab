"""A claimed parity flag cannot hide different actual CPU artifact bytes."""
import json
from pathlib import Path
import tempfile
import unittest

import compare_training


class ActualTrainingEvidence(unittest.TestCase):
    def test_different_cpu_bytes_rejected_even_with_accurate_download_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['weights.npz','normalization.json','predictions.npz','history.json','checkpoint-5000.npz']
            for mode in ('train','restore'):
                (root/mode).mkdir()
                for name in names:
                    (root/mode/name).write_bytes(b'same captured bytes')
                (root/mode/'history.json').write_text(json.dumps({'training_started_unix':10.,'training_finished_unix':12.}))
            (root/'train/checkpoint-000.npz').write_bytes(b'initialization')
            trained = {'mode':'train','execution':{'device':'cuda:0','peak_allocated_bytes':1},
                       'optimizer_updates':5000,'closure':{},'elapsed_s':3.}
            restored = {'mode':'restore','execution':{'device':'cpu','cuda_initialized':False},
                        'optimizer_updates':0,'closure':{}}
            receipt = {'status':'completed','artifact_parity':dict.fromkeys(names,True),
                       'training_receipt':trained,'restore_receipt':restored,'optimizer_updates':5000,
                       'weights_sha256':compare_training.digest(root/'train/weights.npz'),
                       'training_run_id':'fit','restore_run_id':'restore','band_feedback':{'weight':.01},
                       'elapsed_s':4.}
            def write_receipt():
                receipt['downloads'] = [dict(output=str(path),bytes=path.stat().st_size,
                                            sha256=compare_training.digest(path))
                                       for path in root.glob('*/*')]
                (root/'receipt.json').write_text(json.dumps(receipt))
            write_receipt()
            compare_training.measured(root,root)  # Every unrelated prerequisite is valid.
            for row in receipt['downloads']:
                row['output']=str(Path(row['output']).relative_to(root))
            (root/'receipt.json').write_text(json.dumps(receipt))
            # Current cwd is unrelated to the temporary artifact root. Historical
            # root-relative download paths must still verify exactly.
            compare_training.measured(root,root)
            (root/'restore/weights.npz').write_bytes(b'different restored weights')
            write_receipt()  # Download records truthfully describe the changed bytes.
            with self.assertRaisesRegex(ValueError,'^actual CPU artifact bytes differ from CUDA output$'):
                compare_training.measured(root,root)
