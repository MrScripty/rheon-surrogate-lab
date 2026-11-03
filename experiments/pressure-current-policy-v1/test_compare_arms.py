"""Only the declared current-stage payload may differ in matched fitting inputs."""
import unittest
from compare_arms import matched_inputs,project
from pathlib import Path
import tempfile
import json
import hashlib


class MatchedInputTest(unittest.TestCase):
    def test_rewritten_witness_cannot_redefine_actual_training_settings(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);captured=root/'project';captured.mkdir();study=root/'study';study.mkdir()
            config=captured/'config.json';witness=captured/'source-witness.json';assembly=captured/'assembly.json'
            def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
            def snapshot(value):
                config.write_text(json.dumps(value));witness.write_text(json.dumps({'sources':{'config.json':sha(config)}}))
                assembly.write_text(json.dumps({'sources':[{'path':'project/'+p.name,'sha256':sha(p)}for p in (config,witness)]}))
            snapshot({'selected':'original'})
            (study/'intent.json').write_text(json.dumps({'assembly':str(assembly)}))
            receipt={'assembly_sha256':sha(assembly),'training_receipt':{'closure':{'source_witness_sha256':sha(witness),'experiment_sha256':sha(config)}}}
            self.assertEqual(project(study,root,receipt)[0],{'selected':'original'})
            snapshot({'selected':'rewritten'})
            receipt['assembly_sha256']=sha(assembly)  # Local assembly/source inventory remains self-consistent.
            with self.assertRaisesRegex(ValueError,'source witness differs from actual training closure'):
                project(study,root,receipt)

    def test_shared_input_changes_are_rejected_while_stage_is_allowed(self):
        original={key:{'sha256':key,'bytes':17}for key in ('dataset','dataset_extra','splits','direction_dataset','direction_manifest')}
        early={**original,'current_manifest':{'sha256':'shared','bytes':31},'current_dataset':{'sha256':'early','bytes':99}}
        late={**early,'current_dataset':{'sha256':'late','bytes':99}}
        matched_inputs(early,late,original)
        for key in ('dataset','splits','current_manifest'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'unselected input differs'):
                matched_inputs(early,{**late,key:{'sha256':'changed','bytes':17}},original)
        with self.assertRaisesRegex(ValueError,'original fitting input changed'):
            matched_inputs({**early,'splits':{'sha256':'samewrong','bytes':17}},{**late,'splits':{'sha256':'samewrong','bytes':17}},original)

if __name__=='__main__':unittest.main()
