"""Reject self-consistent replacement training data and escaping destinations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import prepare


class BoundaryTest(unittest.TestCase):
    def test_self_consistent_replacement_cannot_change_checkpoint_training_population(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);project=root/'project';project.mkdir();data=root/'splits.json';data.write_bytes(b'original training identities')
            expected={'bytes':data.stat().st_size,'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            (project/'source-witness.json').write_text(json.dumps({'inputs':{'splits':expected}}))
            document={'data':[{'id':'splits','path':'splits.json','sha256':expected['sha256']}]}
            (project/'project.json').write_text(json.dumps(document))
            raw,_=prepare.bound_inputs(root,{'project':'project'});self.assertEqual(raw['splits'],data.read_bytes())
            data.write_bytes(b'replacement identities');document['data'][0]['sha256']=hashlib.sha256(data.read_bytes()).hexdigest()
            (project/'project.json').write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,'dataset differs from actual checkpoint captured input: splits'):
                prepare.bound_inputs(root,{'project':'project'})

    def test_memory_observation_failure_reaches_owner_once(self):
        from threading import Event
        state={'minimum_available_bytes':100};notifications=[]
        def unavailable():raise OSError('fixture proc observation unavailable')
        prepare.monitor_memory(Event(),state,10,lambda:notifications.append('stop'),unavailable)
        self.assertEqual(notifications,['stop'])
        self.assertEqual(state['observation_error'],'OSError: fixture proc observation unavailable')

    def test_physical_containment_rejects_symlink_and_traversal_without_creating(self):
        with tempfile.TemporaryDirectory()as folder:
            parent=Path(folder);root=parent/'root';root.mkdir();outside=parent/'outside';outside.mkdir();(root/'alias').symlink_to(outside,target_is_directory=True)
            for path in (root/'alias/result',root/'../outside/result'):
                with self.assertRaisesRegex(ValueError,'fresh rooted output required'):prepare.destination(root,path)
            self.assertFalse((outside/'result').exists())
            self.assertEqual(prepare.destination(root,root/'new'),root/'new')

if __name__=='__main__':unittest.main()
