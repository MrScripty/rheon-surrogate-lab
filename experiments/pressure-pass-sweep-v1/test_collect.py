"""Running supervision has no terminal clocks or reaping claim yet."""
import json
from pathlib import Path
import tempfile
import unittest
from collect import collect


class ProgressTests(unittest.TestCase):
    def test_running_campaign_keeps_missing_measurements_pending(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            model=dict(id='C01',name='Pending fixture',tags=['frontier'],weights_sha256='0'*64,
                       original_worst_error=1,original_inference_median_ms=1)
            material=dict(root=str(root),models=[model],config=dict(passes=[1],scene={}),chart_sha256='0'*64)
            path=root/'material.json';path.write_text(json.dumps(material))
            (root/'execution.json').write_text(json.dumps(dict(status='running',maximum_workers=1)))
            result=collect(path,root/'chart')
            self.assertEqual(result['status'],'running')
            self.assertEqual(result['points'],[])
            self.assertEqual(result['trajectories'],[])
            self.assertNotIn('all_owned_children_reaped',result['execution'])


if __name__=='__main__':unittest.main()
