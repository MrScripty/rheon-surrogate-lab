"""The general Project owner's exact source set imports in an isolated package."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import project


class ClippingCapture(unittest.TestCase):
    def test_owned_source_packages_include_clipping_dependency(self):
        for schema in ('rheon-water-residual-experiment-v2','rheon-water-residual-experiment-v10'):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                captured=Path(directory)/'capture'/'training';captured.mkdir(parents=True)
                sources=project.training_sources(dict(schema=schema))
                self.assertIn('gradient_clipping.py',sources)
                self.assertEqual(sources['gradient_clipping.py'],(project.HERE/'gradient_clipping.py').read_bytes())
                for name,raw in sources.items():(captured/name).write_bytes(raw)
                result=subprocess.run([sys.executable,'-B','-c',
                    'import train, gradient_clipping; assert train.clip_l2_ is gradient_clipping.clip_l2_'],
                    cwd=captured,env=os.environ.copy(),capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
