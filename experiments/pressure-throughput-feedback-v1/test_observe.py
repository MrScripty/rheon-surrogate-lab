"""An in-progress source receipt cannot kill the external observation loop."""
from pathlib import Path
import tempfile
import unittest
import observe


class Observer(unittest.TestCase):
    def test_partial_receipt_remains_unreadable_until_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'lowband-regression-01/case-00';folder.mkdir(parents=True)
            source=folder/'episode.json';source.write_text('{')
            class Sink:
                def publish(self,*args,**kwargs):self.args=args;self.kwargs=kwargs
            sink=Sink();observe.once(root,root,sink)
            self.assertIn('unreadable: 1',sink.args[1]);self.assertEqual(sink.kwargs['completed'],0)
            source.write_text('{"status":"failed"}')
            observe.once(root,root,sink)
            self.assertIn('0 passed, 1 failed',sink.args[1]);self.assertEqual(sink.kwargs['completed'],1)
