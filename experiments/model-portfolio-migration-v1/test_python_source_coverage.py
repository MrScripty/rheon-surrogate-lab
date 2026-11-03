"""Synthetic runtime __file__ names must not masquerade as real lab paths."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest

from python_source_coverage import collect


def module(name,filename):
    value = ModuleType(name)
    value.__file__ = filename
    return value


class PythonSourceCoverage(unittest.TestCase):
    def test_known_torch_virtual_names_are_classified_not_hashed_as_lab_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = collect({'torch.classes':module('torch.classes','_classes.py'),
                'torch.ops':module('torch.ops','_ops.py')},root)
            self.assertEqual(result['lab_python'],[])
            self.assertEqual([r['module'] for r in result['runtime_virtual_modules']],['torch.classes','torch.ops'])

    def test_real_absolute_and_relative_lab_sources_remain_in_coverage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'actual.py').write_text('# Original source fixture.\n')
            result = collect({'one':module('one',str(root/'actual.py')),'two':module('two','actual.py')},root)
            self.assertEqual(result['lab_python'],['actual.py'])

    def test_missing_real_lab_source_still_rejects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError,'^Real imported lab Python source is unavailable: unknown.py$'):
                collect({'unbound':module('unbound','unknown.py')},root)

    def test_virtual_allowlist_cannot_hide_changed_metadata(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'^Pinned Torch virtual namespace metadata changed$'):
                collect({'torch.classes':module('torch.classes','unexpected.py')},Path(directory))


if __name__ == '__main__':
    unittest.main()
