"""No-execution declaration checks for the new scientific project owner."""
from pathlib import Path
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('groupnorm_project_under_test', Path(__file__).with_name('project.py'))
project = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = project
SPEC.loader.exec_module(project)


class ProjectDeclarationTests(unittest.TestCase):
    def test_owned_repository_root(self):
        self.assertEqual(project.LAB, Path(__file__).resolve().parents[3])
        self.assertTrue((project.LAB / '.git').exists())

    def test_complete_history_declared_in_both_modes(self):
        sources = {name: b'fixture' for name in ('model.py', 'train.py', 'original_model.py',
                   'original_train.py', 'original_data.py', 'config.json', 'runtime-manifest.json',
                   'source-witness.json', 'model-qualification.json', 'output-budget.json')}
        train = project.definition('train', sources, 'a'*64)
        restore = project.definition('restore', sources, 'a'*64)
        expected = {f'checkpoint-{step:03d}.npz' for step in range(0, 201, 20)}
        self.assertEqual(len(train['outputs']), 18)
        self.assertTrue(expected <= {row['path'] for row in train['outputs']})
        self.assertEqual(len(train['inputs']), 5)
        self.assertEqual(len(restore['inputs']), 22)
        self.assertEqual(len(restore['arguments']), 55)
        self.assertEqual(train['resources']['max_output_bytes'], 32*1024**2)
        self.assertEqual(train['entrypoint'], 'train.py')
        self.assertEqual({row['path'] for row in restore['outputs']}, {row['path'] for row in train['outputs']})

    def test_contracts_ignore_ambient_train_module(self):
        impostor = types.ModuleType('train')
        impostor.OUTPUTS = {'wrong.npz': ('checkpoint', 1)}
        with patch.dict(sys.modules, {'train': impostor}):
            owned = project.contracts()
            self.assertEqual(Path(owned.__file__).resolve(), Path(__file__).with_name('train.py').resolve())
            self.assertEqual(len(owned.OUTPUTS), 18)


if __name__ == '__main__':
    unittest.main()
