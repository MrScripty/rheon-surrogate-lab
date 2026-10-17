"""Diagnostic admission cannot accept an ordinary or unbounded fit."""
import copy
import unittest

from run_diagnostic import diagnostic_arguments


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(entrypoint='capture/gradient_screen.py',
            arguments=[{'literal':'--gradient-limit'},{'literal':'1500'},
                       {'literal':'--gradient-reference'},{'source':'capture/gradient-reference.json'}],
            sources=[dict(path='capture/gradient_screen.py',sha256='a'*64),
                     dict(path='capture/gradient-reference.json',sha256='b'*64)])

    def test_qualified_argument_contract(self):
        self.assertEqual(diagnostic_arguments(self.recipe),
                         (1500,'capture/gradient-reference.json','a'*64))

    def test_reject_ordinary_training(self):
        self.recipe['entrypoint']='capture/evidence.py'
        with self.assertRaisesRegex(ValueError,'Captured gradient diagnostic entrypoint required'):
            diagnostic_arguments(self.recipe)

    def test_reject_missing_cap(self):
        self.recipe['arguments']=self.recipe['arguments'][2:]
        with self.assertRaisesRegex(ValueError,'One explicit diagnostic argument required: --gradient-limit'):
            diagnostic_arguments(self.recipe)

    def test_reject_duplicate_cap(self):
        self.recipe['arguments']+=copy.deepcopy(self.recipe['arguments'][:2])
        with self.assertRaisesRegex(ValueError,'One explicit diagnostic argument required: --gradient-limit'):
            diagnostic_arguments(self.recipe)

    def test_reject_unbounded_cap(self):
        self.recipe['arguments'][1]['literal']='5000'
        with self.assertRaisesRegex(ValueError,'Diagnostic cap must be1..1500'):
            diagnostic_arguments(self.recipe)

    def test_reject_unbound_reference(self):
        self.recipe['sources']=self.recipe['sources'][:1]
        with self.assertRaisesRegex(ValueError,'Diagnostic entrypoint and reference must belong to captured sources'):
            diagnostic_arguments(self.recipe)


if __name__ == '__main__':
    unittest.main()
