"""Published outcome changes do not authorize changed fit/checkpoint facts."""
import copy
import unittest

from record_original_pressure_timing_coverage import unchanged_fit_facts


class IndexTransitionTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(project='p',aliases=[dict(name='C1',fit='a')],sources={'original':'receipt'},fits=[
            dict(item_id='a',task='full_state_generator',checkpoints=['weight-a'],card_sha256='old'),
            dict(item_id='b',task='pressure_proposal',checkpoints=['weight-b'],card_sha256='unchanged')])
        self.after = copy.deepcopy(self.before)
        self.after['fits'][0].update(card_sha256='new',matched_GPU_benchmark={'K':4})
        self.after['sources']['matched_generator_GPU_benchmark'] = 'accepted'

    def test_exact_index_bytes_change_for_legitimate_outcome_publication(self):
        self.assertNotEqual(self.before,self.after)
        self.assertEqual(unchanged_fit_facts(self.before,self.after),1)

    def test_checkpoint_or_alias_mutation_still_rejected(self):
        changed = copy.deepcopy(self.after)
        changed['fits'][0]['checkpoints'] = ['different-weight']
        with self.assertRaisesRegex(ValueError,'checkpoint facts'):
            unchanged_fit_facts(self.before,changed)
        changed = copy.deepcopy(self.after)
        changed['aliases'][0]['fit'] = 'b'
        with self.assertRaisesRegex(ValueError,'aliases'):
            unchanged_fit_facts(self.before,changed)

    def test_non_generator_and_unrelated_source_change_rejected(self):
        changed = copy.deepcopy(self.after)
        changed['fits'][1]['card_sha256'] = 'unexpected'
        with self.assertRaisesRegex(ValueError,'outcome revision'):
            unchanged_fit_facts(self.before,changed)
        changed = copy.deepcopy(self.after)
        changed['sources']['original'] = 'unexpected'
        with self.assertRaisesRegex(ValueError,'source revision'):
            unchanged_fit_facts(self.before,changed)


if __name__ == '__main__':
    unittest.main()
