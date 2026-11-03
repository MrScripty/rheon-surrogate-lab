"""Exercise selection and aggregation detection using original recorded scores."""
import copy
import unittest

from review_guided_temporal_components import BASE, inspect_scores, read


class GuidedScoreReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = read(BASE / 'guided-pressure-managed-pair-01/train/history.json')
        cls.windows = [(r['identifier'], r['start_fine_step'])
                       for r in cls.history['initial_full_score']['rows']]

    def test_original_pressure_selection_prefers_evaluated_probe(self):
        self.assertEqual(inspect_scores(self.history, self.windows, .1), ('probe-3', 'candidate-4'))

    def test_wrong_selected_label_reaches_selection_diagnostic(self):
        history = copy.deepcopy(self.history)
        history['selected_worst_trial'] = 'candidate-4'
        with self.assertRaises(ValueError) as error:
            inspect_scores(history, self.windows, .1)
        self.assertEqual(str(error.exception), 'Independent recorded-score checkpoint selection differs')

    def test_changed_recorded_window_reaches_aggregation_diagnostic(self):
        history = copy.deepcopy(self.history)
        history['trials'][0]['score']['rows'][0]['objective'] += 1.
        with self.assertRaises(ValueError) as error:
            inspect_scores(history, self.windows, .1)
        self.assertEqual(str(error.exception), 'Independent balanced objective aggregation differs')


if __name__ == '__main__':
    unittest.main()
