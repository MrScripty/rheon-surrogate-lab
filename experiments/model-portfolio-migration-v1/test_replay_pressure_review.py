"""Negative checks for channel-group slicing and replay population boundaries."""
import copy
from pathlib import Path
import unittest

import numpy as np

from captured_replay_pressure_review import replay_order
from review_replay_pressure_series import check_slice
from review_water_learning_sequence import read

ROOT = Path(__file__).resolve().parents[2]


class ReplayPressureReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teacher = ROOT / 'runs/pressure-methods-v3/v3-native-study-07/restore/weights.npz'
        student = ROOT / 'runs/pressure-methods-v4/v4-student75-control-study-01/train/checkpoint-000.npz'
        with np.load(teacher, allow_pickle=False) as a:
            cls.teacher = {n:a[n].copy() for n in a.files}
        with np.load(student, allow_pickle=False) as a:
            cls.student = {n:a[n].copy() for n in a.files}
        cls.parents = read(ROOT / 'runs/pressure-methods-v4/v4-student75-control-study-01/train/splits.json')['parents']

    def test_original_student_is_exact_slice(self):
        check_slice(self.teacher, self.student, 3, 4)

    def test_wrong_decoder_concat_offset_is_rejected_at_tensor_check(self):
        child = dict(self.student)
        name = 'decoders.0.layers.0.weight'
        shape = child[name].shape
        # This preserves all shapes/dtypes but takes a contiguous channel prefix
        # instead of preserving the original coarse/skip concatenation groups.
        child[name] = self.teacher[name][:shape[0], :shape[1]].copy()
        with self.assertRaises(ValueError) as error:
            check_slice(self.teacher, child, 3, 4)
        self.assertEqual(str(error.exception), 'Independent channel slice differs from warm initialization')

    def test_small_sequence_retains_original_recorded_digest(self):
        import hashlib
        order, small = replay_order(self.parents, 2026101804)
        self.assertEqual(hashlib.sha256(small.tobytes()).hexdigest(),
                         '9d6532d2873d341b93ee2f41369f6c8ad91dc31bd5a3bc68fd0499104960054a')
        self.assertTrue(np.array_equal(order[np.arange(2400) % 3 != 2,0], small))

    def test_validation_parent_is_rejected_before_sampling(self):
        parents = copy.deepcopy(self.parents)
        parents[0]['split'] = 'validation'
        with self.assertRaises(ValueError) as error:
            replay_order(parents, 2026101804)
        self.assertEqual(str(error.exception), 'Replay requires training-only parents')


if __name__ == '__main__':
    unittest.main()
