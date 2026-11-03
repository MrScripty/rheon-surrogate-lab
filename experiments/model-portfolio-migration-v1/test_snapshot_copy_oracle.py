"""Negatives for the independent snapshot value/ownership proof."""
import copy
import math
import unittest

from benchmark_snapshot_copy import value_proof


class SnapshotProofTests(unittest.TestCase):
    def test_json_equal_scalars_with_different_types_have_distinct_proofs(self):
        self.assertEqual(len({value_proof(v)[0] for v in [False, 0, 0.0, None]}), 4)

    def test_signed_zero_and_adjacent_float_bits_remain_distinct(self):
        values = [-0.0, 0.0, 5e-324, -5e-324, 1.0, math.nextafter(1.0, math.inf)]
        self.assertEqual(len({value_proof(v)[0] for v in values}), len(values))

    def test_alias_into_a_different_source_branch_is_rejected(self):
        source = dict(left=[dict(value='same')], right=[dict(value='same')])
        _, identities = value_proof(source)
        clone = copy.deepcopy(source)
        clone['left'][0] = source['right'][0]
        with self.assertRaisesRegex(ValueError, 'shares a mutable source'):
            value_proof(clone, identities)

    def test_complete_owned_copy_preserves_proof_and_counts_every_container(self):
        source = dict(a=[dict(b='山\x00🌊'), []], z=-0.0)
        expected, identities = value_proof(source)
        actual, owned = value_proof(copy.deepcopy(source), identities)
        self.assertEqual(expected, actual)
        self.assertEqual(len(identities), 4)
        self.assertEqual(len(owned), 4)
        self.assertFalse(identities & owned)

    def test_non_json_value_is_not_silently_coerced(self):
        with self.assertRaisesRegex(ValueError, 'non-JSON'):
            value_proof(dict(invalid=object()))


if __name__ == '__main__':
    unittest.main()
