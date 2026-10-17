"""Reject extra scientific changes and unqualified custody changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from compare import configuration, validator_cutover


class ComparisonTest(unittest.TestCase):
    def test_actual_configuration_allows_only_new_weight(self):
        source = json.loads((Path(__file__).parents[1] / 'pressure-current-policy-v1/configs/late.json').read_text())
        candidate = copy.deepcopy(source)
        candidate['id'] = 'candidate'
        candidate['current_feedback']['weight'] = 0.03
        configuration(source, candidate, 0.03)
        for key, value in [('band_feedback', {'weight': 0.01}),
                           ('training', {**source['training'], 'updates': 4999})]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'only declared'):
                configuration(source, {**candidate, key: value}, 0.03)

    def test_custody_cutover_cannot_change_receipt_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, candidate, owner = [Path(folder) / name for name in ('before', 'after', 'owner')]
            reference.write_text('def _outputs(value): return value\ndef validate_receipt(value): return value\n')
            candidate.write_text('def _outputs(value): return value + 1\ndef validate_receipt(value): return value\n')
            owner.write_bytes(candidate.read_bytes())
            validator_cutover(reference, candidate, owner)
            candidate.write_text('def _outputs(value): return value + 1\ndef validate_receipt(value): return None\n')
            owner.write_bytes(candidate.read_bytes())
            with self.assertRaisesRegex(ValueError, 'exceeds the qualified'):
                validator_cutover(reference, candidate, owner)

if __name__ == '__main__':
    unittest.main()
