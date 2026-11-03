"""Real nested temporal closures preserve full paths and reject omitted assets."""
import copy
from pathlib import Path
import sys
import unittest

from captured_temporal_review import verified_review_receipt
from review_water_learning_sequence import read

ROOT = Path(__file__).resolve().parents[2]


class ProjectRelativeClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = ROOT/'docs/plans/model-portfolio-migration-v1/reports/captured-coupled-temporal-generator-method-reviews-01.json'
        cls.row = read(report)['rows'][0]
        cls.asset = next(s for s in cls.row['sources'] if '/expert-assets/pressure/' in s['path'])

    def test_complete_original_closure_returns_exact_receipt(self):
        expected = (ROOT/self.row['original_receipt']['path']).read_bytes()
        self.assertEqual(verified_review_receipt(self.row),expected)
        self.assertNotIn('torch',sys.modules)

    def test_missing_nested_asset_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['sources'].remove(self.asset)
        with self.assertRaisesRegex(ValueError,'^Complete project-relative source closure required$'):
            verified_review_receipt(row)

    def test_duplicate_closure_key_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['sources'].append(dict(self.asset))
        with self.assertRaisesRegex(ValueError,'^Duplicate temporal source closure key$'):
            verified_review_receipt(row)

    def test_misdeclared_asset_digest_is_rejected(self):
        row = copy.deepcopy(self.row)
        next(s for s in row['sources'] if s['path'] == self.asset['path'])['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'^Temporal source path/bytes differ$'):
            verified_review_receipt(row)


if __name__ == '__main__':
    unittest.main()
