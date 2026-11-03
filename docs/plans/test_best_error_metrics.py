"""Check population selection and provenance for reported best-case scores."""
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import tempfile
import unittest

from best_error_metrics import enrich_best_error


class BestErrorTests(unittest.TestCase):
    def dataset(self, base, grids, errors, name='C95'):
        cases = [dict(trajectory_id=f'simulation-{i}', frame=i, counts_xyz=counts,
                      raw_learned=dict(quality=dict(relative_residual=error)))
                 for i, (counts, error) in enumerate(zip(grids, errors))]
        source = name+'.jsonl'
        raw = ''.join(json.dumps(c)+'\n' for c in cases).encode()
        (base/source).write_bytes(raw)
        return dict(cohort_points=[dict(id=name, source=source, weights_sha256=name,
                        best_error=min(errors), median_error=statistics.median(errors),
                        worst_error=max(errors), cases=len(cases), inference_median_ms=10)],
                    history_sources={source:hashlib.sha256(raw).hexdigest()})

    def test_same_population_rule_for_every_method_and_repeated_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a = self.dataset(base, [[5,6,5],[7,6,5],[16,16,16]], [1e-16,1e-4,1e-3])
            b = self.dataset(base, [[5,6,5],[7,6,5],[16,16,16]], [1e-8,2e-4,2e-3], 'C53')
            a['cohort_points'] += b['cohort_points']
            a['history_sources'].update(b['history_sources'])
            originals = copy.deepcopy(a['cohort_points'])
            enrich_best_error(a, base)
            for p, original in zip(a['cohort_points'], originals):
                self.assertEqual(p['best_error_eligible_cases'], 2)
                self.assertEqual(p['best_error_excluded_cases'], 1)
                self.assertGreater(p['best_error'], original['best_error'])
                self.assertEqual(p['best_error_raw'], original['best_error'])
                for key in ['median_error','worst_error','cases','inference_median_ms']:
                    self.assertEqual(p[key], original[key])
            once = copy.deepcopy(a)
            enrich_best_error(a, base)
            self.assertEqual(a, once)

    def test_eligibility_uses_axis_dimensions_without_an_error_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = self.dataset(base, [[6,6,6],[7,1,1],[5,6,7]], [1e-30,1e-20,1e-19])
            enrich_best_error(data, base)
            point = data['cohort_points'][0]
            self.assertEqual(point['best_error'], 1e-20)
            self.assertEqual(point['best_error_case']['counts_xyz'], [7,1,1])
            self.assertEqual(point['best_error_excluded_cases'], 1)

    def test_changed_measurements_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = self.dataset(base, [[5,6,5],[7,6,5]], [1e-16,1e-4])
            (base/'C95.jsonl').write_bytes((base/'C95.jsonl').read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'measurement source changed'):
                enrich_best_error(data, base)

    def test_retained_c95_case_replaces_roundoff_minimum(self):
        base = Path(__file__).resolve().parent
        page = (base/'cohort-progress.html').read_text()
        data = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>',page,re.S)[1])
        point = next(p for p in data['cohort_points'] if p['id']=='C95')
        original = copy.deepcopy(point)
        enrich_best_error(data, base)
        self.assertAlmostEqual(point['best_error']/1.6580110395586821e-6, 1)
        self.assertEqual(point['best_error_raw'], 3.8009727327381634e-16)
        self.assertEqual(point['best_error_eligible_cases'], 126)
        self.assertEqual(point['best_error_excluded_cases'], 2)
        self.assertEqual(point['worst_error'], original['worst_error'])


if __name__ == '__main__':
    unittest.main()
