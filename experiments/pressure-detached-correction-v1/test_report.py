"""Scientific gate fixtures: a good aggregate must not hide one failed case."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReportGates(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'runs'
        self.output = self.directory / 'cpu'
        self.reports = self.root / 'reports'
        self.reports.mkdir()
        self.patches = [patch.object(evaluate, 'ROOT', self.root), patch.object(evaluate, 'DIRECTORY', self.directory),
                        patch.object(evaluate, 'REPORTS', self.reports)]
        for change in self.patches:
            change.start()
            self.addCleanup(change.stop)
        self.campaign = dict(rows=[dict(key=initial.lower()+'-'+mode.lower(), initialization=initial, mode=mode)
                                  for initial in ('C53', 'C44', 'C37') for mode in ('U', 'A', 'D')])
        for name in ('material.json', 'evaluation-config.json', 'diagnostic-config.json'):
            write(self.output / name, {})
        for identifier in ['C53', 'C44', 'C37'] + [r['key'] for r in self.campaign['rows']]:
            baseline = identifier.startswith('C')
            values = {1: .2, 2: .03, 4: 1e-4, 8: 1e-8} if baseline else {1: .1, 2: .02, 4: 1e-8, 8: 1e-9}
            summary = [dict(passes=n, cases=128, equation_passes=128 if v<=1e-6 else 0,
                            worst_equation_residual=v, median_equation_residual=v) for n,v in values.items()]
            write(self.output / identifier / 'snapshots.json', dict(summary=summary))
            write(self.output / (identifier+'-fitted.json'), dict(
                summary=[dict(r, cases=8, equation_passes=8 if values[r['passes']]<=1e-6 else 0) for r in summary],
                measurements=[dict(case=c, passes=n, relative_equation_residual=v) for c in range(8) for n,v in values.items()]))
            write(self.output / (identifier+'-diagnosis.json'), dict(rows=[dict(passes=8,
                relative_equation_residual=1e-8 if c<230 else 1e-4, low_mode_pressure_error_fraction=.9) for c in range(240)]))
            if not baseline:
                path = self.directory / (identifier+'-study-01')
                write(path / 'receipt.json', dict(optimizer_updates=1600, elapsed_s=250,
                    training_receipt=dict(elapsed_s=80, runtime=dict(peak_rss_bytes=100), execution=dict(peak_allocated_bytes=20))))
                write(path / 'restore/history.json', dict(training_started_unix=100, training_finished_unix=180,
                    optimizer_work=dict(network_forwards=6400 if identifier.endswith('-u') else 8000)))

    def result(self):
        evaluate.report(self.campaign, self.output)
        return json.loads((self.reports / 'results.json').read_text())

    def test_all_nine_groups_and_reference_evidence_are_retained(self):
        result = self.result()
        self.assertEqual(len(result['rows']), 9)
        self.assertEqual(len(result['paired']), 9)
        self.assertEqual(result['actual_optimizer_updates'], 14400)
        self.assertEqual(result['actual_network_forwards'], 67200)
        self.assertEqual(len(result['eligible_for_independent_scenes']), 9)
        self.assertIn('runs/cpu/C37/snapshots.json', result['evidence'])
        self.assertIn('runs/cpu/C53-diagnosis.json', result['evidence'])

    def test_one_early_fitting_regression_fails_capacity(self):
        path = self.output / 'c37-d-fitted.json'
        data = json.loads(path.read_text())
        next(row for row in data['measurements'] if row['case']==7 and row['passes']==1)['relative_equation_residual'] = .21
        write(path, data)
        result = self.result()
        row = next(row for row in result['rows'] if row['key']=='c37-d')
        self.assertFalse(row['capacity_checks']['one_pass_no_case_worse'])
        self.assertFalse(row['capacity_gate_passed'])
        self.assertNotIn('c37-d', result['eligible_for_independent_scenes'])
        self.assertEqual(len(result['rows']), 9)

    def test_real_and_manufactured_gates_remain_separate_from_capacity(self):
        path = self.output / 'c53-d/snapshots.json'
        data = json.loads(path.read_text())
        next(row for row in data['summary'] if row['passes']==4)['equation_passes'] = 127
        write(path, data)
        path = self.output / 'c53-d-diagnosis.json'
        data = json.loads(path.read_text())
        data['rows'][229]['relative_equation_residual'] = 1e-4
        write(path, data)
        result = self.result()
        row = next(row for row in result['rows'] if row['key']=='c53-d')
        self.assertTrue(row['capacity_gate_passed'])
        self.assertFalse(row['regression_checks']['four_pass_all_128'])
        self.assertFalse(row['regression_checks']['manufactured_eight_pass_no_worse_than_C53'])
        self.assertFalse(row['eligible_for_independent_scenes'])


class WorkerLifecycle(unittest.TestCase):
    def test_failed_or_expired_workers_are_reaped_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(evaluate, 'ROOT', root):
                records = []
                with self.assertRaises(RuntimeError):
                    evaluate.run_worker([sys.executable, '-c', 'raise SystemExit(7)'],
                                        root/'failed.log', None, records)
                with self.assertRaises(subprocess.TimeoutExpired):
                    evaluate.run_worker([sys.executable, '-c', 'import time; time.sleep(10)'],
                                        root/'expired.log', None, records, timeout=.01)
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0]['returncode'], 7)
                self.assertTrue(all(row['reaped'] and row['returncode'] is not None for row in records))


if __name__ == '__main__':
    unittest.main()
