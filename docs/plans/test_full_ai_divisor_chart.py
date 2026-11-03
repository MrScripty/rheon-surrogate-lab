"""Recorded learned-divisor accuracy, absence of timing, and untimed work."""
import json
from pathlib import Path
import unittest

from dashboard_build import DATA

ROOT=Path(__file__).resolve().parent


class DivisorChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']

    def test_all_scored_states_and_new_fresh_population(self):
        rows=[r for r in self.study['temporal'] if r.get('population','').startswith('divisor_')]
        self.assertEqual(len(rows),48)
        for population,scenes,states in [('divisor_consumed',72,69120),('divisor_fresh',16,15360)]:
            chosen=[r for r in rows if r['population']==population]
            self.assertEqual(len(chosen),24)
            value=json.loads((ROOT/chosen[0]['source']).read_text())
            self.assertEqual(len(value['rows']),states)
            self.assertEqual(len({r['identifier'] for r in value['rows']}),scenes)
            self.assertEqual(value['fresh_holdout_consumed'],population=='divisor_fresh')
            for row in chosen:
                selected=[r for r in value['rows'] if r['horizons']==row['horizons'] and ('small' if max(r['counts_xyz'])<=16 else '32cubed')==row['regime'] and r['step']<=int(row['interval'].split('_')[1])]
                self.assertEqual(row['scored_states'],len(selected))
                self.assertEqual(row['field_worst'],{f:max(r['metrics']['relative'][f] for r in selected) for f in row['field_worst']})
                self.assertTrue(row['accuracy_pass'])
                self.assertTrue(all(r['metrics']['all_accuracy_criteria_pass'] for r in selected))
        models=[m for m in self.study['temporal_executions'] if m['label'].endswith('· learned divisor')]
        self.assertEqual(len(models),4)
        self.assertTrue(all(m['optimizer_updates']==0 for m in models))
        self.assertFalse(any(t['identifier'] in {m['label'] for m in models} for t in self.study['temporal_timing']))
        self.assertFalse(self.study['divisor_result']['latency_qualified'])

    def test_untimed_work_and_receipt_only_readiness(self):
        work=self.study['divisor_work']
        self.assertFalse(work['latency_claim'])
        self.assertFalse(work['clocks_read'])
        raw=json.loads((ROOT/work['source']).read_text())
        self.assertEqual(len(work['rows']),8)
        self.assertEqual({r['horizons'] for r in work['rows']},{1,4})
        for row,original in zip(work['rows'],raw['rows']):
            self.assertTrue(original['complete_output_bitwise_parity'])
            for stage,metrics in row['stages'].items():
                for field,value in metrics.items():self.assertEqual(value,sum(op[field] for op in original['operations'] if op['stage']==stage))
        readiness=self.study['expanded_readiness']
        source=json.loads((ROOT/readiness['source']).read_text())
        self.assertEqual(readiness['completed']['train'],len({r['identifier'] for r in source['rows'] if r['split']=='train'}))
        self.assertEqual(readiness['completed']['reserved'],0)
        self.assertFalse(readiness['field_archives_reverified'])
        self.assertFalse(source['field_archives_opened'])


if __name__=='__main__':unittest.main()
