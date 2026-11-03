"""Population and reference-only invariants for the latest chart additions."""
import json
from pathlib import Path
import unittest

from dashboard_build import DATA

ROOT=Path(__file__).resolve().parent


class HistoryChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=json.loads(DATA.search((ROOT/'cohort-progress.html').read_text())[2])['full_ai_study']
        cls.cache={}

    def read(self,source):
        if source not in self.cache:self.cache[source]=json.loads((ROOT/source).read_text())
        return self.cache[source]

    def test_recorded_populations_and_matched_timing_subset(self):
        rows=[r for r in self.study['temporal'] if r.get('population','').startswith(('history_','force_')) or r.get('scene_identifiers')]
        self.assertEqual(len(rows),182)
        self.assertEqual(len([r for r in rows if r['population']=='history_fresh']),32)
        self.assertEqual(len([r for r in rows if r['population']=='force_fresh']),24)
        self.assertTrue(all(r['scenes']==30 if r['regime']=='small' else r['scenes']==10 for r in rows if r['population']=='history_consumed'))
        for row in rows:
            value=self.read(row['source'])
            selected=[r for r in value['rows'] if r['horizons']==row['horizons'] and ('small' if max(r['counts_xyz'])<=16 else '32cubed')==row['regime']]
            if row.get('scene_identifiers'):selected=[r for r in selected if r['identifier'] in row['scene_identifiers']]
            if row['interval']!='single_block':selected=[r for r in selected if r['step']<=int(row['interval'].split('_')[1])]
            self.assertEqual(row['scored_states'],len(selected))
            self.assertEqual(row['scenes'],len({r['identifier'] for r in selected}))
            self.assertEqual(row['field_worst'],{f:max(r['metrics']['relative'][f] for r in selected) for f in row['field_worst']})
            self.assertEqual(row['accuracy_pass'],all(r['metrics']['all_accuracy_criteria_pass'] for r in selected))
            self.assertFalse(value.get('numerical_solver_in_candidate',False))
            self.assertFalse(value.get('numerical_pressure_oracle_used',False))
        history_times=[t for t in self.study['temporal_timing'] if '· pressure history' in t['identifier']]
        self.assertEqual(len(history_times),144)
        self.assertTrue(all(t['population']=='development' for t in history_times))
        self.assertFalse(any(t.get('population','').startswith(('history_','force_')) for t in self.study['temporal_timing']))
        self.assertTrue(any(not r['accuracy_pass'] for r in rows if r['population']=='history_fresh'))
        self.assertTrue(any(not r['accuracy_pass'] for r in rows if r['population']=='force_fresh'))

    def test_frequency_values_are_native_reference_comparisons(self):
        pilot=self.study['reference_frequency']
        self.assertTrue(pilot['reference_only'])
        self.assertFalse(pilot['temporal_convergence_qualified'])
        raw=self.read(pilot['raw_source'])
        self.assertTrue(raw['reference_only'])
        self.assertFalse(raw['model_targets_verified'])
        self.assertEqual(len(pilot['rows']),64)
        for row in pilot['rows']:
            source=next(t for t in raw['trials'] if t['identifier']==row['identifier'] and t['substeps']==row['substeps'])
            for key,value in row.items():self.assertEqual(source[key],value)
            if row['substeps']==64:self.assertTrue(all(v==0 for v in row['worst_relative_linf_to_finest_peak'].values()))
        self.assertEqual({r['force_frequency_parameter_hz'] for r in pilot['rows']},{.1,1.,10.,100.})


if __name__=='__main__':unittest.main()
