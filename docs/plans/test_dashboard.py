"""Scientific invariants at the dashboard's measurement and rendering seam."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from dashboard_build import DATA, render_dashboard
from evaluation_measurements import measurement_catalog

ROOT = Path(__file__).resolve().parent


class MeasurementTests(unittest.TestCase):
    def catalog(self, repeats=5):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / 'pressure-compiled-timing-01/summary.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(status='completed', threads=1,
                warmup='Whole population each repeat', rows=[dict(repeats=repeats)])))
            row = dict(model='C107', identifier='C107/original',
                       timing_source=str(path.relative_to(base)), regime='historical_small',
                       median_ms=4, worst_error=.001, min_repeat_ms=3.9, max_repeat_ms=4.2)
            data = dict(cohort_points=[dict(id='C107')],
                        full_ai_study=dict(pressure=[row]))
            original = deepcopy(data)
            catalog = measurement_catalog(data, base)
            self.assertEqual(data, original)
            self.assertEqual(catalog['benchmarks'][0]['source_sha256'],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            return catalog

    def test_ranking_requires_repeated_warm_capture(self):
        self.assertTrue(self.catalog()['benchmarks'][0]['ranking_eligible'])
        self.assertFalse(self.catalog(repeats=1)['benchmarks'][0]['ranking_eligible'])

    def test_training_device_cannot_create_gpu_timings(self):
        catalog = self.catalog()
        self.assertFalse(catalog['gpu_timings_available'])
        self.assertEqual(catalog['benchmarks'][0]['device'], 'cpu')

    def test_comparison_rules_through_shared_javascript_interface(self):
        script = r"""
const assert=require('node:assert/strict');
const {analyze,pressurePoints}=require('./dashboard/measurements.js');
const point=(id,error,time,key='common')=>({id,worst_error:error,inference_median_ms:time,comparison_population:key,ranking_eligible:true});
const a=point('fast',4,1),b=point('balanced',1,2),c=point('dominated',2,3);
assert.equal(analyze([a,b,c]).ranked[0].id,'balanced');
assert.deepEqual(analyze([a,b,c]).frontier.map(p=>p.id),['fast','balanced']);
assert.equal(analyze([a,point('other',.01,.01,'different capture')]).ranked.length,0);
assert.equal(analyze([a,{...b,ranking_eligible:false}]).ranked.length,0);
assert.equal(analyze([a,point('missing',0,1)]).ranked.length,1);
assert.equal(analyze([a,b],0).ranked[0].id,'fast');
assert.equal(analyze([a,b],1).ranked[0].id,'balanced');
const cohort={id:'C1',cases:128,timed_cases:128,best_error:.001,best_error_raw:1e-20,timing_cases:[{}]};
const data={cohort_points:[cohort],solver_comparison:{classification:{C1:{kind:'full'}}},evaluation:{benchmarks:[{id:'fresh',measurements:[{cohort:'C1',execution:'original',historical_population:false,worst_error:.1,inference_median_ms:2}]}]}};
assert.equal(pressurePoints(data,'fresh')[0].best_error,null);
assert.equal(pressurePoints(data,'fresh')[0].cases,null);
assert.equal(pressurePoints(data,'fresh')[0].timed_cases,null);
assert.equal(pressurePoints(data,'historical')[0].ranking_eligible,false);
assert.equal(cohort.best_error,.001);
"""
        subprocess.run(['node', '-e', script], cwd=ROOT, check=True, capture_output=True)

    def test_render_is_idempotent_and_retains_scientific_data(self):
        text = (ROOT / 'cohort-progress.html').read_text()
        before = json.loads(DATA.search(text)[2])
        rendered = render_dashboard(text, ROOT)
        after = json.loads(DATA.search(rendered)[2])
        self.assertEqual({k: v for k, v in before.items() if k != 'evaluation'},
                         {k: v for k, v in after.items() if k != 'evaluation'})
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         hashlib.sha256(rendered.encode()).hexdigest())
        self.assertNotIn('@@', rendered)
        self.assertEqual(after['evaluation']['default_benchmark'],
                         'pressure-compiled-timing-01/historical_small')


if __name__ == '__main__':
    unittest.main()
